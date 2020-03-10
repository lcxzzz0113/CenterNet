from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import _init_paths

import os

import torch
import torch.utils.data
from lib.opts import opts
from lib.models.model import create_model, load_model, save_model
from lib.models.data_parallel import DataParallel
from lib.logger import Logger
from lib.datasets.dataset_factory import get_dataset
from lib.trains.train_factory import train_factory
from test import test
import math


def main(opt):
    torch.manual_seed(opt.seed)
    # benchmark=True 自动寻找最适合当前配置的高效算法，来达到优化运行效率的wento
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)
    
    logger = Logger(opt)

    def adjust_learning_rate(optimizer, epoch):
        # use warmup
        if epoch < 5:
            lr = opt.lr * ((epoch + 1) / 5)
        else:
        # use cosine lr
            PI = 3.14159
            lr = opt.lr * 0.5 * (1 + math.cos(epoch * PI / 20)) 
            # print(1111)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    
    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    # optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    optimizer = torch.optim.SGD(model.parameters(), opt.lr)
    start_epoch = 0
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)
            
    Trainer = train_factory[opt.task]
    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)
    
    print('Setting up data...')
    # num_worker=0是主进程读取; >0使用多进程读取，子进程读取数据时，训练程序会卡住，GPU utils为0，
    val_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'val'), 
        batch_size=1, 
        shuffle=False,
        num_workers=1,
        pin_memory=True)

    if opt.test:
        _, preds = trainer.val(0, val_loader)
        val_loader.dataset.run_eval(preds, opt.save_dir)
        return

    train_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'train'), 
        batch_size=opt.batch_size, 
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True)

    print('Starting training...')
    best = 1e10
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        adjust_learning_rate(optimizer, epoch)
        mark = epoch if opt.save_all else 'last'
        log_dict_train, _ = trainer.train(epoch, train_loader)
        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))
        if opt.val_intervals > 0 and epoch % 2 == 0 and epoch > 10:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                 epoch, model, optimizer)
            with torch.no_grad():
                log_dict_val, preds = trainer.val(epoch, val_loader)
            for k, v in log_dict_val.items():
                logger.scalar_summary('val_{}'.format(k), v, epoch)
                logger.write('{} {:8f} | '.format(k, v))
            if log_dict_val[opt.metric] < best:
                best = log_dict_val[opt.metric]
                save_model(os.path.join(opt.save_dir, 'model_best.pth'), 
                    epoch, model)
        
                # test(opt)
                # opt.model = None
        if epoch > 20:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
                epoch, model, optimizer)
        # elif 80 < epoch <=100 and epoch % 3 == 0:
        #     save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
        #         epoch, model, optimizer)
        logger.write('\n')
        # if epoch in opt.lr_step:
        #     save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
        #         epoch, model, optimizer)
        #     lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
        #     print('Drop LR to', lr)
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
        print('Epoch is Finished')
    logger.close()

if __name__ == '__main__':
    opt = opts().parse()
    main(opt)