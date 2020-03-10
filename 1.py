import numpy as np 
import torch 
from torch.autograd import Variable
import os
import xml.etree.ElementTree as ET
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt
import math

# a, b = np.meshgrid([1, 2, 3], [1, 2, 3, 4, 5], indexing='ij')
# a = a.flatten().reshape(1, 1, 3, 5).repeat(3, axis=1)
# b = b.flatten().reshape(1, 1, 3, 5).repeat(3, axis=1)
# c = np.concatenate((a, b), axis=1)
# print(c)

# N = 9
# offset = torch.arange(18).expand((2, 18, 10, 10))
# offsets_index = Variable(torch.cat([torch.arange(0, 2 * N, 2), \
#     torch.arange(1, 2 * N + 1, 2)]), requires_grad=False)
# print(offsets_index)
# offsets_index = offsets_index.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1).expand((2, 18, 10, 10))
# print(offsets_index)
# offset = torch.gather(offset, dim=1, index=offsets_index)
# print(offset)

# a = torch.ones(2, 3, 3, 4).lt(0.2) 
# print(a.detach())
# print(False + False)
# print(True + False)

# torch.nn.functional


# img_path = '/mnt/data-1/data/lcx/train/image'
# txt_path = '/mnt/data-1/data/lcx/train'
# imgs = os.listdir(img_path)
# train_imgs, val_imgs = imgs[:4000], imgs[4000:]
# sizs = set()
# for img in train_imgs:
#     siz = cv2.imread(os.path.join(img_path,img)).shape
#     sizs.add(siz)
# print(sizs)
    
# with open(os.path.join(txt_path, 'val.txt'), 'w') as f:
#     for img in train_imgs:
#         f.write(img[:-4] + '\n')

# xml_path = '/home/users/chenxin.lu/SeaShips/Annotations'
# xml_path = '/mnt/data-1/data/lcx/train/box'
xml_path = '/mnt/data-1/data/lcx/CenterNet-master/data/Water/box'
files = os.listdir(xml_path)
# print(files)
# name = set()
# img = []
label_name = ['holothurian', 'starfish', 'echinus', 'scallop']
label_num = defaultdict(int)
# box_area = defaultdict(int)
# box_ratio = defaultdict(int)
box = list()

for f in files:
    # h, w, c = cv2.imread(os.path.join('/mnt/data-1/data/lcx/CenterNet-master/data/Water/images', f[:-4] + '.jpg')).shape
    f_xml = os.path.join(xml_path, f)
    bndbox = dict()
    # print(f_xml)
    # try:
    tree = ET.parse(f_xml)
    root = tree.getroot()
    for elem in root:
        # for subelem in elem:
        #     if elem.tag == 'object' and subelem.tag == 'name':
        #         label_num[subelem.text] += 1
        for subelem in elem:
            bndbox['xmin'] = None
            bndbox['xmax'] = None
            bndbox['ymin'] = None
            bndbox['ymax'] = None
            if elem.tag == 'object' and subelem.tag == 'bndbox':
                for option in subelem:
                    bndbox[option.tag] = int(option.text)
                # if bndbox['xmin'] >= bndbox['xmax']:
                #     if bndbox['xmin'] >= 1:
                #         subelem[0].text = str(int(subelem[0].text) - 1)
                #     elif bndbox['xmin'] < 1 and bnd['xmax'] <= w - 1:
                #         subelem[2].text = str(int(subelem[2].text) + 1)
                # if bndbox['ymin'] >= bndbox['ymax']:
                #     if bndbox['ymin'] >= 1:
                #         subelem[1].text = str(int(subelem[1].text) - 1)
                #     elif bndbox['ymin'] < 1 and bnd['ymax'] <= h - 1:
                #         subelem[3].text = str(int(subelem[3].text) + 1)
                # if bndbox['ymax'] >= h:
                #     subelem[3].text = str(h - 1)
                # if bndbox['xmax'] >= w:
                #     subelem[2].text = str(w - 1)
                # area = sqrt((int(subelem[3].text) - int(subelem[1].text)) * (int(subelem[2].text) - int(subelem[0].text)))
                # ratio = (int(subelem[2].text) - int(subelem[0].text))/ (int(subelem[3].text) - int(subelem[1].text))
                area = math.sqrt((bndbox['ymax'] - bndbox['ymin']) * (bndbox['xmax'] - bndbox['xmin']))
                ratio = (bndbox['xmax'] - bndbox['xmin']) / (bndbox['ymax'] - bndbox['ymin']) 
                # print(area)
                box.append([ratio, area])
                # if subelem.text == 'waterweeds':
                    # img.append(i)
                    # root.remove(elem)  
                # name.add(subelem.text)
    # except:
    #     continue
    # tree.write(f_xml) 
# print(box)
fig, ax = plt.subplots()
ax.set_title('bndbox analysis')
ax.set_xlabel('ratio')
ax.set_ylabel('area')
plt.axis([0, 10, 0, 1500])
# ax.set_xtick([0, 0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# ax.set_ytick([0, 50, 100, 150, 200, 250, 300, ])
x = [a[0] for a in box]
y = [a[1] for a in box]
plot = plt.scatter(x, y, color='b', s=0.00001, marker='*')
plt.savefig('1.png', dpi=2000)


# c = box.copy()
# box.sort(key = lambda x:x[0])
# print('area list', box[0], box[-1])
# c.sort(key = lambda x:x[1])
# print('ratio list', c[0], c[-1])
# print(name)
# print(img)


# a1 = torch.zeros([3, 4])
# a2 = torch.ones([3, 4])
# a3 = torch.full([2, 3], 7)
# a4 = torch.tensor([1, 2.2])
# a5 = torch.FloatTensor(2, 3)
# a = np.array([1, 2])
# a5 = torch.from_numpy(a)
# a6 = torch.randperm(10)
# a7 = torch.rand(2, 3)
# a8 = torch.randm(2, 3)  # 标准正态分布
# a9 = torch.normal(mean=0.5, std=torch.arange(1, 6)) # 离散正态分布
# a10 = torch.linspace(2, 10, steps=7)
# a1 = torch.rand(2, 3, 4, 5)
# a1[1][2][3][4] = 1
# # print(a1)
# b = a1.eq(1).float()
# print(b)

# xml_path = '/home/users/chenxin.lu/SeaShips/Annotations'
# xml_path = '/mnt/data-1/data/lcx/CenterNet-master/data/Water/box'
# img_path = '/mnt/data-1/data/lcx/CenterNet-master/data/Water/images'
# files = os.listdir(xml_path)
# for f in files:
#     f_xml = os.path.join(xml_path, f)
#     bndbox = dict()
#     # print(f_xml)
#     tree = ET.parse(f_xml)
#     root = tree.getroot()
#     for elem in root:
#         # print(f[:-4])
#         image_path = '/mnt/data-1/data/lcx/CenterNet-master/data/gt/{}'.format(f[:-4] + '.jpg')
#         img = cv2.imread(image_path)
#         try:
#             img.shape
#         except:
#             img = cv2.imread(os.path.join(img_path, f[:-4] + '.jpg'))
#         for subelem in elem:
#             if elem.tag == 'object' and subelem.tag == 'name':
#                 label_name = subelem.text       
#         for subelem in elem:
#             bndbox['xmin'] = None
#             bndbox['xmax'] = None
#             bndbox['ymin'] = None
#             bndbox['ymax'] = None
#             if elem.tag == 'object' and subelem.tag == 'bndbox':
#                 for option in subelem:
#                     bndbox[option.tag] = int(option.text)
#                 # print(label_name)
#                 cv2.rectangle(img,
#                     (bndbox['xmin'], bndbox['ymin']),
#                     (bndbox['xmax'], bndbox['ymax']), (0, 0, 255), 1)
#                 cv2.putText(img, str(label_name), (bndbox['xmin'], bndbox['ymin'] - 2), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
#                 cv2.imwrite(image_path, img)



    