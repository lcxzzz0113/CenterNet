import os
from PIL import Image
import matplotlib.pyplot as plt 
import numpy as np 
# from scipy.misc import imread   #numpy, RGB, 0~255
import cv2, os, argparse
from tqdm import tqdm

def cal_mean_and_std(file_path, shape):
    w, h = shape
    pathDir = os.listdir(file_path)

    R_channel = 0
    G_channel = 0
    B_channel = 0
    for idx in range(len(pathDir)):
        filename = pathDir[idx]
        img = imread(os.path.join(file_path, filename)) / 255.0
        R_channel = R_channel + np.sum(img[:, :, 0])
        G_channel = G_channel + np.sum(img[:, :, 1])
        B_channel = B_channel + np.sum(img[:, :, 2])
    num = len(pathDir) * w * h
    R_mean = R_channel / num
    G_mean = G_channel / num
    B_mean = B_channel / num

    R_channel = 0
    G_channel = 0
    B_channel = 0
    for idx in range(len(pathDir)):
        filename = pathDir[idx]
        img = imread(os.path.join(file_path, filename)) / 255.0
        R_channel = R_channel + np.sum((img[:, :, 0] - R_mean) ** 2)
        G_channel = G_channel + np.sum((img[:, :, 1] - G_mean) ** 2)
        B_channel = B_channel + np.sum((img[:, :, 2] - B_mean) ** 2)
 
    R_var = np.sqrt(R_channel / num)
    G_var = np.sqrt(G_channel / num)
    B_var = np.sqrt(B_channel / num)
    print((R_mean, G_mean, B_mean), (R_var, G_var, B_var))
    return (R_mean, G_mean, B_mean), (R_var, G_var, B_var)



def main():
    dirs = '/mnt/data-1/data/lcx/CenterNet-master/data/Water/images' 
    img_file_names = os.listdir(dirs)
    m_list, s_list = [], []
    for img_filename in tqdm(img_file_names):
        img = cv2.imread(dirs + '/' + img_filename)
        img = img / 255.0
        m, s = cv2.meanStdDev(img)
        m_list.append(m.reshape((3,)))
        s_list.append(s.reshape((3,)))
    m_array = np.array(m_list)
    s_array = np.array(s_list)
    m = m_array.mean(axis=0, keepdims=True)
    s = s_array.mean(axis=0, keepdims=True)
    print("mean = ", m[0][::-1])
    print("std = ", s[0][::-1])


# if __name__ == '__main__':   
#     # data_path = '/home/users/chenxin.lu/SeaShips/JPEGImages'
#     # data_path = '/mnt/data-1/data/lcx/CenterNet-master/data/Water/images'
#     # cal_mean_and_std(data_path, (1920, 1080))
#     main()

import torch
print(torch.hub.list('pytorch/vision'))