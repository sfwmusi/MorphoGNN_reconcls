#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM

Modified by 
@Author: An Tao, Pengliang Ji
@Contact: ta19@mails.tsinghua.edu.cn, jpl1723@buaa.edu.cn
@Time: 2021/7/20 7:49 PM
"""


import os
import sys
import glob
import h5py
import numpy as np
import torch
import json
# import cv2
from torch.utils.data import Dataset
import tifffile as tiff


def download_modelnet40():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % ('modelnet40_ply_hdf5_2048', DATA_DIR))
        os.system('rm %s' % (zipfile))


def download_shapenetpart():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data')):
        www = 'https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % ('hdf5_data', os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data')))
        os.system('rm %s' % (zipfile))


def download_S3DIS():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data')):
        www = 'https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % ('indoor3d_sem_seg_hdf5_data', DATA_DIR))
        os.system('rm %s' % (zipfile))
    if not os.path.exists(os.path.join(DATA_DIR, 'Stanford3dDataset_v1.2_Aligned_Version')):
        if not os.path.exists(os.path.join(DATA_DIR, 'Stanford3dDataset_v1.2_Aligned_Version.zip')):
            print('Please download Stanford3dDataset_v1.2_Aligned_Version.zip \
                from https://goo.gl/forms/4SoGp4KtH1jfRqEj2 and place it under data/')
            sys.exit(0)
        else:
            zippath = os.path.join(DATA_DIR, 'Stanford3dDataset_v1.2_Aligned_Version.zip')
            os.system('unzip %s' % (zippath))
            os.system('mv %s %s' % ('Stanford3dDataset_v1.2_Aligned_Version', DATA_DIR))
            os.system('rm %s' % (zippath))


def load_data_cls_img(partition):
    # download_modelnet40()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    # 增加图像
    all_image = []
    all_name = []
    print('---------------------------------')
    print(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048_new1', '*%s*.h5'%partition))
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048_new1', '*%s*.h5'%partition)):
        print('测试',h5_name)
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        image = f['image'][:].astype('float32')/255.0
        name = f['name'][:].astype('str')
        f.close()
        all_data.append(data)
        all_label.append(label)
        all_image.append(image)
        all_name.append(name)
        #####
        # print(all_label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_image = np.concatenate(all_image, axis=0)
    all_name = np.concatenate(all_name, axis=0)
    print(type(all_data))
    print(np.array(all_data).shape)
    print('image size: ',np.array(all_image).shape)
    print('name size: ',np.array(all_name).shape)

    # tiff.imsave(r'C:\Users\yaogang\Desktop\66.tif',all_image[0])
    return all_data, all_label, all_image


def load_data_cls_img_K(partition,idname,return_name=False):
    # download_modelnet40()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    # 增加图像
    all_image = []
    all_name = []
    print('---------------------------------')
    # print(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', '*%s*%s.h5'%(partition,str(idname))))
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048_new1', '*%s*_%s.h5'%(partition,str(idname)))):
        print('测试',h5_name)
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        image = f['image'][:].astype('float32')/255.0
        name = f['name'][:].astype('str')
        f.close()
        all_data.append(data)
        all_label.append(label)
        all_image.append(image)
        name = np.char.add(h5_name, name[:])
        all_name.append(name)
        #####
        # print(all_label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_image = np.concatenate(all_image, axis=0)
    all_name = np.concatenate(all_name, axis=0)
    print(type(all_data))
    print(np.array(all_data).shape)
    print('image size: ',np.array(all_image).shape)
    print('name size: ',np.array(all_name).shape)

    np.savetxt('name.txt', all_name,delimiter=' ', fmt='%s')


    return all_data, all_label, all_image, all_name

def load_data_cls_img_K_test(partition,idname=1):
    # download_modelnet40()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    # 增加图像
    all_image = []
    all_name = []
    print('---------------------------------')
    # print(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', '*%s*%s.h5'%(partition,str(idname))))
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'image_size_128_128_128/test', '*%s*_%s.h5'%(partition,str(idname)))):
        print('测试',h5_name)
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        image = f['image'][:].astype('float32')/255.0
        name = f['name'][:].astype('str')
        f.close()
        all_data.append(data)
        all_label.append(label)
        all_image.append(image)
        name = np.char.add(h5_name, name[:])
        all_name.append(name)
        #####
        # print(all_label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_image = np.concatenate(all_image, axis=0)
    all_name = np.concatenate(all_name, axis=0)
    print(type(all_data))
    print(np.array(all_data).shape)
    print('image size: ',np.array(all_image).shape)
    print('name size: ',np.array(all_name).shape)

    np.savetxt('name.txt', all_name, delimiter=' ', fmt='%s')
    # print('save')
    # tiff.imsave(r'C:\Users\yaogang\Desktop\66.tif',all_image[0])
    return all_data, all_label, all_image


def load_data_cls(partition):
    # download_modelnet40()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    print('---------------------------------')
    print(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', '*%s*.h5'%partition))
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', '*%s*.h5'%partition)):
        print('测试',h5_name)
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)

        #####
        # print(all_label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)

    print(type(all_data))
    print(np.array(all_data).shape)

    # tiff.imsave(r'C:\Users\yaogang\Desktop\66.tif',all_image[0])
    return all_data, all_label





def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud):
    theta = np.pi*2 * np.random.uniform()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    pointcloud[:,[0,2]] = pointcloud[:,[0,2]].dot(rotation_matrix) # random rotation (x,z)
    return pointcloud


class MultiNetData(Dataset):
    def __init__(self, num_points, partition='train',id_name=1):
        # self.data, self.label, self.image = load_data_cls_img(partition)
        self.return_name = True
        # self.data, self.label, self.image, self.name = load_data_cls_img_K(partition, id_name,self.return_name)
        # else:
        #     self.data, self.label, self.image = load_data_cls_img_K(partition,id_name)
        self.data, self.label, self.image = load_data_cls_img_K_test(partition, id_name)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        image = self.image[item]
        # name = self.name[item]
        # train
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        # if self.return_name == True:
        #     return pointcloud, label, image, name
        # else:
        return pointcloud, label, image

    def __len__(self):
        return self.data.shape[0]



class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data_cls(partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        # train
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

class ShapeNetPart(Dataset):
    def __init__(self, num_points, partition='train', class_choice=None):
        self.data, self.label, self.seg = load_data_partseg(partition)
        self.cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4, 
                       'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9, 
                       'motor': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
        self.seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
        self.index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
        self.num_points = num_points
        self.partition = partition        
        self.class_choice = class_choice
        self.partseg_colors = load_color_partseg()
        
        if self.class_choice != None:
            id_choice = self.cat2id[self.class_choice]
            indices = (self.label == id_choice).squeeze()
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.seg = self.seg[indices]
            self.seg_num_all = self.seg_num[id_choice]
            self.seg_start_index = self.index_start[id_choice]
        else:
            self.seg_num_all = 50
            self.seg_start_index = 0
            
      
    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        seg = self.seg[item][:self.num_points]
        if self.partition == 'trainval':
            # pointcloud = translate_pointcloud(pointcloud)
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]
        return pointcloud, label, seg

    def __len__(self):
        return self.data.shape[0]


class S3DIS(Dataset):
    def __init__(self, num_points=4096, partition='train', test_area='1'):
        self.data, self.seg = load_data_semseg(partition, test_area)
        self.num_points = num_points
        self.partition = partition    
        self.semseg_colors = load_color_semseg()

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        seg = self.seg[item][:self.num_points]
        if self.partition == 'train':
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]
        seg = torch.LongTensor(seg)
        return pointcloud, seg

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    train = ModelNet40(1024)
    test = ModelNet40(1024, 'test')
    data, label = train[0]
    print(data.shape)
    print(label.shape)


    # trainval = ShapeNetPart(2048, 'trainval')
    # test = ShapeNetPart(2048, 'test')
    # data, label, seg = trainval[0]
    # print(data.shape)
    # print(label.shape)
    # print(seg.shape)
    #
    # train = S3DIS(4096)
    # test = S3DIS(4096, 'test')
    # data, seg = train[0]
    # print(data.shape)
    # print(seg.shape)
