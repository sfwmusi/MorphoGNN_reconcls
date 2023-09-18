import os
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt


error_list = []
def sizeRatio(root_path):
    dir_path = os.listdir(root_path)
    print(dir_path)
    for dir_name in dir_path:
        width_list = []
        height_list = []
        depth_list = []
        name_list = []
        img_path = os.path.join(root_path, dir_name)
        img_name = os.listdir(img_path)
        print(dir_name)
        for name in img_name:
            image_path = os.path.join(img_path, name)
            # print(image)
            image = tiff.imread(image_path)
            # print(image.shape)
            width = image.shape[2]
            height = image.shape[1]
            depth = image.shape[0]
            # print('width:', width, ' height:', height, ' depth:', depth)
            width_list.append(width)
            height_list.append(height)
            depth_list.append(depth)
            name_list.append(int(name.split('.')[0]))

        # print(name_list,type(name_list))
        width_arr = np.array(width_list).reshape(-1, 1)
        height_arr = np.array(height_list).reshape(-1, 1)
        depth_arr = np.array(depth_list).reshape(-1, 1)
        name_arr = np.array(name_list).reshape(-1, 1)
        size_arr = np.concatenate((width_arr, height_arr, depth_arr, name_arr), axis=1)
        print(size_arr, size_arr.shape)

        save_arr, index_arr = np.unique(size_arr[:, :3], axis=0, return_index=True)
        # print(save_arr)
        print(sorted(size_arr[index_arr,-1]))
        np.savetxt("%s.txt"%dir_name,sorted(size_arr[index_arr,-1]),fmt='%d')

    # # visual
    # x = np.array(width_list)
    # y = np.array(height_list)
    # z = np.array(depth_list)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x, y, z, 'blue')
    # ax.set_xlabel('width')
    # ax.set_ylabel('height')
    # ax.set_zlabel('depth')
    # plt.show()
    return width_list, height_list, depth_list, name_list

def imgResize(root_path, dir_path,save_root_path, thresholds=300):
    '''
    阈值后crop到最大有数据的形状
    '''

    img_path = os.path.join(root_path,dir_path,'image')
    img_name = os.listdir(img_path)
    print(img_name)
    for name in img_name:
        image_path = os.path.join(img_path, name)
        # print(image)
        # print(image_path)
        image = tiff.imread(image_path)

        image_old = image.copy()

        image[image<thresholds] = np.uint16(0)
        depth = image.sum(axis=1).sum(axis=1)
        depth_begin = (depth!=0).argmax(axis=0)
        depth_end = len(depth)-(depth[::-1]!=0).argmax(axis=0)-1

        width = image.sum(axis=0).sum(axis=1)
        width_begin = (width!=0).argmax(axis=0)
        width_end = len(width)-(width[::-1]!=0).argmax(axis=0)-1

        height = image.sum(axis=0).sum(axis=0)
        height_begin = (height != 0).argmax(axis=0)
        height_end = len(height) - (height[::-1] != 0).argmax(axis=0) - 1
        # image_old new2在原始图上截取; image new在处理后的图上截取
        image_crop = image_old[depth_begin:depth_end,width_begin:width_end,height_begin:height_end]

        save_dir_path = os.path.join(save_root_path,dir_path)
        # print(save_dir_path)
        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)
        save_image_path = os.path.join(save_dir_path,name)
        if image_crop.size>0:
            tiff.imsave(save_image_path,image_crop)
        else:
            error_list.append(image_path)
            print('empty array error')

def labelUpdate(root_path,dir_path):
    pass





if __name__ == '__main__':
    root_path = r'D:\yaogang\datasets\194511'
    dir_path = os.listdir(root_path)
    dir_path.sort(key=lambda x: int(x))
    print(dir_path)
    # imgResize(root_path,dir_path[0])



    ## visual
    # width_list, height_list, depth_list, name_list = sizeRatio(root_path)
    # x = width_arr - np.mean(width_arr,axis=0)
    # y = height_arr - np.mean(height_arr,axis=0)
    # z = depth_arr - np.mean(depth_arr,axis=0)
    ## 计算size平均大小
    # print(np.mean(width_arr,axis=0),np.mean(height_arr,axis=0),np.mean(depth_arr,axis=0))
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x, y, z, 'blue')
    # ax.set_xlabel('width')
    # ax.set_ylabel('height')
    # ax.set_zlabel('depth')
    # plt.show()

    # crop size
    save_root_path = r'D:\yaogang\datasets\new_1\194511'
    for path in dir_path:
        imgResize(root_path,path,save_root_path)
    print(error_list)

