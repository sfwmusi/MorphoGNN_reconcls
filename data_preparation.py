import os
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import tifffile as tiff
import SimpleITK as sitk
from glob import glob
from skimage import transform, img_as_ubyte
from sklearn.model_selection import KFold
from imageSize import imgResize


# 均匀(下)采样 in:narray  out:list
def uni_sampling(swc, num):
    num_array = np.linspace(0, len(swc) - 1, num)
    num_array = np.round(num_array)
    num_list = num_array.tolist()
    num_list = [int(i) for i in num_list]
    # print('采样序号: ',num_list)
    # swc = np.array(swc)
    new_swc = swc[:][num_list][:]
    # print(new_swc,type(new_swc))
    new_swc = new_swc.tolist()
    return new_swc


# 获取分支
def getBranch(swc):
    '''神经元swc分支获取'''
    branch_list = []
    branch = []
    for j in range(len(swc)):
        if j==0:
            branch.append(j)
            continue
        if swc[j,6]==swc[j-1,0]:
            # print(swc[j,6])
            branch.append(j)
            if j == len(swc)-1:
                branch_list.append(branch)
        else:
            branch_list.append(branch)
            branch = []
            branch.append(int(swc[j,6])-1)
            branch.append(int(swc[j,0])-1)
            continue
    return branch_list


## 上采样,在原始的swc基础上进行 in:narray out:list
def uni_sampling_up(swc, num):
    if num > len(swc):
        upsamp_k = len(swc) - 1
        # print(upsamp_k)
        need_pnum = int((num - len(swc)) / upsamp_k) + 1
        # print(need_pnum)
        swc_branch = getBranch(swc)
        # print(swc_branch)
        upsamp_data = []
        for i in range(len(swc_branch)):
            for j in range(len(swc_branch[i]) - 1):
                swc_arr = swc[swc_branch[i][j]]
                next_swc_arr = swc[swc_branch[i][j + 1]]
                distance = np.linalg.norm(swc_arr - next_swc_arr)
                if distance < 6:

                    # print(swc_arr,next_swc_arr)

                    for k in range(need_pnum):
                        # 系数
                        alpha = (k + 1) / (need_pnum + 1)

                        upsamp_arr = swc_arr[2:5] - alpha * (swc_arr[2:5] - next_swc_arr[2:5])
                        upsamp_list = upsamp_arr.tolist()
                        upsamp_data.append(upsamp_list)
                # print(upsamp_data)
                # print(swc_arr[2:5])
        upsamp_swc = swc[:, 2:5].tolist()
        upsamp_swc = upsamp_swc + upsamp_data
        upsamp_swc = np.array(upsamp_swc)
        # print(upsamp_swc)
        swc_data = uni_sampling(upsamp_swc, num)
    else:
        swc_data = uni_sampling(swc[:, 2:5], num)
        print('无法上采样,swc数据点数大于需要上采样的点数')
    return swc_data


def samplingSWC(swc, file, ratio=0.15):
    '''神经元swc随机采样'''
    file_path = "re_" + file
    if os.path.exists(file_path):
        print('已存在')
        return 0
    num = len(swc)
    branch = getBranch(swc)
    # 删除节点的下一个节点[6]，改成该节点[6]
    # 分支起始点和终止点不能删除
    ignorePoint_begin = [i[0] for i in branch]
    ignorePoint_end = [i[-1] for i in branch]
    ignorePoint = ignorePoint_begin + ignorePoint_end
    print("忽略的点", ignorePoint)


    range_list = list(range(0, num-1))
    # 忽略分支点，即去除相同元素
    new_range = [x for x in range_list if x not in ignorePoint]

    print("原始范围", range_list)
    print("修正范围", new_range)
    # 随机打乱,取给定的个数
    np.random.shuffle(new_range)
    # 删除率
    # del_list = new_range[:int(num*ratio)]
    if num>=2048:
        del_list = new_range[:num-2048]
    else:
        del_list = []
    print("待删除索引", del_list)

    print("原始索引", branch)
    re_branch = []
    for i in branch:
        re_branch.append([x for x in i if x not in del_list])
    print("修正后索引", re_branch)

    # 修正连接关系
    for i in sorted(del_list):
        swc[i+1][6] = swc[i][6]

    # 保存文件

    with open(file_path, 'a+') as f:
        for i in re_branch:
            for j in i:
                if i.index(j) == 0 and re_branch.index(i) != 0:
                    continue
                np.savetxt(f, swc[j], fmt = '%.2f', newline=' ')
                f.write('\n')


def noiseSWC(swc, file, sigma = 1):
    '''神经元swc增加正态分布高斯噪声'''
    noise = np.random.normal(0, sigma, len(swc)-1) / 100
    # print(len(noise), noise)
    # xyz加相同噪声
    # print(swc[2], noise[2])
    swc[1:, 2] = swc[1:, 2] + noise
    swc[1:, 3] = swc[1:, 3] + noise
    swc[1:, 4] = swc[1:, 4] + noise
    # print(swc[2])
    file_path = "noise_" + file
    np.savetxt(file_path, swc, fmt='%.2f')


def pc_normalize(pc):
    """
    对点云数据进行归一化
    :param pc: 需要归一化的点云数据
    :return: 归一化后的点云数据
    """
    # 求质心，也就是一个平移量，实际上就是求均值
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    # 对点云进行缩放
    pc = pc / m
    return pc


def tifRescale(tif_path):
    '''
    各向同性缩放，目的是缩放到满足条件的最大形状，最后填充到目标形状
    input: new_1
    output:
    '''
    if os.path.exists(tif_path):
        images = tiff.imread(tif_path)
        # print('img_arr: ',images.shape)
        # 16bit 转 8bit
        images_uint8 = np.zeros(images.shape, dtype=np.uint8)
        # print(type(images_uint8), images_uint8.dtype)

        for i in range(len(images[:])):
            # print(images[i].max(),images[i].min())
            if (images[i].max() - images[i].min()) == 0:
                images_uint8[i, :, :] = np.uint8(images[i, :, :])
                # print('image: ',i,' ',tif_path)
            else:
                images_uint8[i, :, :] = np.uint8(
                    255 * (np.float32(images[i, :, :]) - images[i].min()) / (images[i].max() - images[i].min()))
        # print(images_uint8)
        # print(images_uint8.shape, images_uint8.dtype)

        # resize
        new_images_uint8 = transform.rescale(images_uint8)
        new_images_uint8 = img_as_ubyte(new_images_uint8)
        # print('img type dtype shape :',type(new_images_uint8),new_images_uint8.dtype, new_images_uint8.shape)

        return new_images_uint8



def tifResize(tif_path, size):
    if os.path.exists(tif_path):
        images = tiff.imread(tif_path)
        # print('img_arr: ',images.shape)
        # 16bit 转 8bit
        images_uint8 = np.zeros(images.shape, dtype=np.uint8)
        # print(type(images_uint8), images_uint8.dtype)

        images[images >= 255] = 255
        images_uint8 = np.uint8(255 * (np.float32(images) - images.min()) / (images.max() - images.min()))
        # print(images_uint8)
        # print(images_uint8.shape, images_uint8.dtype)



        # resize
        new_images_uint8 = transform.resize(images_uint8, size)
        new_images_uint8 = img_as_ubyte(new_images_uint8)
        # print('img type dtype shape :',type(new_images_uint8),new_images_uint8.dtype, new_images_uint8.shape)
        new_images_uint81 = np.uint8(255 * (np.float32(new_images_uint8) - new_images_uint8.min()) / (new_images_uint8.max() - new_images_uint8.min()))


        return new_images_uint81


    else:
        print('文件不存在')


print('#'*50,'Begin','#'*50)

def dataExtraction(root_path,root_id):
    '''
    初始label h5
    '''
    # swc
    swc_path = os.path.join(root_path,root_id,'swc')
    print(swc_path)
    files = os.listdir(swc_path)
    # print(files)
    files.sort(key=lambda x: int(x.split('.')[0]))
    # print(len(files))

    data = []

    threshold = 1024
    for file in files:
        # print(os.path.join(swc_path,name))
        swc_data = np.loadtxt(os.path.join(swc_path, file))
        if len(swc_data) > threshold:
            swc_data_list = uni_sampling(swc_data[:, 2:5], threshold)
            # print(type(swc_data_list))
        else:
            # swc_data_list = fill_zeros(swc_data[:,2:5],threshold)
            swc_data_list = uni_sampling_up(swc_data, threshold)
        swc_data_arr = np.array(swc_data_list)
        swc_data_arr_norm = pc_normalize(swc_data_arr)
        swc_data_list_norm = swc_data_arr_norm.tolist()
        data.append(swc_data_list_norm)
    data = np.array(data)
    #####################################################
    # image
    image_path = os.path.join(root_path,root_id,'image_crop')
    tifs = os.listdir(image_path)
    tifs.sort(key=lambda x: int(x.split('.')[0]))
    # print(len(tifs))

    img_list = []
    for tif in tifs:
        tif_path = os.path.join(image_path, tif)
        new_tif = tifResize(tif_path, (64, 64, 64))
        img_list.append(new_tif.tolist())
    img_arr = np.array(img_list, dtype=np.uint8)
    #####################################################
    # label
    dir_label_path = os.path.join(root_path,root_id)
    label_path = os.path.join(dir_label_path, 'label.txt')
    label = np.loadtxt(label_path)[:, 1]
    label = label.reshape(-1, 1)
    name = np.loadtxt(label_path)[:, 0]
    name = name.reshape(-1,1)


    print(img_arr.shape)
    print(data.shape)
    print(label.shape)
    print(name.shape)
    return data,label,img_arr,name


'''
def dataExtractionNew(root_path,root_id):
    #####################################################
    # label
    dir_label_path = os.path.join(root_path,root_id)
    label_path = os.path.join(dir_label_path, 'label_new_level1.txt')
    label = np.loadtxt(label_path)[:, 1]
    label = label.reshape(-1, 1)
    name = np.loadtxt(label_path)[:, 0]
    name = name.reshape(-1,1)
    # swc
    swc_path = os.path.join(root_path,root_id,'swc')
    # print(swc_path)
    files = name[:]
    # print(files)
    data = []
    threshold = 1024
    for file in files:
        # print(os.path.join(swc_path, str(int(file[0]))+'.swc'))
        swc_data = np.loadtxt(os.path.join(swc_path, str(int(file[0]))+'.swc'))
        if len(swc_data) > threshold:
            swc_data_list = uni_sampling(swc_data[:, 2:5], threshold)
            # print(type(swc_data_list))
        else:
            # swc_data_list = fill_zeros(swc_data[:,2:5],threshold)
            swc_data_list = uni_sampling_up(swc_data, threshold)
        swc_data_arr = np.array(swc_data_list)
        swc_data_arr_norm = pc_normalize(swc_data_arr)
        swc_data_list_norm = swc_data_arr_norm.tolist()
        data.append(swc_data_list_norm)
    data = np.array(data)
    print(data.shape)
    print(data[0, 0, :])
    print(type(data[0, 0, 0]))
    #####################################################
    # image
    # root_path = r'D:\yaogang\datasets\new_1\194511'
    image_path = os.path.join(root_path,root_id,'image')
    # tifs = os.listdir(image_path)
    # tifs.sort(key=lambda x: int(x.split('.')[0]))
    # # print(len(tifs))

    tifs = name[:]
    img_list = []
    for tif in tifs:
        tif_path = os.path.join(image_path, str(int(tif[0]))+'.tif')
        print(tif_path)
        new_tif = tifResize(tif_path, (64, 64, 64))
        img_list.append(new_tif.tolist())
    img_arr = np.array(img_list, dtype=np.uint8)

    # print(img_arr.shape)
    # print(img_arr[0].dtype)
    print(img_arr.shape)
    print(data.shape)
    print(label.shape)
    print(name.shape)
    return data,label,img_arr,name
'''
def h5Write(data,label,img_arr,name,root_path,root_id):
    x = data
    y = label
    z = img_arr
    n = name
    kf = KFold(n_splits=5,shuffle=True)
    d = kf.split(x)
    i = 1
    for train_idx, test_idx in d:
        train_data = x[train_idx]
        train_label = y[train_idx]
        train_img = z[train_idx]
        train_name = n[train_idx]

        test_data = x[test_idx]
        test_label = y[test_idx]
        test_img = z[test_idx]
        test_name = n[test_idx]
        # print('train_idx:{}, train_data:{}'.format(train_idx, train_data))
        # print('train_idx:{}, train_label:{}'.format(train_idx, train_label))

        # print('test_idx:{}, test_data:{}'.format(test_idx, test_data.shape))

        print(train_idx)
        print(test_idx)

        # print(test_idx, type(train_data[-1]), type(train_img[-1]))

        if not os.path.exists(os.path.join(root_path,root_id,'label_new_level1')):
            os.makedirs(os.path.join(root_path,root_id,'label_new_level1'))
        train_file_name = os.path.join(root_path,root_id,'label_new_level1','train_%s_%s_%s.h5'%(root_path.split('\\')[-1],root_id,str(i)))
        test_file_name = os.path.join(root_path, root_id,'label_new_level1', 'test_%s_%s_%s.h5' % (root_path.split('\\')[-1], root_id,str(i)))
        if not os.path.exists(train_file_name):
            with h5py.File(train_file_name, 'w') as f:
                f.create_dataset('data', data=train_data)
                f.create_dataset('label', data=train_label)
                f.create_dataset('image', data=train_img)
                f.create_dataset('name', data=train_name)
            with h5py.File(test_file_name, 'w') as f:
                f.create_dataset('data', data=test_data)
                f.create_dataset('label', data=test_label)
                f.create_dataset('image', data=test_img)
                f.create_dataset('name', data=test_name)
        else:
            print('已存在')
        i = i + 1


if __name__ == '__main__':
    root_path = r'D:\yaogang\datasets'
    save_path = r'D:\yaogang\datasets\image_crop'
    # datasets_list = ['192026','194511','201545','18704']
    datasets_list = ['192026']

    for dataset in datasets_list:
        base_path = os.path.join(root_path,dataset)
        # print(base_path)
        dir_list = os.listdir(base_path)
        dir_list.sort(key=lambda x:int(x))
        print(dir_list)
        for i in dir_list:
			## 调整图像数据,需要手动调整阈值
			imgResize(base_path, i, os.path.join(save_path, dataset),300)
			## 提取数据
            data, label, img_arr, name = dataExtraction(base_path, i)
			## 保存h5文件
            h5Write(data, label, img_arr, name, base_path, i)
                



    # h5dir = os.path.join(root_path,dir_list[3])
    # h5file = os.path.join(h5dir,'test_%s_%s.h5'%(h5dir.split('\\')[-2],h5dir.split('\\')[-1]))
    # print(h5file)
    # with h5py.File(h5file,'r') as f:
    #     for k in f.keys():
    #         print(k,f[k])
    #     data = f['data'][:]
    #     image = f['image'][:]
    #     label = f['label'][:]
    #     name = f['name'][:]
    # print(label[15])
    # print(name[15])
    #
    # tiff.imsave(r'D:\yaogang\datasets\18704\5\0.tif',image[15])
    # print(data[15].shape,type(data[15]))


    # x = data[15][:,0]
    # y = data[15][:,1]
    # z = data[15][:,2]
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # ax.scatter(x, y, z, 'blue')
    # plt.show()
    #
    # swc = np.loadtxt(r"D:\yaogang\datasets\18704\4\swc\15.swc")
    # xs = swc[:, 2]
    # ys = swc[:, 3]
    # zs = swc[:, 4]
    # fig1 = plt.figure(2)
    # ax1 = fig1.add_subplot(111, projection='3d')
    #
    # ax1.scatter(xs, ys, zs, 'blue')
    # plt.show()















