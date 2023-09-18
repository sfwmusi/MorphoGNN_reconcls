# MorphoGNN_NeuronCls
用于MorphoGNN的神经元重建块的质量分类实验https://github.com/fun0515/MorphoGNN

## 运行需求：
- Python 3.8
- PyTorch 1.10.1
- CUDA 11.2
- Python包：glob, h5py, sklearn, plyfile，tifffile

&nbsp;

## 数据准备：

``` 
python data_preparation.py
```

注：

1.图像数据 Resize  

通过阈值，crop到最大有数据的形状（手动调整）

2.提取数据：图像，标签，swc，名称id

3.保存h5文件



## 运行训练脚本：

- 1024点

``` 
python main_cls_K.py --model=multinet --num_points=1024 --k=20 
```



## 训练结束后运行评估脚本：

- 1024点

``` 
python visual.py --exp_name=cls_1024_eval --num_points=1024 --k=20 --eval=True --model_path=outputs/exp_202212121658_epochs500_batchsize32/models/model.t7
```

注：评价指标主要是acc、precision、recall、F1





## 致谢

代码参考源于https://github.com/WangYueFt/dgcnn/tree/master/pytorch和https://github.com/xmuyzz/3D-CNN-PyTorch/blob/master/models/C3DNet.py
