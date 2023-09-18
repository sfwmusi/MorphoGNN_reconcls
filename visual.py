#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: main_cls.py
@Time: 2018/10/13 10:39 PM

Modified by
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@Time: 2019/12/30 9:32 PM
"""

from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data import ModelNet40, MultiNetData
from model import PointNet, DGCNN_cls, MultiNet
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import load_iris


def _init_():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/' + args.exp_name):
        os.makedirs('outputs/' + args.exp_name)
    if not os.path.exists('outputs/' + args.exp_name + '/' + 'models'):
        os.makedirs('outputs/' + args.exp_name + '/' + 'models')
    os.system('cp main_cls.py outputs' + '/' + args.exp_name + '/' + 'main_cls.py.backup')
    os.system('cp model.py outputs' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py outputs' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py outputs' + '/' + args.exp_name + '/' + 'data.py.backup')


def test(args, io):
    test_loader = DataLoader(MultiNetData(partition='test', num_points=args.num_points, id_name=1), num_workers=0,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=True)
    device = torch.device("cuda" if args.cuda else "cpu")

    # Try to load models
    if args.model == 'pointnet':
        model = PointNet(args).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN_cls(args).to(device)
    elif args.model == 'multinet':
        model = MultiNet(args).to(device)
    else:
        raise Exception("Not implemented")

    model = nn.DataParallel(model)
    # ['model']
    model.load_state_dict(torch.load(args.model_path)['model'])
    model = model.eval()
    # print(model)
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    for data, label, image in test_loader:
        data, label, image = data.to(device), label.to(device).squeeze(), image.to(device).unsqueeze(1)
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits, feature = model(data, image)
        # 每一行的最大值的索引
        preds = logits.max(dim=1)[1]
        # print(logits)
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    print('测试预测结果:\n', test_pred)
    print('测试真实标签:\n', test_true)
    # 得到混淆矩阵(confusion matrix,简称cm)
    # confusion_matrix 需要的参数：y_true(真实标签),y_pred(预测标签)
    cm = confusion_matrix(y_true=test_true, y_pred=test_pred)
    cm_norm = confusion_matrix(y_true=test_true, y_pred=test_pred, normalize='true')

    # 打印混淆矩阵
    print("Confusion Matrix: ")
    print(cm)
    print(cm_norm)
    TN = cm[0, 0]
    TP = cm[1, 1]
    FP = cm[0, 1]
    FN = cm[1, 0]

    precision = TP * 1.0 / (TP + FP) * 1.0  # 预测为正样本中实际为正样本
    precision_N = TN * 1.0 / (TN + FN) * 1.0
    recall = TP * 1.0 / (TP + FN) * 1.0  # 实际为正样本中预测为正样本
    recall_N = TN * 1.0 / (TN + FP) * 1.0
    acc = (TP + TN) * 1.0 / (TP + TN + FP + FN) * 1.0
    F1_score = 2 * precision * recall * 1.0 / (precision * 1.0 + recall * 1.0)

    # 画出混淆矩阵
    # ConfusionMatrixDisplay 需要的参数: confusion_matrix(混淆矩阵), display_labels(标签名称列表)
    ll = [0, 1]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ll)
    disp.plot(cmap='Blues')
    plt.show()

    ll = [0, 1]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=ll)
    disp.plot(cmap='Blues')
    plt.show()

    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f' % (test_acc, avg_per_class_acc)
    print('acc: %.6f' % acc)
    print('precision: %.6f' % precision)
    print('recall: %.6f' % recall)
    print('F1 score: %.6f' % F1_score)
    io.cprint(outstr)


def swc_test(args, io):
    test_loader = DataLoader(MultiNetData(partition='test', num_points=args.num_points, id_name=1), num_workers=0,
                             batch_size=8, shuffle=True, drop_last=True)
    device = torch.device("cuda" if args.cuda else "cpu")

    # Try to load models
    if args.model == 'pointnet':
        model = PointNet(args).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN_cls(args).to(device)
    elif args.model == 'multinet':
        model = MultiNet(args).to(device)
    else:
        raise Exception("Not implemented")

    model = nn.DataParallel(model)
    # ['model']
    model.load_state_dict(torch.load(args.model_path)['model'])
    model = model.eval()
    # print(model)
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    test_p = []
    test_name = []
    for data, label, image in test_loader:
        data, label, image = data.to(device), label.to(device).squeeze(), image.to(device).unsqueeze(1)
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits,_ = model(data, image)
        # print('logits: ',logits)
        # print('label: ', label)
        # 每一行的最大值的索引
        preds = logits.max(dim=1)[1]
        # print('pred: ',F.softmax(logits,dim=1))
        p = F.softmax(logits, dim=1)
        test_p.append(p.detach().cpu().numpy())
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_p = np.concatenate(test_p)
    # test_name = np.concatenate(test_name)
    print('测试预测结果:\n', test_pred,'\n')
    print('测试真实标签:\n', test_true,'\n')
    print('预测概率:\n', test_p,'\n')
    print('01比率:\n', sum(test_true)/len(test_true), '\n')
    p_last = test_p.max(axis=1)
    np.savetxt('test_p.txt', test_p)
    # np.savetxt('test_name.txt', test_name)
    rate_range = []
    acc_range = []
    for p_l in range(50,95):
        # print('p:\n', p_last[p_last >= p_l*1.0/100], test_pred[p_last >= p_l*1.0/100], test_true[p_last >= p_l*1.0/100], '\n')
        all_true = test_pred[p_last >= p_l*1.0/100] == test_true[p_last >= p_l*1.0/100]
        if len(all_true) != 0:
            print(('类间距离区间:%.2f, 准确率:%.6f, 数目比例:%.6f') %(2*p_l*1.0/100-1.0, np.sum(all_true == 1) * 1.0 / len(all_true) * 1.0, len(all_true) * 1.0 / len(test_pred)*1.0))
            rate_range.append(len(all_true) * 1.0 / len(test_pred)*1.0)
            acc_range.append(np.sum(all_true == 1) * 1.0 / len(all_true) * 1.0)
        # print(np.sum(all_true == 1), '\n', np.sum(all_true == 0), '\n', len(all_true))
    all_true = test_pred[p_last >= 70 * 1.0 / 100] == test_true[p_last >= 70 * 1.0 / 100]
    rate_range = np.array(rate_range)
    acc_range = np.array(acc_range)
    np.savetxt('rate_range.txt', rate_range)
    np.savetxt('acc_range.txt', acc_range)
    print(len(rate_range))
    x_range = np.arange(0,len(rate_range)*2,2)/100.0
    plt.plot(x_range,rate_range,'^-')
    plt.plot(x_range,acc_range, 'v-')
    plt.legend(['proportion','acc'])
    plt.xlabel('inter-class distance')
    plt.ylabel('proportion|acc')
    # plt.show()
    plt.savefig("save/image.png")
    plt.close()
    print(len(all_true) * 1.0 / len(test_pred)*1.0,len(all_true))
    # 得到混淆矩阵(confusion matrix,简称cm)
    # confusion_matrix 需要的参数：y_true(真实标签),y_pred(预测标签)
    cm = confusion_matrix(y_true=test_true, y_pred=test_pred)
    cm_norm = confusion_matrix(y_true=test_true, y_pred=test_pred, normalize='true')

    # 打印混淆矩阵
    print("Confusion Matrix: ")
    print(cm)
    print(cm_norm)
    TN = cm[0, 0]
    TP = cm[1, 1]
    FP = cm[0, 1]
    FN = cm[1, 0]

    precision = TP * 1.0 / (TP + FP) * 1.0  # 预测为正样本中实际为正样本
    precision_N = TN * 1.0 / (TN + FN) * 1.0
    recall = TP * 1.0 / (TP + FN) * 1.0  # 实际为正样本中预测为正样本
    recall_N = TN * 1.0 / (TN + FP) * 1.0
    acc = (TP + TN) * 1.0 / (TP + TN + FP + FN) * 1.0
    F1_score = 2 * precision * recall * 1.0 / (precision * 1.0 + recall * 1.0)
    F1_score_N = 2 * precision_N * recall_N * 1.0 / (precision_N * 1.0 + recall_N * 1.0)

    # 画出混淆矩阵
    # ConfusionMatrixDisplay 需要的参数: confusion_matrix(混淆矩阵), display_labels(标签名称列表)
    ll = [0, 1]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ll)
    disp.plot(cmap='Blues')
    # plt.show()
    plt.savefig("save/image1.png")
    plt.close()
    ll = [0, 1]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=ll)
    disp.plot(cmap='Blues')
    # plt.show()
    plt.savefig("save/image2.png")
    plt.close()
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f' % (test_acc, avg_per_class_acc)
    print('acc: %.6f' % acc)
    print('precision: %.6f' % precision,'precision_N: %.6f' % precision_N)
    print('recall: %.6f' % recall,'recall_N: %.6f' % recall_N)
    print('F1 score: %.6f' % F1_score,'F1 score_N: %.6f' % F1_score_N)
    io.cprint(outstr)




if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='multinet', metavar='N',
                        choices=['pointnet', 'dgcnn','multinet'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    # batch 32-->16 2022/3/4 20:43
    parser.add_argument('--batch_size', type=int, default=8, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=8, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    # 0.001->0.0005
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='initial dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    #outputs/cls_1024/models/model.t7
    parser.add_argument('--model_path', type=str, default=r'outputs/exp_202212121658_epochs500_batchsize32/models/model.t7', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--sample_size', default=112,
                        type=int, help='Height and width of inputs')
    parser.add_argument('--sample_duration', default=16,
                        type=int, help='Temporal duration of inputs')
    parser.add_argument('--n_classes', default=2, type=int,
                        help='Number of classes (activitynet: 200, kinetics: 400, ucf101: 101, hmdb51: 51)')
    args = parser.parse_args()

    # _init_()

    io = IOStream('outputs/' +'exp_202212111634_epochs500_batchsize32' + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    swc_test(args, io)
