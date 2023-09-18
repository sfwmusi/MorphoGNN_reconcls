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
import time
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
from util import cal_loss, IOStream, TripletLoss
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from torch.utils.tensorboard import SummaryWriter
import tifffile as tiff
from tqdm import tqdm
from torch.autograd import Variable


def _init_():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/'+args.exp_name+'_epochs'+str(args.epochs)+'_batchsize'+str(args.batch_size)):
        os.makedirs('outputs/'+args.exp_name+'_epochs'+str(args.epochs)+'_batchsize'+str(args.batch_size))
    if not os.path.exists('outputs/'+args.exp_name+'_epochs'+str(args.epochs)+'_batchsize'+str(args.batch_size)+'/'+'models'):
        os.makedirs('outputs/'+args.exp_name+'_epochs'+str(args.epochs)+'_batchsize'+str(args.batch_size)+'/'+'models')
    os.system('cp main_cls.py outputs'+'/'+args.exp_name+'_epochs'+str(args.epochs)+'_batchsize'+str(args.batch_size)+'/'+'main_cls.py.backup')
    os.system('cp model.py outputs' + '/' + args.exp_name+'_epochs'+str(args.epochs)+'_batchsize'+str(args.batch_size) + '/' + 'model.py.backup')
    os.system('cp util.py outputs' + '/' + args.exp_name+'_epochs'+str(args.epochs)+'_batchsize'+str(args.batch_size) + '/' + 'util.py.backup')
    os.system('cp data.py outputs' + '/' + args.exp_name+'_epochs'+str(args.epochs)+'_batchsize'+str(args.batch_size) + '/' + 'data.py.backup')

def train(args, io):
    train_loss_sum = 0.0
    train_acc_sum = 0.0
    test_loss_sum = 0.0
    test_acc_sum = 0.0

    best_test_acc = 0
    for idname in range(1,6):
        print('#'*25,'第',idname,'折','#'*25)
		## 五折交叉

        log_writer = SummaryWriter()
        ### train 和 test 结合在一起
        train_loader = DataLoader(MultiNetData(partition='train', num_points=args.num_points,id_name=idname), num_workers=0,
                                  batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(MultiNetData(partition='test', num_points=args.num_points,id_name=idname), num_workers=0,
                                 batch_size=args.test_batch_size, shuffle=True, drop_last=True)
        device = torch.device("cuda" if args.cuda else "cpu")
        # device = torch.device("cpu")
        #Try to load models
        if args.model == 'pointnet':
            model = PointNet(args).to(device)
        elif args.model == 'dgcnn':
            model = DGCNN_cls(args).to(device)
        elif args.model == 'multinet':
            model = MultiNet(args).to(device)
        else:
            raise Exception("Not implemented")

        # print(str(model))

        model = nn.DataParallel(model)
        print("Let's use", torch.cuda.device_count(), "GPUs!")

        if args.use_sgd:
            print("Use SGD")
            opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=5e-4)
        else:
            print("Use Adam")
            opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

        if args.scheduler == 'cos':
            scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
        elif args.scheduler == 'step':
            scheduler = StepLR(opt, step_size=20, gamma=0.7)

        criterion = cal_loss
        criterion2 = TripletLoss()


        test_acc_epoch = []
        train_acc_epoch = []
        test_loss_epoch = []
        train_loss_epoch = []
        for epoch in range(args.epochs):
            ####################
            # Train
            ####################
            train_loss = 0.0
            count = 0.0
            model.train()
            # print(model)
            train_pred = []
            train_true = []
            for batch_id, (data, label, image) in tqdm(enumerate(train_loader, 0), total=len(train_loader) ,smoothing=0.9):
                data, label, image = data.to(device), label.to(device).squeeze(), image.to(device).unsqueeze(1)
                # data, label, image = data.to(device), label.to(device).squeeze(), image.to(device)
                data = data.permute(0, 2, 1)
                # print(data.size())
                # print(image.size())
                batch_size = data.size()[0]
                opt.zero_grad()
                logits, feature = model(data, image)

                ######loss的增加？##########
                loss1 = criterion(logits, label)
                loss2, _ = criterion2(logits, label)
                loss = loss1 + loss2
                ###########################
                # print('loss: ',loss)
                loss.backward()
                opt.step()
                preds = logits.max(dim=1)[1]
                count += batch_size
                train_loss = train_loss + loss.item() * batch_size
                # print('loss: ',loss.item())
                # print('train loss: ',train_loss)
                train_true.append(label.cpu().numpy())
                train_pred.append(preds.detach().cpu().numpy())
            if args.scheduler == 'cos':
                scheduler.step()
            elif args.scheduler == 'step':
                if opt.param_groups[0]['lr'] > 1e-5:
                    scheduler.step()
                if opt.param_groups[0]['lr'] < 1e-5:
                    for param_group in opt.param_groups:
                        param_group['lr'] = 1e-5

            # 可视化

            train_true = np.concatenate(train_true)
            train_pred = np.concatenate(train_pred)
            log_writer.add_scalar('Loss/train',train_loss*1.0/count,epoch)
            log_writer.add_scalar('Accuracy/train', metrics.accuracy_score(train_true, train_pred), epoch)

            outstr = 'K = %d, train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (idname,epoch,
                                                                                     train_loss*1.0/count,
                                                                                     metrics.accuracy_score(
                                                                                         train_true, train_pred),
                                                                                     metrics.balanced_accuracy_score(
                                                                                         train_true, train_pred))
            io.cprint(outstr)
            train_loss_epoch.append(train_loss*1.0/count)
            train_acc_epoch.append(metrics.accuracy_score(train_true, train_pred))

            ####################
            # Test
            ####################
            test_loss = 0.0
            count = 0.0
            model.eval()
            test_pred = []
            test_true = []
            for data, label, image in test_loader:
                data, label, image = data.to(device), label.to(device).squeeze(), image.to(device).unsqueeze(1)
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                logits, feature = model(data, image)

                loss1 = criterion(logits, label)
                loss2, _ = criterion2(logits, label)

                loss = loss1 + loss2
                preds = logits.max(dim=1)[1]
                count += batch_size
                # # print(data,'#############',loss,'################',loss.item())
                test_loss += loss.item() * batch_size
                # if test_true.size == 1:
                test_true.append(label.cpu().numpy())
                # print('test: ',test_true)
                test_pred.append(preds.detach().cpu().numpy())


            # print(test_true)
            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)
            test_acc = metrics.accuracy_score(test_true, test_pred)
            avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
            log_writer.add_scalar('Loss/test', test_loss*1.0/count, epoch)
            log_writer.add_scalar('Accuracy/test', test_acc, epoch)
            outstr = 'K = %d, test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (idname,epoch,
                                                                                  test_loss*1.0/count,
                                                                                  test_acc,
                                                                                  avg_per_class_acc)
            io.cprint(outstr)
            if test_acc >= best_test_acc:
                best_test_acc = test_acc
                state = {'model': model.state_dict(), 'optimizer': opt.state_dict(), 'epoch': epoch}
                torch.save(state, 'outputs/%s/models/model.t7' % (args.exp_name+'_epochs'+str(args.epochs)+'_batchsize'+str(args.batch_size)))

            test_acc_epoch.append(test_acc)
            test_loss_epoch.append(test_loss*1.0/count)
        train_acc_sum = train_acc_sum + np.sum(train_acc_epoch)
        train_loss_sum = train_loss_sum + np.sum(train_loss_epoch)
        test_acc_sum = test_acc_sum + np.sum(test_acc_epoch)
        test_loss_sum = test_loss_sum + np.sum(test_loss_epoch)

    print('train acc: ', train_acc_sum / args.epochs /5) ### 五折平均
    print('train loss: ', train_loss_sum / args.epochs /5)
    print('test acc: ', test_acc_sum / args.epochs /5)
    print('test loss: ', test_loss_sum / args.epochs /5)



def test(args, io):
    test_loader = DataLoader(MultiNetData(partition='test', num_points=args.num_points, id_name=5), num_workers=0,
               batch_size=args.test_batch_size, shuffle=True, drop_last=True)
    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN_cls(args).to(device)
    elif args.model == 'multinet':
        model = MultiNet(args).to(device)
    else:
        raise Exception("Not implemented")

    model = nn.DataParallel(model)
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
        logits = model(data, image)
        # 每一行的最大值的索引
        preds = logits.max(dim=1)[1]
        # print(logits)
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    print('测试预测结果:\n',test_pred)
    print('测试真实标签:\n',test_true)
    # 得到混淆矩阵(confusion matrix,简称cm)
    # confusion_matrix 需要的参数：y_true(真实标签),y_pred(预测标签)
    cm = confusion_matrix(y_true=test_true, y_pred=test_pred)
    cm_norm = confusion_matrix(y_true=test_true, y_pred=test_pred, normalize='true')

    # 打印混淆矩阵
    print("Confusion Matrix: ")
    print(cm)
    print(cm_norm)
    TN = cm[0,0]
    TP = cm[1,1]
    FP = cm[0,1]
    FN = cm[1,0]

    precision = TP*1.0 / (TP + FP)*1.0 # 预测为正样本中实际为正样本
    recall = TP*1.0 / (TP + FN)*1.0  # 实际为正样本中预测为正样本
    acc = (TP+TN)*1.0/(TP+TN+FP+FN)*1.0
    F1_score = 2*precision*recall*1.0/(precision*1.0+recall*1.0)


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
    print('acc: %.6f'%acc)
    print('precision: %.6f'%precision)
    print('recall: %.6f'%recall)
    print('F1 score: %.6f'%F1_score)
    io.cprint(outstr)

def retrain(args, io):
    print('#'*50,'retrain','#'*50)
    train_loss_sum = 0.0
    train_acc_sum = 0.0
    test_loss_sum = 0.0
    test_acc_sum = 0.0

    best_test_acc = 0.75
    for idname in range(1, 6):
        print('#' * 25, '第', idname, '折', '#' * 25)

        log_writer = SummaryWriter()
        ### train 和 test 结合在一起
        train_loader = DataLoader(MultiNetData(partition='train', num_points=args.num_points, id_name=idname),
                                  num_workers=0,
                                  batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(MultiNetData(partition='test', num_points=args.num_points, id_name=idname),
                                 num_workers=0,
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

        # print(str(model))

        model = nn.DataParallel(model)
        # 加载
        # model.load_state_dict(torch.load(args.model_path)['model'])

        print("Let's use", torch.cuda.device_count(), "GPUs!")

        if args.use_sgd:
            print("Use SGD")
            opt = optim.SGD(model.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=5e-4)
        else:
            print("Use Adam")
            opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
            # 加载
            # opt.load_state_dict(torch.load(args.model_path)['optimizer'])

            # print(opt.state_dict()['param_groups'][0]['lr'])
            # print('load state dict success!')
        if args.scheduler == 'cos':
            scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
        elif args.scheduler == 'step':
            scheduler = StepLR(opt, step_size=20, gamma=0.7)

        criterion = cal_loss
        criterion2 = TripletLoss()

        test_acc_epoch = []
        train_acc_epoch = []
        test_loss_epoch = []
        train_loss_epoch = []
        for epoch in range(args.epochs):
            ####################
            # Train
            ####################
            train_loss = 0.0
            count = 0.0
            model.train()
            # print(model)
            train_pred = []
            train_true = []
            for batch_id, (data, label, image) in tqdm(enumerate(train_loader, 0), total=len(train_loader),
                                                       smoothing=0.9):
                data, label, image = data.to(device), label.to(device).squeeze(), image.to(device).unsqueeze(1)
                # data, label, image = data.to(device), label.to(device).squeeze(), image.to(device)
                data = data.permute(0, 2, 1)
                # print(data.size())
                # print(image.size())
                batch_size = data.size()[0]
                opt.zero_grad()
                logits, feature = model(data, image)

                loss1 = criterion(logits, label)
                loss2, _ = criterion2(feature, label)

                loss = loss1 + loss2
                # print('loss: ',loss)

                loss.backward()
                opt.step()
                preds = logits.max(dim=1)[1]
                count += batch_size
                train_loss = train_loss + loss.item() * batch_size
                # print('loss: ',loss.item())
                # print('train loss: ',train_loss)
                train_true.append(label.cpu().numpy())
                train_pred.append(preds.detach().cpu().numpy())
            if args.scheduler == 'cos':
                scheduler.step()
            elif args.scheduler == 'step':
                if opt.param_groups[0]['lr'] > 1e-5:
                    scheduler.step()
                if opt.param_groups[0]['lr'] < 1e-5:
                    for param_group in opt.param_groups:
                        param_group['lr'] = 1e-5

            # 可视化

            train_true = np.concatenate(train_true)
            train_pred = np.concatenate(train_pred)
            log_writer.add_scalar('Loss/train', train_loss * 1.0 / count, epoch)
            log_writer.add_scalar('Accuracy/train', metrics.accuracy_score(train_true, train_pred), epoch)

            outstr = 'K = %d, train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (idname, epoch,
                                                                                             train_loss * 1.0 / count,
                                                                                             metrics.accuracy_score(
                                                                                                 train_true,
                                                                                                 train_pred),
                                                                                             metrics.balanced_accuracy_score(
                                                                                                 train_true,
                                                                                                 train_pred))
            io.cprint(outstr)
            train_loss_epoch.append(train_loss * 1.0 / count)
            train_acc_epoch.append(metrics.accuracy_score(train_true, train_pred))

            ####################
            # Test
            ####################
            test_loss = 0.0
            count = 0.0
            model.eval()
            test_pred = []
            test_true = []
            for data, label, image in test_loader:
                data, label, image = data.to(device), label.to(device).squeeze(), image.to(device).unsqueeze(1)
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                logits, feature = model(data, image)

                loss1 = criterion(logits, label)
                loss2, _ = criterion2(feature, label)

                loss = loss1 + loss2
                preds = logits.max(dim=1)[1]
                count += batch_size
                # # print(data,'#############',loss,'################',loss.item())
                test_loss += loss.item() * batch_size
                # if test_true.size == 1:
                test_true.append(label.cpu().numpy())
                # print('test: ',test_true)
                test_pred.append(preds.detach().cpu().numpy())

            # print(test_true)
            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)
            test_acc = metrics.accuracy_score(test_true, test_pred)
            avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
            log_writer.add_scalar('Loss/test', test_loss * 1.0 / count, epoch)
            log_writer.add_scalar('Accuracy/test', test_acc, epoch)
            outstr = 'K = %d, test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (idname, epoch,
                                                                                          test_loss * 1.0 / count,
                                                                                          test_acc,
                                                                                          avg_per_class_acc)
            io.cprint(outstr)
            if test_acc >= best_test_acc:
                best_test_acc = test_acc
                state = {'model': model.state_dict(), 'optimizer': opt.state_dict(), 'epoch': epoch}
                torch.save(state, 'outputs/%s/models/model.t7' % args.exp_name)

            test_acc_epoch.append(test_acc)
            test_loss_epoch.append(test_loss * 1.0 / count)
        train_acc_sum = train_acc_sum + np.sum(train_acc_epoch)
        train_loss_sum = train_loss_sum + np.sum(train_loss_epoch)
        test_acc_sum = test_acc_sum + np.sum(test_acc_epoch)
        test_loss_sum = test_loss_sum + np.sum(test_loss_epoch)
        break
    print('train acc: ', train_acc_sum / 5.0 / 200)
    print('train loss: ', train_loss_sum / 5.0 / 200)
    print('test acc: ', test_acc_sum / 5.0 / 200)
    print('test loss: ', test_loss_sum / 5.0 / 200)



if __name__ == "__main__":
    times = time.time()
    # Times
    time_tuple = time.localtime(time.time())
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp_%s%s%s%s%s'%(time_tuple[0],time_tuple[1],time_tuple[2],time_tuple[3],time_tuple[4]), metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='multinet', metavar='N',
                        choices=['pointnet', 'dgcnn', 'multinet'],
                        help='Model to use, [pointnet, dgcnn, multinet]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    # batch 32-->16 2022/3/4 20:43
    parser.add_argument('--batch_size', type=int, default=8, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=8, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=120, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    # 0.001->0.0005
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.0005, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='initial dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')

    parser.add_argument('--sample_size', default=112,
                        type=int, help='Height and width of inputs')
    parser.add_argument('--sample_duration', default=16,
                        type=int, help='Temporal duration of inputs')
    parser.add_argument('--n_classes', default=2, type=int,
                        help='Number of classes (activitynet: 200, kinetics: 400, ucf101: 101, hmdb51: 51)')

    args = parser.parse_args()

    _init_()

    io = IOStream('outputs/' + args.exp_name+'_epochs'+str(args.epochs)+'_batchsize'+str(args.batch_size) + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
        # retrain(args, io)
    else:
        test(args, io)

