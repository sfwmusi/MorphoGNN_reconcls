#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: util
@Time: 4/5/19 3:47 PM
"""


import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn


class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=None, class_num=2 , reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            self.alpha = alpha

    def forward(self, predict, target):
        target = target.contiguous().view(-1)
        eps = 0.2
        n_class = predict.size(1)
        one_hot = torch.zeros_like(predict).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)

        # loss = F.binary_cross_entropy_with_logits(pred, one_hot, reduction='mean')
        print(predict)
        pt = F.softmax(predict, dim=1)
        probs = -(one_hot * pt).sum(dim=1)+1e-10
        print('pt: ', pt)
        log_prb = probs.log()
        ids = target.view(-1, 1)
        alpha = self.alpha[ids.data.view(-1)].view(-1, 1)
        alpha = alpha.cuda()
        probs = probs.cuda()
        log_prb = log_prb.cuda()
        #在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
        loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_prb

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

def cal_loss(pred, gold, smoothing=False,weight=None):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)



    if smoothing:
        eps = 0.2
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)

        # loss = F.binary_cross_entropy_with_logits(pred, one_hot, reduction='mean')
        # print(one_hot)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1).mean()
        # print('pred: ', pred)
        # print('loss: ', loss)
    else:
        ###类别不均衡，增加权重###
        if weight is not None:
            crit = nn.CrossEntropyLoss(weight=weight)
        else:
            crit = nn.CrossEntropyLoss()
        loss = crit(pred, gold)
        # tripletloss:

        # print(pred.size())
        # print(gold.size())
        # print(pred.shape)
        # print(gold.shape)
        # print(pred.dtype)
        # print(gold.dtype)


    return loss


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()




def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    # dist.addmm_(1, -2, x, y.t())
    dist.addmm_(x,y.t(),beta=1,alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()
    return dist


def cosine_dist(x, y):
    bs1, bs2 = x.size(0), y.size(0)
    frac_up = torch.matmul(x, y.transpose(0, 1))
    frac_down = (torch.sqrt(torch.pow(x, 2).sum(dim=1)).view(bs1, 1).repeat(1, bs2)) * \
                (torch.sqrt(torch.pow(y, 2).sum(dim=1).view(1, bs2).repeat(bs1, 1)))
    cosine = frac_up / frac_down
    cos_d = 1 - cosine
    return cos_d


def _batch_hard(mat_distance, mat_similarity, indice=False):
    sorted_mat_distance, positive_indices = torch.sort(mat_distance + (-100000.0) * (1 - mat_similarity), dim=1,
                                                       descending=True)
    hard_p = sorted_mat_distance[:, 0]
    hard_p_indice = positive_indices[:, 0]
    sorted_mat_distance, negative_indices = torch.sort(mat_distance + 100000.0 * mat_similarity, dim=1,
                                                       descending=False)
    hard_n = sorted_mat_distance[:, 0]
    hard_n_indice = negative_indices[:, 0]
    if (indice):
        return hard_p, hard_n, hard_p_indice, hard_n_indice
    return hard_p, hard_n


class TripletLoss(nn.Module):
    def __init__(self, margin=0.5, normalize_feature=True):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.normalize_feature = normalize_feature
        self.margin_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, emb, label):
        if self.normalize_feature:
            emb = F.normalize(emb)
        # print('emb')
        # print(emb)
        mat_dist = euclidean_dist(emb, emb)

        assert mat_dist.size(0) == mat_dist.size(1)
        N = mat_dist.size(0)
        mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()
        # print(mat_dist)
        # print(mat_sim)
        dist_ap, dist_an = _batch_hard(mat_dist, mat_sim)
        assert dist_an.size(0) == dist_ap.size(0)
        y = torch.ones_like(dist_ap)
        loss = self.margin_loss(dist_an, dist_ap, y)

        prec = (dist_an.data > dist_ap.data).sum() * 1.0 / y.size(0)
        return loss, prec

if __name__ == '__main__':
    loss1 = TripletLoss()

    an = torch.randn(8,4086)
    y = torch.ones(8).long()
    yy = torch.randint(2,(8,)).long()
    xx = torch.randn(8,2)
    print(an)
    print(yy)
    print(xx)
    loss2 = cal_loss(xx,yy)
    print('loss2: ',loss2)
    l, _ = loss1(an,yy)
    print('l: ',l)
    loss = l + loss2
    print('loss: ',loss)
    loss.requires_grad_(True)
    loss.backward()

