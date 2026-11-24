import argparse
import os
import pickle
import random
import sys
import tempfile
import time

import gc
import matplotlib.cm
import networkx as nx
import numpy as np
import scipy.sparse as spsprs
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from layers import *
from utils import *
class GraphLearn(nn.Module):
    def __init__(self, input_dim, th, mode = 'Sigmoid-like'):
        super(GraphLearn, self).__init__()
        self.mode = mode
        self.w = nn.Linear(input_dim, 1)
        self.t = nn.Parameter(torch.ones(1))
        self.p = nn.Linear(input_dim, input_dim)
        self.threshold = nn.Parameter(torch.zeros(1))
        self.th = th    
    def forward(self, x):
        initial_x = x.clone()
        num, feat_dim = x.size(0), x.size(1)
        th = self.th
        x = self.p(x)
        x_norm = F.normalize(x,dim=-1)
        score = torch.matmul(x_norm, x_norm.T)
        self.score431 = score
        mask = (score > th).detach().float()
        markoff_value = 0
        output = score * mask + markoff_value * (1 - mask)
        return output  
    
    
class GCN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCN, self).__init__()
        self.gcn1 = GraphConv(input_dim, output_dim)
        self.gcn2 = GraphConv(input_dim, output_dim)
        
    def forward(self, x, adj):
        x = F.relu(self.gcn1(x, adj))
        x = self.gcn2(x, adj)
        return x
    


# 获取权重矩阵

def adj_normalization(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj

    
class GFR(nn.Module):
    def __init__(self, input_dim , output_dim ,th , mode='Sigmoid-like', label_drop = 0.,proj_drop =0.):
        super(GFR, self).__init__()
        self.th = th
        self.mode = mode
        # breakpoint()
        
        input_sum = sum(input_dim) 
    
        self.fusion_weight = nn.Parameter(torch.ones([len(input_dim),1]))
        self.combined_weight = nn.Parameter(torch.ones([len(input_dim),1]))
        
        self.GraphConstruct0 = GraphLearn(input_dim = input_dim[0],th = self.th, mode = self.mode)
        self.MessagePassing0 = GCN (input_dim = input_sum, output_dim = input_sum)

        self.GraphConstruct1 = GraphLearn(input_dim = input_dim[1], th = self.th, mode = self.mode)
        self.MessagePassing1 = GCN(input_dim = input_sum, output_dim = input_sum)

        self.GraphConstruct2 = GraphLearn(input_dim = input_dim[2], th = self.th, mode = self.mode)
        self.MessagePassing2 = GCN(input_dim = input_sum, output_dim = input_sum)

        self.GraphConstruct3 = GraphLearn(input_dim = input_dim[3], th = self.th, mode = self.mode)
        self.MessagePassing3 = GCN(input_dim= input_sum, output_dim= input_sum)

        self.proj_drop = nn.Dropout(proj_drop)
        self.label_drop = nn.Dropout(label_drop)


    def forward(self, x, model_num, label, index):
        combined_feat = []
        all_adj = []
        N = len(x[0])
        new_label = label.clone()
        new_label[new_label == 0] = -1
        new_label = [x if i in index else 0 for i, x in enumerate(new_label)]
        new_label = torch.tensor(new_label)
        l1 = new_label.unsqueeze(1).repeat(1, N)  
        l2 = new_label.t().repeat(N, 1)  
        adj1 = l1.eq_(l2)
        A = normalize_adj(adj1.cuda() + torch.eye(adj1.size(0)).cuda())
 
        for i in range(model_num):  
            f_w = self.fusion_weight[i]
            f_w = torch.sigmoid(f_w)
            c_w = self.combined_weight[i]
            c_w = torch.sigmoid(c_w)
            
            if i==0:
                adj = self.GraphConstruct0(x[i])
                normalized_adj0 = normalize_adj(adj + torch.eye(adj.size(0)).to(f_w.device))
                ALoss = torch.linalg.norm(normalized_adj0 - A).to(f_w.device)
                feat0 = self.MessagePassing0(torch.cat(x,dim=1), normalized_adj0)
                combined_feat = f_w*feat0+torch.cat(x,dim=1)

            elif i==1:
                adj = self.GraphConstruct1(x[i])
                normalized_adj1 = normalize_adj(adj + torch.eye(adj.size(0)).to(f_w.device))
                ALoss = torch.linalg.norm(normalized_adj1 - A).to(f_w.device) + ALoss
                feat1 = self.MessagePassing1(torch.cat(x,dim=1), normalized_adj1)
                combined_feat = f_w*feat1+torch.cat(x,dim=1)
                
            elif i==2:
                adj = self.GraphConstruct2(x[i])
                normalized_adj2 = normalize_adj(adj + torch.eye(adj.size(0)).to(f_w.device))
                ALoss = torch.linalg.norm(normalized_adj2 - A).to(f_w.device) + ALoss
                feat3 = self.MessagePassing2(torch.cat(x,dim=1), normalized_adj2)
                combined_feat = f_w*feat3+combined_feat

                        
            elif i==3:
                adj = self.GraphConstruct3(x[i])
                normalized_adj3 = normalize_adj(adj + torch.eye(adj.size(0)).to(f_w.device))
                ALoss = torch.linalg.norm(normalized_adj3 - A).to(f_w.device) + ALoss
                feat3 = self.MessagePassing3(torch.cat(x,dim=1), normalized_adj3)
                combined_feat = f_w*feat3+combined_feat
            all_adj.append(adj)
                   
        return combined_feat , all_adj, ALoss/model_num
    
    
        
