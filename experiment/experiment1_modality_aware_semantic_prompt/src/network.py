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
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.t1 = nn.Linear(input_dim, hidden_dim)
        self.t2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.t1(x))
        x = F.relu(self.t2(x))
        return x

class PLN(nn.Module):
    def __init__(self, input_data_dims, hyperpm, prompt_text_features, prompt_config='P0'):
        super(PLN, self).__init__()
        self.hyperpm = hyperpm
        self.input_data_dims = input_data_dims
        # self.input_data_dims = [x * 2 for x in self.input_data_dims]
        self.d_q = hyperpm.n_hidden
        self.d_k = hyperpm.n_hidden
        self.d_v = hyperpm.n_hidden
        self.n_head = hyperpm.n_head
        self.dropout = hyperpm.dropout
        self.n_layer = hyperpm.nlayer
        self.modal_num = hyperpm.nmodal
        self.n_class = hyperpm.nclass
        self.prompt_config = prompt_config
        self.prompt_output_dim = 9
        self.d_out = self.d_v * self.n_head * self.modal_num
        self.fusion_weight = nn.Parameter(torch.ones([6,1]))
        self.InputLayer = VariLengthInputLayer(self.input_data_dims, self.d_k, self.d_v, self.n_head, self.dropout)
        self.Encoder = []
        self.FeedForward = []
        
        for i in range(self.n_layer):
            encoder = EncodeLayer(self.d_k * self.n_head, self.d_k, self.d_v, self.n_head, self.dropout)
            #encoder = nn.MultiheadAttention(self.d_k * self.n_head, self.n_head, dropout = self.dropout) #nn.multi_head_attn
            self.add_module('encode_%d' % i, encoder)
            self.Encoder.append(encoder)
            
            feedforward = FeedForwardLayer(self.d_v * self.n_head, self.d_v * self.n_head, dropout = self.dropout)
            self.add_module('feed_%d' % i, feedforward)
            self.FeedForward.append(feedforward)
        d_in = self.d_v * self.n_head * self.modal_num
        self.Outputlayer = OutputLayer(d_in, self.d_v * self.n_head, self.n_class, self.modal_num, self.dropout)
        self.avg_pool = nn.AvgPool1d(kernel_size=4, stride=4)
        self.MLP = MLP(input_dim = 4096,hidden_dim = 128, output_dim = 9)
        if tuple(prompt_text_features.shape) != (self.modal_num, 2, 4, 1024):
            raise ValueError(f'expected frozen prompt features {(self.modal_num, 2, 4, 1024)}, got {tuple(prompt_text_features.shape)}')
        self.register_buffer('prompt_text_features', prompt_text_features.detach().float().clone())
        self.capture_prompt_diagnostics = False
        self.last_prompt_diagnostics = None

    def semantic_prototypes(self):
        if self.prompt_config == 'P0':
            raw = self.prompt_text_features[0].reshape(2, -1)
            prototypes = self.MLP(raw)
            return prototypes.unsqueeze(0).expand(self.modal_num, -1, -1)
        raw = self.prompt_text_features.reshape(self.modal_num * 2, -1)
        return self.MLP(raw).reshape(self.modal_num, 2, self.prompt_output_dim)
        
    def forward(self, x):
        bs = x.size(0)
        attn_map = []
        combine_part = []
        x, _attn = self.InputLayer(x)
        attn = _attn.mean(dim=1)
        attn_map.append(attn.detach().cpu().numpy())
        for i in range(self.n_layer):
            x, _attn = self.Encoder[i](q=x, k=x, v=x, modal_num=self.modal_num)
            attn = _attn.mean(dim=1)
            x = self.FeedForward[i](x)
            attn_map.append(attn.detach().cpu().numpy())
        part_count = x.size(-1) // self.prompt_output_dim
        if x.size(-1) % self.prompt_output_dim:
            raise ValueError('PLN feature width must be divisible by the prompt output width')
        part_x = x.reshape(bs, self.modal_num, part_count, self.prompt_output_dim)
        part_x_norm = F.normalize(part_x, p=2, dim=-1)
        part_x_norm_expanded = part_x_norm.unsqueeze(-2)  # 变为 [871, 4, 4, 1, 9]
        prototypes = self.semantic_prototypes()
        if self.prompt_config == 'P0':
            # Keep the original P0 operation order for direct forward parity.
            text_features_norm = F.normalize(prototypes[0], p=2, dim=-1)
            text_features_norm = text_features_norm.unsqueeze(0).unsqueeze(0).unsqueeze(2)
            similarity_matrix = torch.matmul(part_x_norm_expanded, text_features_norm.transpose(-1, -2))
            similarity_matrix = similarity_matrix[:, :, :, -1, :]
        else:
            text_features_norm = F.normalize(prototypes, p=2, dim=-1)
            text_features_norm = text_features_norm.unsqueeze(0).unsqueeze(2).transpose(-1, -2)
            similarity_matrix = torch.matmul(part_x_norm_expanded, text_features_norm).squeeze(-2)
        similarity_matrix = similarity_matrix.clamp(min=-5, max=5)
        similarity_weights = torch.softmax(similarity_matrix, dim=-1)
        similarity_max, _ = torch.max(similarity_weights, dim=-1)
        local_before = part_x.detach().clone() if self.capture_prompt_diagnostics else None
        part_x = similarity_max.unsqueeze(-1) * part_x
        if self.capture_prompt_diagnostics:
            relative_change = (part_x - x.reshape_as(part_x)).norm(dim=(-2, -1)) / x.reshape_as(part_x).norm(dim=(-2, -1)).clamp_min(1e-12)
            if self.prompt_config == 'P0':
                proto_norm = F.normalize(prototypes, p=2, dim=-1)
                prototype_cosine = (proto_norm[:, 0] * proto_norm[:, 1]).sum(-1)
            else:
                proto_norm = F.normalize(prototypes, p=2, dim=-1)
                prototype_cosine = (proto_norm[:, 0] * proto_norm[:, 1]).sum(-1)
            self.last_prompt_diagnostics = {
                'raw_similarity': similarity_matrix.detach(),
                'similarity_weights': similarity_weights.detach(),
                'prototype_cosine': prototype_cosine.detach(),
                'relative_token_change': relative_change.detach(),
                'local_features_before': local_before,
                'local_features_after': part_x.detach(),
            }
        else:
            self.last_prompt_diagnostics = None
        reshaped_x = x.reshape(-1, 1, x.size(-1))
        global_x = self.avg_pool(reshaped_x)
        m = part_x.shape[2]
        global_x = global_x.reshape(bs, self.modal_num, self.prompt_output_dim)
        for i in range(m):
            part = part_x[:,:,i,:]
            Vsim = F.softmax(part*global_x, dim=-1) 
            part_embedding = Vsim*part
            combine_part.append(part_embedding)
        x = x + torch.cat(combine_part,dim=2)
        x = x.view(bs, -1)
        attn_embedding = attn.view(bs, -1) 
        output, hidden = self.Outputlayer(x, attn_embedding)
        return output, hidden, attn_map
  
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


def build_label_target(label, train_index, device):
    n = int(label.numel())
    train_mask = torch.zeros(n, dtype=torch.bool, device=label.device)
    train_mask[torch.as_tensor(train_index, dtype=torch.long, device=label.device)] = True
    visible_label = torch.where(label == 0, torch.full_like(label, -1), label)
    visible_label = torch.where(train_mask, visible_label, torch.zeros_like(visible_label))
    relation = visible_label[:, None].eq(visible_label[None, :])
    return normalize_adj(relation.to(device) + torch.eye(n, device=device))

    
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
        A = build_label_target(label, index, x[0].device)
 
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
    
    
        
