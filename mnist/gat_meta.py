#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, AdjacencyLayer, DistanceLayer
from torch.nn.modules.distance import PairwiseDistance

# l2_dist = PairwiseDistance(2) # 2 norm distance

logger = logging.getLogger(__name__)

class GradientReversalLayer(torch.autograd.Function):
    """
    Implement the gradient reversal layer for the convenience of domain adaptation neural network.
    The forward part is the identity function while the backward part is the negative function.
    """
    def forward(self, inputs):
        return inputs

    def backward(self, grad_output):
        grad_input = grad_output.clone()
        grad_input = -grad_input
        return grad_input


class Linear(nn.Module):
    """Same function as nn.Linear"""
    def __init__(self, weights, name, device, index=None):
        super(Linear, self).__init__()
        self.device = device
        if name == "fc":
            self.W = weights["fc_W"+str(index)]
            nn.init.xavier_uniform_(self.W.data, gain=1.414)
            self.b = weights["fc_b"+str(index)]

        elif name == "cls":
            self.W = weights["cls_W"]
            nn.init.xavier_uniform_(self.W.data, gain=1.414)
            self.b = weights["cls_b"]

        elif name == "dsc":
            self.W = weights["dsc_W"+str(index)]
            nn.init.xavier_uniform_(self.W.data, gain=1.414)
            self.b = weights["dsc_b"+str(index)]

        else:
            print ("error!")


    def forward(self, input):
        output = torch.mm(input, self.W)
        if self.b is not None:
            output += self.b
        ret = output
        return ret



class GAT(nn.Module): # reshape
    def __init__(self, weights, nfeat, nhid, adj, dropout, alpha, nheads, num_domains, batch_size, k, l, device):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(weights, nfeat, nhid, dropout=dropout, alpha=alpha, layer=l, device=device, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions): # multi-head attention， 8
            self.add_module('layer_{}'.format(l) + '_attention_{}'.format(i), attention)

    def forward(self, x, adj):
        """
        :param x:    tensor with size [(num_domains+1) x batch_size, num_features].
        :return:
        """

        x = F.dropout(x, self.dropout, training=self.training)
        # print (len(self.attentions)) # 8 heads
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.elu(self.out_att(x, adj))
        # x = torch.mean(torch.stack([att(x, adj) for att in self.attentions], dim=1), dim=1).squeeze(1)
        # print (x.size()) # [80x800] concat 8 heads
        #                  # [80x100] average 8 heads
        return x


class GraphMDANet(nn.Module):
    """
    Multi-layer perceptron with graph attentive adversarial regularizer by domain classification.
    """
    def __init__(self, configs, device):
        super(GraphMDANet, self).__init__()
        self.device = device

        self.input_dim = configs["input_dim"]
        self.num_hidden_layers = len(configs["hidden_layers"])
        self.num_gat_hidden_layers = len(configs["gat_hidden_layers"])

        self.num_neurons = [self.input_dim] + configs["hidden_layers"]
        self.num_gat_neurons = [self.num_neurons[-1]] + configs["gat_hidden_layers"]
        self.num_classes = configs["num_classes"]

        self.num_domains = configs["num_domains"] - 1
        self.batch_size = configs["batch_size"]

        self.dropout = configs["dropout"]
        self.alpha = configs["alpha"]
        self.nheads = configs["nheads"]

        self.k = configs["k"]
        self.margin = configs["margin"]

        # meta-learning
        self.update_lr = configs["update_lr"]
        self.update_step = configs["update_step"]
        self.val_task = configs["val_task"]
        self.test_task = configs["test_task"]
        self.data_name = configs["data_name"]
        self.gamma = configs["gamma"]
        self.lamda = configs["lambda"]
        self.mu = configs["mu"]

        self.val_task_id = self.data_name.index(self.val_task)
        self.test_task_id = self.data_name.index(self.test_task)
        self.tr_task_id = list(range(len(self.data_name)))
        self.tr_task_id.remove(self.val_task_id)
        self.tr_task_id.remove(self.test_task_id)


        self.weights = self.build_weights()

        # self.adjs = [AdjacencyLayer(self.num_domains, self.batch_size, self.k, target=False, i_domain=i) for i in range(self.num_domains)]
        self.adj = AdjacencyLayer(self.num_domains, self.batch_size, self.k, self.device)
        self.dist_f = DistanceLayer(self.num_domains, self.batch_size)

        # Parameters of hidden, fully-connected layers, feature learning component.
        self.hiddens = nn.ModuleList([Linear(self.weights, 'fc', self.device, i) for i in range(self.num_hidden_layers)])

        self.gat_hiddens = nn.ModuleList([GAT(self.weights, self.num_gat_neurons[0], self.num_gat_neurons[1], self.adj, self.dropout, self.alpha,
                                                                   self.nheads, self.num_domains, self.batch_size, self.k, 0, self.device)])

        # self.gat_hiddens = nn.ModuleList([GAT(self.num_gat_neurons[0], self.num_gat_neurons[1], self.adj, self.dropout, self.alpha,
        #                                                            self.nheads, self.num_domains, self.batch_size, self.k, 0, self.device),
        #                                   GAT(self.nheads * self.num_gat_neurons[1], self.num_gat_neurons[2], self.adj, self.dropout, self.alpha,
        #                                                                                              self.nheads, self.num_domains, self.batch_size, self.k, 1, self.device)])
        # self.gat_hiddens = nn.ModuleList([GAT(self.num_gat_neurons[i], self.num_gat_neurons[i+1], self.adj, self.dropout, self.alpha,
        #                                                            self.nheads, self.num_domains, self.batch_size, self.k, i, self.device)
        #                                   for i in range(self.num_gat_hidden_layers)])

        # Parameter of the final softmax classification layer.
        # self.softmax = nn.Linear(self.num_gat_neurons[-1], configs["num_classes"])
        self.softmax = Linear(self.weights, 'cls', self.device)
        # Parameter of the domain classification layer, multiple sources single target domain adaptation.
        # self.domains = nn.ModuleList([nn.Linear(self.num_gat_neurons[-1], 2) for _ in range(self.num_domains)])
        self.domains = nn.ModuleList([Linear(self.weights, 'dsc', self.device, i) for i in range(self.num_domains)])
        # Gradient reversal layer.
        self.grls = [GradientReversalLayer() for _ in range(self.num_domains)]

    # construct weights
    def build_weights(self):
        weights = {}
        weights = self.build_fc_weights(weights)
        weights = self.build_gat_weights(weights)
        weights = self.build_cls_weights(weights)
        weights = self.build_dsc_weights(weights)
        return weights

    def build_fc_weights(self, weights):
        for i, dim_in in enumerate(self.num_neurons):
           if i == len(self.num_neurons)-1: break
           dim_out = self.num_neurons[i+1]
           weights["fc_W"+str(i)] = nn.Parameter(torch.randn(size=(dim_in, dim_out)), requires_grad=True)
           weights["fc_b"+str(i)] = nn.Parameter(torch.randn(size=(dim_out,)), requires_grad=True)
        return weights

    def build_gat_weights(self, weights):
        for i, dim_in in enumerate(self.num_gat_neurons):
            if i == len(self.num_gat_neurons)-1: break
            dim_out = self.num_gat_neurons[i+1]
            weights["gat_W"+str(i)] = nn.Parameter(torch.randn(size=(dim_in, dim_out)), requires_grad=True)
            weights["gat_a"+str(i)] = nn.Parameter(torch.randn(size=(2*dim_out, 1)), requires_grad=True)
        return weights

    def build_cls_weights(self, weights):
        dim_in = self.nheads*self.num_gat_neurons[-1]
        dim_out = self.num_classes
        weights["cls_W"] = nn.Parameter(torch.randn(size=(dim_in, dim_out)), requires_grad=True)
        weights["cls_b"] = nn.Parameter(torch.randn(size=(dim_out,)), requires_grad=True)
        return weights


    def build_dsc_weights(self, weights):
        dim_in = self.nheads*self.num_gat_neurons[-1]
        dim_out = 2
        for i in range(self.num_domains):
           weights["dsc_W"+str(i)] = nn.Parameter(torch.rand(size=(dim_in, dim_out)), requires_grad=True)
           weights["dsc_b"+str(i)] = nn.Parameter(torch.rand(size=(dim_out,)), requires_grad=True)
        return weights
