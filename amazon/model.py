#!/usr/bin/env python
# -*- coding: utf-8 -*-

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


class GAT(nn.Module): # reshape
    def __init__(self, nfeat, nhid, adj, dropout, alpha, nheads, num_domains, batch_size, k, l, device):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, device=device, layer=l, concat=True) for _ in range(nheads)]
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

        self.num_domains = configs["num_domains"]
        self.batch_size = configs["batch_size"]

        self.dropout = configs["dropout"]
        self.alpha = configs["alpha"]
        self.nheads = configs["nheads"]

        self.k = configs["k"]
        self.margin = configs["margin"]

        # self.adjs = [AdjacencyLayer(self.num_domains, self.batch_size, self.k, target=False, i_domain=i) for i in range(self.num_domains)]
        self.adj = AdjacencyLayer(self.num_domains, self.batch_size, self.k, self.device)
        self.dist_f = DistanceLayer(self.num_domains, self.batch_size)

        # Parameters of hidden, fully-connected layers, feature learning component.
        self.hiddens = nn.ModuleList([nn.Linear(self.num_neurons[i], self.num_neurons[i+1])
                                      for i in range(self.num_hidden_layers)])
        # self.gat_hiddens = nn.ModuleList([GAT(self.num_gat_neurons[0], self.num_gat_neurons[1], self.adj, self.dropout, self.alpha,
        #                                                            self.nheads, self.num_domains, self.batch_size, self.k, 0, self.device)])

        self.gat_hiddens = nn.ModuleList([GAT(self.num_gat_neurons[0], self.num_gat_neurons[1], self.adj, self.dropout, self.alpha,
                                                                   self.nheads, self.num_domains, self.batch_size, self.k, 0, self.device),
                                          GAT(self.nheads * self.num_gat_neurons[1], self.num_gat_neurons[2], self.adj, self.dropout, self.alpha,
                                                                                                     self.nheads, self.num_domains, self.batch_size, self.k, 1, self.device)])
        # self.gat_hiddens = nn.ModuleList([GAT(self.num_gat_neurons[i], self.num_gat_neurons[i+1], self.adj, self.dropout, self.alpha,
        #                                                            self.nheads, self.num_domains, self.batch_size, self.k, i, self.device)
        #                                   for i in range(self.num_gat_hidden_layers)])

        # Parameter of the final softmax classification layer.
        # self.softmax = nn.Linear(self.num_gat_neurons[-1], configs["num_classes"])
        self.softmax = nn.Linear(self.nheads*self.num_gat_neurons[-1], configs["num_classes"])
        # Parameter of the domain classification layer, multiple sources single target domain adaptation.
        # self.domains = nn.ModuleList([nn.Linear(self.num_gat_neurons[-1], 2) for _ in range(self.num_domains)])
        self.domains = nn.ModuleList([nn.Linear(self.nheads*self.num_gat_neurons[-1], 2) for _ in range(self.num_domains)])
        # Gradient reversal layer.
        self.grls = [GradientReversalLayer() for _ in range(self.num_domains)]


    def forward(self, sinputs, tinputs, slabels): # mdan entrance is forward
        """
        :param sinputs:     A list of inputs from k source domains.
        :param tinputs:     Input from the target domain.
        :param slabels:     A list of labels from k source domains.
        :return:
        """
        sh_relu, th_relu = sinputs, tinputs

        # oringinal feature extractor
        for i in range(self.num_domains): # the number of domains is 3
            for hidden in self.hiddens:
                sh_relu[i] = F.relu(hidden(sh_relu[i]))

        for hidden in self.hiddens:
            th_relu = F.relu(hidden(th_relu))

        # Graph Attentive Transferrable Learning
        x = [sh_relu[i] for i in range(self.num_domains)]# [num_domains, batch_size, num_features]
        x.append(th_relu) # [num_domains+1, batch_size, num_features]
        x = torch.stack(x)

        x_ = x[0]
        for i in range(1, self.num_domains+1):
            x_ = torch.cat((x_, x[i]), 0)

        # compute adjacency matrix, GAT
        adj_mtr, topk_ngh = self.adj(x_) # N x N , N = (num_domains+1)*batch_size
        for gat_hidden in self.gat_hiddens:
            x_ = gat_hidden(x_, adj_mtr) # graph_emb should be [(num_domains+1) x batch_size, num_features]
        graph_emb = x_
        gsh_relu = []
        for i in range(self.num_domains):
            gsh_relu.append(graph_emb[i*self.batch_size:(i+1)*self.batch_size, :])
        gth_relu = graph_emb[self.num_domains*self.batch_size:(self.num_domains+1)*self.batch_size, :]

        # Triplet loss on source domain: choose hard examples
        gsh_relu_norm = [F.normalize(gsh_relu[i]) for i in range(self.num_domains)] # normalize
        pos_dist, neg_dist, anc_embed, pos_embed, neg_embed = self.dist_f(gsh_relu_norm, slabels, topk_ngh) # pos_dist or neg_dist should be [num_domains x batch_size, 1]
        tripletloss = []
        num_domains = len(pos_dist)
        for i in range(num_domains):
            all = (neg_dist[i] - pos_dist[i] < self.margin).cpu().numpy().flatten() # only consider the distance satisfy the inequality
            hard_triplets = np.where(all == 1) # a list of indices for embeddings
            # if hard_triplets[0].size == 0:
            #     continue
            a_embed = anc_embed[i][hard_triplets].to(self.device)
            p_embed = pos_embed[i][hard_triplets].to(self.device)
            n_embed = neg_embed[i][hard_triplets].to(self.device)
            tripletloss.append(TripletLoss(self.margin).forward(a_embed, p_embed, n_embed).to(self.device))

        # Classification probabilities on k source domains.
        logprobs = []
        for i in range(self.num_domains):
            logprobs.append(F.log_softmax(self.softmax(F.relu(gsh_relu[i])), dim=1))

        # Domain classification accuracies.
        sdomains, tdomains = [], []
        for i in range(self.num_domains): # 3 domains
            sdomains.append(F.log_softmax(self.domains[i](self.grls[i](F.relu(gsh_relu[i]))), dim=1))
            tdomains.append(F.log_softmax(self.domains[i](self.grls[i](F.relu(gth_relu))), dim=1))

        return logprobs, sdomains, tdomains, tripletloss


    def inference(self, sinputs, tinputs):
        # logprobs = []
        gth_relu_ = []
        size = tinputs.size()[0]
        input_sizes = [data.shape[0] for data in sinputs]
        for begin in range(0, tinputs.size()[0], self.batch_size):
            end = begin + self.batch_size
            end = min([end, size])
            if end-begin < self.batch_size: break

            # preparing batch
            sh_relu = []
            for i in range(self.num_domains):
                s_batch_idx = np.random.choice(input_sizes[i], self.batch_size)
                sh_relu.append(sinputs[i][s_batch_idx, :])
            t_batch_idx = range(begin, end)
            th_relu = tinputs[t_batch_idx, :]

            # feature extractor Stage-I
            for i in range(self.num_domains):
                for hidden in self.hiddens:
                    sh_relu[i] = F.relu(hidden(sh_relu[i]))

            for hidden in self.hiddens:
                th_relu = F.relu(hidden(th_relu))

            # feature extractor Stage-II
            # propagation for tranduction
            x = [sh_relu[i] for i in range(self.num_domains)]# [num_domains, batch_size, num_features]
            x.append(th_relu) # [num_domains+1, batch_size, num_features]
            x = torch.stack(x)

            ## construct graph
            x_ = x[0]
            for i in range(1, self.num_domains+1):
                x_ = torch.cat((x_, x[i]), 0)

            adj_mtr, _ = self.adj(x_)
            for gat_hidden in self.gat_hiddens:
                x_ = gat_hidden(x_, adj_mtr)
            graph_emb = x_
            gth_relu = graph_emb[self.num_domains*self.batch_size:(self.num_domains+1)*self.batch_size, :]
            gth_relu_.append(gth_relu)

            # Classification probabilities on target domains.
        gth_relu_ = torch.stack(gth_relu_).view(len(gth_relu_) * gth_relu_[0].size()[0], -1)
        logprobs = F.log_softmax(self.softmax(F.relu(gth_relu_)), dim=1)
        # th_relu_ = torch.stack(th_relu).view(len(th_relu) * th_relu[0].size()[0], -1)
        # logprobs = F.log_softmax(self.softmax(F.relu(th_relu_)), dim=1)
        # logprobs = torch.stack(logprobs)
        return logprobs


class TripletLoss(torch.autograd.Function):

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.pdist  = PairwiseDistance(2)

    def forward(self, anchor, positive, negative):
        """
        :param anchor:     samples embedding from source domain. n x d: n is sample number, d is dimension
        :param positive:   hard positives
        :param negative:   hard negatives
        :return:
        """
        pos_dist   = self.pdist.forward(anchor, positive)
        neg_dist   = self.pdist.forward(anchor, negative)

        hinge_dist = torch.clamp(self.margin + pos_dist - neg_dist, min = 0.0) # Clamp all elements in input into the range [ 0.0, self.margin + pos_dist - neg_dist ] and return a resulting tensor
        loss       = torch.mean(hinge_dist)
        return loss

class MDANet(nn.Module):
    """
    Multi-layer perceptron with adversarial regularizer by domain classification.
    """
    def __init__(self, configs):
        super(MDANet, self).__init__()
        self.input_dim = configs["input_dim"]
        self.num_hidden_layers = len(configs["hidden_layers"])
        self.num_neurons = [self.input_dim] + configs["hidden_layers"]
        self.num_domains = configs["num_domains"]

        # Parameters of hidden, fully-connected layers, feature learning component.
        self.hiddens = nn.ModuleList([nn.Linear(self.num_neurons[i], self.num_neurons[i+1])
                                      for i in range(self.num_hidden_layers)])
        # Parameter of the final softmax classification layer.
        self.softmax = nn.Linear(self.num_neurons[-1], configs["num_classes"])
        # Parameter of the domain classification layer, multiple sources single target domain adaptation.
        self.domains = nn.ModuleList([nn.Linear(self.num_neurons[-1], 2) for _ in range(self.num_domains)])
        # Gradient reversal layer.
        self.grls = [GradientReversalLayer() for _ in range(self.num_domains)]


    def forward(self, sinputs, tinputs): # mdan entrance is forward
        """
        :param sinputs:     A list of k inputs from k source domains.
        :param tinputs:     Input from the target domain.
        :return:
        """
        sh_relu, th_relu = sinputs, tinputs
        for i in range(self.num_domains): # the number of domains is 3
            for hidden in self.hiddens:
                sh_relu[i] = F.relu(hidden(sh_relu[i]))
        for hidden in self.hiddens:
            th_relu = F.relu(hidden(th_relu))

        # Classification probabilities on k source domains.
        logprobs = []
        for i in range(self.num_domains):
            logprobs.append(F.log_softmax(self.softmax(sh_relu[i]), dim=1))

        # Domain classification accuracies.
        sdomains, tdomains = [], []
        for i in range(self.num_domains): # 3 domains
            sdomains.append(F.log_softmax(self.domains[i](self.grls[i](sh_relu[i])), dim=1))
            tdomains.append(F.log_softmax(self.domains[i](self.grls[i](th_relu)), dim=1))
        return logprobs, sdomains, tdomains

    def inference(self, inputs):
        h_relu = inputs
        for hidden in self.hiddens:
            h_relu = F.relu(hidden(h_relu))
        print (h_relu.size()) # [4465, 100]  [3586, 100]  [5681, 100] [5945, 100]
        # Classification probability.
        logprobs = F.log_softmax(self.softmax(h_relu), dim=1)
        print (logprobs.size())
        print (logprobs)
        return logprobs
