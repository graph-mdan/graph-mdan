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


    def forward(self, sinputs, tinputs, slabels): # mdan entrance is forward
        """
        :param sinputs:     A list of inputs from k source domains.
        :param tinputs:     Input from the target domain.
        :param slabels:     A list of labels from k source domains.
        :return:
        """

        # s_feat = [torch.tensor(sinputs[i]) for i in self.tr_task_id]
        # s_label = [torch.tensor(slabels[i]) for i in self.tr_task_id]
        # st_feat, st_label = torch.tensor(sinputs[self.val_task_id]), torch.tensor(slabels[self.val_task_id])
        s_feat, s_label = [], []
        for i in range(len(self.data_name)):
            if i in self.tr_task_id:
                s_feat.append(sinputs[i])
                s_label.append(slabels[i])
            elif i == self.val_task_id:
                st_feat = sinputs[i]
                st_label = slabels[i]
            # else:
            #     print ("test domain is :", self.data_name[i])

        val_losses = []
        val_preds = []

        # compute all the loss
        tr_losses, _ = self.learner(s_feat, st_feat, s_label, is_train=True, weights=self.weights, counter=0)

        # compute gradients
        # for key in self.weights.keys():
        #     print (key)
        #     print (self.weights[key].size())
        grad = torch.autograd.grad(tr_losses, list(self.weights.values()))
        gvs = dict(zip(self.weights.keys(), grad))
        fast_weights = dict(zip(self.weights.keys(), [self.weights[key] - self.update_lr * gvs[key] for key in self.weights.keys()]))
        # fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.learner.parameters()))) # all the parameters

        with torch.no_grad():
            # fast adaption
            val_loss, val_pred = self.learner(s_feat, st_feat, s_label, tlabels=st_label, is_train=False, weights=fast_weights, counter=1)
            val_losses.append(val_loss)
            val_preds.append(val_pred)

        # Inner Loop: continue to build T1-TK steps graph
        for _ in range(1, self.update_step): # i.e., num_updates = 4, update 3 times
            # T_k loss on meta-train
            # we need meta-train loss to fine-tune the task and meta-test loss to update theta
            tr_loss = self.adplearner(s_feat, st_feat, s_label, weights=fast_weights)

            # compute gradients
            # grad = torch.autograd.grad(tr_loss, [self.weights[key] for key in self.weights.keys() if "gat" in key or "cls" in key])
            adp_weights = dict()
            for key in self.weights.keys() :
                if "gat" in key or "cls" in key:
                    adp_weights[key] = self.weights[key]
            grad = torch.autograd.grad(tr_loss, list(adp_weights.values()))

            # update theta_G and theta_y
            gvs = dict(zip(adp_weights.keys(), grad))
            fast_weights = dict(zip(self.weights.keys(), [self.weights[key] - self.update_lr * gvs[key] if "gat" in key or "cls" in key else self.weights[key] for key in self.weights.keys()]))
            # fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.adplearner.parameters()))) # only partial the parameters

            # forward on all the parameters Theta
            val_loss, val_pred = self.learner(s_feat, st_feat, s_label, tlabels=st_label, is_train=False, weights=fast_weights, counter=2)
            val_losses.append(val_loss)
            val_preds.append(val_pred)

        return val_losses[-1], val_preds[-1]

    # def clear(self, sh_relu, th_relu):
    #     sh_relu = []

    def learner(self, sinputs, tinputs, slabels, tlabels=None, is_train=False, weights=None, counter=0): # mdan entrance is forward
        """
        :param sinputs:     A list of inputs from k-1 source domains.
        :param tinputs:     Input from the simulated target domain.
        :param labels:      A list of labels from k-1 source domains or the simulated target domain.
        :param is_train:    used which labels for loss
        :param weights:     all parameters
        :return:
        """

        sh_relu, th_relu = copy.copy(sinputs), copy.copy(tinputs)

        # oringinal feature extractor
        for i in range(self.num_domains): # the number of domains is 3
            for hidden in self.hiddens:
                sh_relu[i] = F.relu(hidden(sh_relu[i]))

        for hidden in self.hiddens:
            th_relu = F.relu(hidden(th_relu))

        # feature extractor Stage-II: Graph Transferrable Learning
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

        for i in range(self.num_domains):
            all = (neg_dist[i] - pos_dist[i] < self.margin).cpu().numpy().flatten() # only consider the distance satisfy the inequality
            hard_triplets = np.where(all == 1) # a list of indices for embeddings

            a_embed = anc_embed[i][hard_triplets].to(self.device)
            p_embed = pos_embed[i][hard_triplets].to(self.device)
            n_embed = neg_embed[i][hard_triplets].to(self.device)
            tripletloss.append(TripletLoss(self.margin).forward(a_embed, p_embed, n_embed).to(self.device))
        # print(tripletloss)

        # compute losses
        if is_train: # use k-1 sources losses: train_loss
            # Classification probabilities on k-1 source domains.
            logprobs = []
            for i in range(self.num_domains):
                logprobs.append(F.log_softmax(self.softmax(F.relu(gsh_relu[i])), dim=1))
            losses = torch.stack([F.nll_loss(logprobs[j].to(self.device), slabels[j].to(self.device)) for j in range(self.num_domains)]).to(self.device) # classification loss

            # Domain classification accuracies.
            sdlabels = torch.ones(self.batch_size, requires_grad=False).type(torch.LongTensor).to(self.device) # print (slabels) # 1
            tdlabels = torch.zeros(self.batch_size, requires_grad=False).type(torch.LongTensor).to(self.device) # print (tlabels) # 0
            sdomains, tdomains = [], []
            for i in range(self.num_domains): # 3 domains
                sdomains.append(F.log_softmax(self.domains[i](self.grls[i](F.relu(gsh_relu[i]))), dim=1))
                tdomains.append(F.log_softmax(self.domains[i](self.grls[i](F.relu(gth_relu))), dim=1))

            domain_losses = torch.stack([F.nll_loss(sdomains[j], sdlabels) +
                                        F.nll_loss(tdomains[j], tdlabels) for j in range(self.num_domains)]).to(self.device) # here is losses of a pair <sdomains, tdomains>

            # Distance Learner
            tripletlosses = torch.stack(tripletloss)

            train_losses = torch.log(torch.sum(torch.exp(self.gamma * (losses + self.mu * domain_losses)))) / self.gamma + self.lamda * torch.sum(tripletlosses)
            train_pred = None

            return train_losses, train_pred

        else: # use the k-th source losses: val_loss
            # Classification probabilities
            logprobs = F.log_softmax(self.softmax(F.relu(gth_relu)), dim=1)
            losses = F.nll_loss(logprobs, tlabels)

            # Distance Learner
            # print (tripletloss)
            tripletlosses = tripletloss[-1]

            val_losses = torch.log(torch.sum(torch.exp(self.gamma * (losses)))) / self.gamma + self.lamda * torch.sum(tripletlosses)
            val_pred = torch.max(logprobs, 1)[1]
            return val_losses, val_pred


    def adplearner(self, sinputs, tinputs, slabels,  weights=None): # mdan entrance is forward
        """
        :param sinputs:     A list of inputs from k source domains.
        :param tinputs:     Input from the target domain.
        :param labels:     A list of labels from k source domains.
        :param is_train:    used which labels for loss
        :param weights:     all parameters
        :return:
        """

        sh_relu, th_relu = copy.copy(sinputs), copy.copy(tinputs)

        # feature extractor Stage-I
        for i in range(self.num_domains): # fixed weights
            for hidden in self.hiddens:
                sh_relu[i] = F.relu(hidden(sh_relu[i]))

        for hidden in self.hiddens:
            th_relu = F.relu(hidden(th_relu))

        # feature extractor Stage-II: Graph Transferrable Learning
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

            a_embed = anc_embed[i][hard_triplets].to(self.device)
            p_embed = pos_embed[i][hard_triplets].to(self.device)
            n_embed = neg_embed[i][hard_triplets].to(self.device)
            tripletloss.append(TripletLoss(self.margin).forward(a_embed, p_embed, n_embed).to(self.device))

        tripletlosses = torch.stack(tripletloss)

        # Classification probabilities on k source domains.
        logprobs = []
        for i in range(self.num_domains):
            logprobs.append(F.log_softmax(self.softmax(F.relu(gsh_relu[i])), dim=1))

        # source domain losses
        losses = torch.stack([F.nll_loss(logprobs[j], slabels[j]) for j in range(self.num_domains)]).to(self.device) # classification loss

        adp_loss = torch.log(torch.sum(torch.exp(self.gamma * (losses)))) / self.gamma + self.lamda * torch.sum(tripletlosses)
        return adp_loss


    def inference(self, sinputs, tinputs, slabels):
        pred_results = []
        size = tinputs.size()[0]
        input_sizes = [data.shape[0] for data in sinputs]
        for begin in range(0, tinputs.size()[0], self.batch_size):
            end = begin + self.batch_size
            end = min([end, size])
            if end-begin < self.batch_size: break

            # preparing batch
            s_feat, s_label = [], []
            for i in range(self.num_domains):
                s_batch_idx = np.random.choice(input_sizes[i], self.batch_size)
                s_feat.append(sinputs[self.tr_task_id[i]][s_batch_idx, :])
                s_label.append(slabels[self.tr_task_id[i]][s_batch_idx, :])
            t_feat = tinputs[range(begin, end), :]

            # parameter adaptation
            test_losses = [0 for _ in range(self.update_step + 1)]
            test_preds = [0 for _ in range(self.update_step + 1)]

            # compute all the loss
            tr_losses, _ = self.learner(s_feat, t_feat, s_label, is_train=True, weights=self.weights)

            # compute gradients
            grad = torch.autograd.grad(tr_losses, list(self.weights.values()))
            gvs = dict(zip(self.weights.keys(), grad))
            fast_weights = dict(zip(self.weights.keys(), [self.weights[key] - self.update_lr * gvs[key] for key in self.weights.keys()]))

            with torch.no_grad():
                # fast adaption
                test_loss, test_pred = self.learner(s_feat, t_feat, s_label, is_train=False, weights=fast_weights)
                test_losses.append(test_loss)
                test_preds.append(test_pred)

            for _ in range(1, self.update_step): # i.e., num_updates = 4, update 3 times
                # T_k loss on meta-train
                tr_loss = self.adplearner(s_feat, t_feat, s_label, weights=fast_weights)

                # compute gradients
                grad = torch.autograd.grad(tr_loss, [self.weights[key] for key in self.weights.keys() if "gat" in key or "cls" in key])

                # update theta_G and theta_y
                gvs = dict(zip(self.weights.keys(), grad))
                fast_weights = dict(zip(self.weights.keys(), [self.weights[key] - self.update_lr * gvs[key] for key in self.weights.keys() if "gat" in key or "cls" in key]))
                # fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.adplearner.parameters()))) # only partial the parameters

                # forward on all the parameters Theta
                test_loss, test_pred = self.learner(s_feat, t_feat, s_label, is_train=False, weights=fast_weights)
                test_losses.append(test_loss)
                test_preds.append(test_pred)

            pred_results += test_preds[-1]

        return pred_results


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
