import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.distance import PairwiseDistance
from torch.autograd import Variable





class AdjacencyLayer(nn.Module):
    def __init__(self, num_domains, batch_size, k, device):
        super(AdjacencyLayer, self).__init__()
        self.k = k
        self.num_domains = num_domains
        self.batch_size = batch_size
        self.pdist  = PairwiseDistance(2, keepdim=False)
        self.device = device


    def forward(self, input):
        """
        :param input: tensor with size [(num_domains+1) x batch_size, feature_dim].
        :return:
        """
        N = (self.num_domains+1)*self.batch_size
        adj = torch.zeros(N, N)
        # adjacent matrix for targets
        j = self.num_domains
        for i in range(self.num_domains): # compute the pairwise interaction of source x target
            # indices = torch.tensor([i])
            x_i = input[i*self.batch_size:(i+1)*self.batch_size, :] # [batch_size, feature_dim]
            x_j = input[j*self.batch_size:(j+1)*self.batch_size, :] # [batch_size, feature_dim]
            x_i_norm = F.normalize(x_i)
            x_j_norm = F.normalize(x_j)
            sim = torch.mm(x_i_norm, torch.transpose(x_j_norm, 0, 1))

            # find k-NN
            topk, indices = torch.topk(sim, self.k, largest=True) # largest k samples
            neighbor = Variable(torch.zeros(self.batch_size, self.batch_size)).to(self.device)
            neighbor = neighbor.scatter_(1, indices, topk)
            adj[i*self.batch_size:(i+1)*self.batch_size, j*self.batch_size:(j+1)*self.batch_size] = neighbor

        # for i in range(self.num_domains + 1): # compute the pairwise interaction of source x target
        #     for j in range(self.num_domains + 1):
        #         if i == j: continue
        #     # indices = torch.tensor([i])
        #     x_i = input[i*self.batch_size:(i+1)*self.batch_size, :] # [batch_size, feature_dim]
        #     x_j = input[j*self.batch_size:(j+1)*self.batch_size, :] # [batch_size, feature_dim]
        #     x_i_norm = F.normalize(x_i)
        #     x_j_norm = F.normalize(x_j)
        #     sim = torch.mm(x_i_norm, torch.transpose(x_j_norm, 0, 1))
        #
        #     # find k-NN
        #     topk, indices = torch.topk(sim, self.k, largest=True) # largest k samples
        #     neighbor = Variable(torch.zeros(self.batch_size, self.batch_size)).to(self.device)
        #     neighbor = neighbor.scatter_(1, indices, topk)
        #     adj[i*self.batch_size:(i+1)*self.batch_size, j*self.batch_size:(j+1)*self.batch_size] = neighbor

        # adjacent matrix for sources
        topk_ngh =[]
        for i in range(self.num_domains+1): # compute the pairwise interaction of source x source
            # indices = torch.tensor([i])
            x_i = input[i*self.batch_size:(i+1)*self.batch_size, :] # [batch_size, feature_dim]
            x_i_norm = F.normalize(x_i)
            sim = torch.mm(x_i_norm, torch.transpose(x_i_norm, 0, 1))

            # find k-NN
            topk, indices = torch.topk(sim, self.k, largest=True) # largest k samples for distance, contain itself
            neighbor = Variable(torch.zeros(self.batch_size, self.batch_size)).to(self.device)
            neighbor = neighbor.scatter_(1, indices, topk)
            adj[i*self.batch_size:(i+1)*self.batch_size, i*self.batch_size:(i+1)*self.batch_size] = neighbor
            topk_ngh.append(indices)
        # print (adj.size())
        return adj, topk_ngh # adj (N, N), topk_ngh (N, k)


class DistanceLayer(nn.Module):
    def __init__(self, num_domains, batch_size):
         super(DistanceLayer, self).__init__()

         self.num_domains = num_domains
         self.batch_size = batch_size
         self.pdist  = PairwiseDistance(2)

    def forward(self, semb, slabels):
         """
         :param input: tensor with size [(num_domains) x batch_size, feature_dim].
         :return:
             pos_dist: store the largest neg similarity
             neg_dist: store the smallest pos similarity
             pos_embed: store the pos embedding with smallest similarity
             neg_embed: store the neg embedding with largest similarity
         """

         pos_dist, neg_dist, pos_embed, neg_embed = [], [], [], []
         for k in range(self.num_domains):
            emb, label = semb[k], slabels[k]
             # dist = self.pdist.forward(emb, emb)
            sim = torch.mm(emb, torch.transpose(emb, 0, 1))
             # print (sim.size()) # 20 x 20
             # print (label)
            
            if len(pos_indices.size()) == 0 or len(neg_indices.size()) == 0: continue

            pdist, ndist, pembed, nembed = [], [], [], []
            pembed = torch.zeros(self.batch_size, emb.size()[1])
            nembed = torch.zeros(self.batch_size, emb.size()[1])
            for i in range(self.batch_size): # for each sample, find hard pos and neg
                same_indices, diff_indices = [],[]
                for j in range(self.batch_size):
                    if label[i] == label[j] and i!=j:
                        same_indices.append(j)
                    else:
                        adj_container = adj[i].tolist()
                        

                 
                #  same_indices = torch.tensor(same_indices)
                #  diff_indices = torch.tensor(diff_indices)
                 
                 # print (diff_indices)

                if len(same_indices)!=0:
                    min_sim, min_index = torch.min(sim[i, same_indices], 0) # min similarity in positives, hard postive
                    max_sim, max_index = torch.max(sim[i, diff_indices], 0) # max similarity in negatives, hard negative
                    pdist.append(min_sim)
                    ndist.append(max_sim)
                    pembed[i, :] = emb[same_indices[min_index], :]
                    nembed[i, :] = emb[diff_indices[max_index], :]

                else:
                    # zero is placeholder
                    pdist.append(0)
                    ndist.append(0)
                    
               
                    
            pos_dist.append(torch.tensor(pdist))
            neg_dist.append(torch.tensor(ndist))
            pos_embed.append(pembed)
            neg_embed.append(nembed)
         # print (pos_dist)
         # print (neg_dist)

         return pos_dist, neg_dist, pos_embed, neg_embed


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, layer, device, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.device = device
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        # print (self.W.size())
        # print (input.size())
        h = torch.mm(input, self.W)
        N = h.size()[0]
        # print (h.size()) # [80, 100]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # print (self.a.size()) # [200, 1]
        # print (a_input.size()) # [80, 80, 200]
        # print (e.size()) # [80, 80]

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj.to(self.device) > 0, e, zero_vec) # adj < 0, position e == -9e15
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        # print ("--")
        # print (attention.size())
        return h_prime
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'