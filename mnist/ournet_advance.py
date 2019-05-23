import torch.nn as nn
import torch
import torch.nn.functional as F
from gatlayer import GraphAttentionLayer,AdjacencyLayer,DistanceLayer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.nn.modules.distance import PairwiseDistance
import numpy as np

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
    def __init__(self, nfeat, nhid, adj, dropout, alpha, nheads, num_domains, batch_size,k, device):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, device=device, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions): # multi-head attention， 8
            self.add_module('layer_{}' + '_attention_{}'.format(i), attention)

    def forward(self, x, adj):
        """
        :param x:    tensor with size [(num_domains+1) x batch_size, num_features].
        :return:
        """

        x = F.dropout(x, self.dropout, training=self.training)
        # print (len(self.attentions)) # 8 heads
        x = torch.mean(torch.stack([att(x, adj) for att in self.attentions], dim=1), dim=1).squeeze(1)
        # print (x.size()) # [80x800] concat 8 heads
        #                  # [80x100] average 8 heads
        return x



class OurNet(nn.Module):

    def __init__(self,configs):
        super(OurNet, self).__init__()
        self.device = configs['device']
        self.num_domains = configs['num_domains']
        self.num_classes = configs['num_classes']
        self.domains = nn.ModuleList([nn.Linear(2048, 2) for _ in range(self.num_domains)])
        self.grls = [GradientReversalLayer() for _ in range(self.num_domains)]
        self.dropout = configs['dropout']
        self.num_gat_hidden_layers = 1
        self.num_gat_neurons_in = 2048
        self.num_gat_neurons_out = 2048
        self.alpha = configs['alpha']
        self.nheads = configs["nheads"]
        self.k = configs["k"]
        self.batch_size = configs["batch_size"]
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.margin = configs['margin']
        self.fc = nn.Linear(2048, self.num_classes)
        
        self.adj = AdjacencyLayer(self.num_domains, self.batch_size, self.k, self.device)
        self.dist_f = DistanceLayer(self.num_domains, self.batch_size)
        self.gat_hiddens = nn.ModuleList([GAT(self.num_gat_neurons_in, self.num_gat_neurons_out, self.adj, self.dropout, self.alpha,
                                                                   self.nheads, self.num_domains, self.batch_size, self.k, self.device)
                                          for i in range(self.num_gat_hidden_layers)])

    # def build_adj(self, sinputs, tinputs):
    #     """ shape like this e.g.  sinputs = 3, tinputs = 2
    #     top2 rows are tinputs
    #         1 0 1 1 1 
    #         0 1 1 1 1
    #         1 1 0 0 0
    #         1 1 0 0 0
    #         1 1 0 0 0 
    #     """
        
    #     shape = tinputs.shape[0]*(len(sinputs)+1)
    #     adj = torch.zeros((shape,shape))
    #     for i in range(tinputs.shape[0]):
    #         adj[i][i] =1
    #         adj[i+1:][i] =1
    #     for j in range(tinputs.shape[0]):
    #         adj[j][j] = 1
    #         adj[j][j+1:]=1
    #     return adj



    def forward(self,sinputs, tinputs,slabels):

        # feature extractor
        sh_relu, th_relu = sinputs, tinputs
        for i in range(self.num_domains):
            sh_relu[i] = self.layer1(sh_relu[i])
            sh_relu[i] = self.layer2(sh_relu[i])
            sh_relu[i] = sh_relu[i].view(sh_relu[i].size(0), -1)
        
        th_relu = self.layer1(th_relu)
        th_relu = self.layer2(th_relu)
        th_relu = th_relu.view(th_relu.size(0), -1)

        # Graph Attentive Transferrable Learning
        x = [sh_relu[i] for i in range(self.num_domains)]# [num_domains, batch_size, num_features]
        x.append(th_relu) # [num_domains+1, batch_size, num_features]
        x = torch.stack(x) #??
        
        x_ = x[0]
        for i in range(1, self.num_domains+1):
            x_ = torch.cat((x_, x[i]), 0) 

        # compute adjacency matrix, GAT
        adj = self.adj(x_) # N x N , N = (num_domains+1)*batch_size
        for gat_hidden in self.gat_hiddens:
            x_ = gat_hidden(x_, adj) # graph_emb should be [(num_domains+1) x batch_size, num_features]
        graph_emb = x_
        gsh_relu = []
        
        for i in range(self.num_domains):
            gsh_relu.append(graph_emb[i*self.batch_size:(i+1)*self.batch_size, :])

        
        gth_relu= graph_emb[self.num_domains*self.batch_size:(self.num_domains+1)*self.batch_size, :]

        # Triplet loss on source domain: choose hard examples
        gsh_relu_norm = [F.normalize(gsh_relu[i]) for i in range(self.num_domains)] # normalize
        
        
        pos_dist, neg_dist, pos_embed, neg_embed = self.dist_f(gsh_relu_norm, slabels) # pos_dist or neg_dist should be [num_domains x batch_size, 1]
        

        num_domains = len(pos_dist)
        
        tripletloss = []
        
        
        for i in range(num_domains):
            if len(pos_embed)==0:
                tripletloss.append(torch.tensor(0).to(self.device))
            else:
                all = (neg_dist[i] - pos_dist[i] < self.margin).cpu().numpy().flatten() # only consider the distance satisfy the inequality
                hard_triplets = np.where(all == 1) # a list of indices for embeddings
                
                if hard_triplets[0].size == 0:
                    continue
                a_embed = gsh_relu_norm[i][hard_triplets].to(self.device)
               
                p_embed = pos_embed[i][hard_triplets].to(self.device)
                n_embed = neg_embed[i][hard_triplets].to(self.device)
             
                tripletloss.append(TripletLoss(self.margin).forward(a_embed, p_embed, n_embed).to(self.device))
        
        
        
        # Classification probabilities on k source domains.
        
        logprobs = []
        for i in range(self.num_domains):
            logprobs.append(F.log_softmax(self.fc(sh_relu[i]), dim=1))

        # Domain classification accuracies.
        sdomains, tdomains = [], []
        for i in range(self.num_domains): # 3 domains
            
            sdomains.append(F.log_softmax(self.domains[i](self.grls[i](gsh_relu[i])), dim=1))
            tdomains.append(F.log_softmax(self.domains[i](self.grls[i](gth_relu)), dim=1))

        return logprobs, sdomains, tdomains, tripletloss




    def forward_naive(self, sinputs, tinputs):
        sh_relu, th_relu = sinputs, tinputs
        for i in range(self.num_domains):
            sh_relu[i] = self.layer1(sh_relu[i])
            sh_relu[i] = self.layer2(sh_relu[i])
            sh_relu[i] = sh_relu[i].reshape(sh_relu[i].size(0), -1)
        
        th_relu = self.layer1(th_relu)
        th_relu = self.layer2(th_relu)
        th_relu = th_relu.reshape(th_relu.size(0), -1)

        
        # Classification probabilities on k source domains.
        logprobs = []
        for i in range(self.num_domains):
            logprobs.append(F.log_softmax(self.fc(sh_relu[i]), dim=1))
        # Domain classification accuracies.
        sdomains, tdomains = [], []
        for i in range(self.num_domains):
            sdomains.append(F.log_softmax(self.domains[i](self.grls[i](sh_relu[i])), dim=1))
            tdomains.append(F.log_softmax(self.domains[i](self.grls[i](th_relu)), dim=1))
        return logprobs, sdomains, tdomains
        
    def inference(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        #print('outshape',out.shape)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out





# if __name__=="__main__":
#     model = OurNet()
#     inputs = torch.Tensor(10,3,32,32)
    
#     print(inputs.shape)
#     model.test_shape(inputs)



