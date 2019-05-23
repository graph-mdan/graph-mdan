import torch.nn as nn
import torch
import torch.nn.functional as F
from gatlayer import GraphAttentionLayer,AdjacencyLayer,DistanceLayer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np


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


class Mdan(nn.Module):
    def __init__(self,configs):
        super(Mdan, self).__init__()
        self.device = configs['device']
        self.num_domains = configs['num_domains']
        self.num_classes = configs['num_classes']
        self.dropout = configs['dropout']
        self.nheads = configs["nheads"]
    
        self.alpha = configs['alpha']
        self.nheads = configs["nheads"]
        self.k = configs["k"]
        self.batch_size = configs["batch_size"]
        self.num_gat_hidden_layers = 1
        self.num_gat_neurons = configs["gat_hidden_layers"]
        self.adj = AdjacencyLayer(self.num_domains, self.batch_size, self.k, self.device).to(device)
        self.dist_f = DistanceLayer(self.num_domains, self.batch_size)
        self.gat_hiddens = nn.ModuleList([GAT(self.num_gat_neurons[i], self.num_gat_neurons[i+1], self.adj, self.dropout, self.alpha,
                                                                   self.nheads, self.num_domains, self.batch_size, self.k, 0, self.device)
                                          for i in range(self.num_gat_hidden_layers)])
        
        
        
        
        self.grls = [GradientReversalLayer() for _ in range(self.num_domains)]
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.reduction = nn.Linear(3200,512)
        
        #self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.target_classifier = nn.Sequential(
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_classes),
        )
        
        self.domain_classifier = nn.Sequential(
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        
        )

        self.domains = nn.ModuleList([self.domain_classifier for _ in range(self.num_domains)])

        
    
    def forward(self, sinputs, tinputs, slabels):
        sh_relu, th_relu = sinputs, tinputs
        #print('tinpusshape', tinputs.shape)
        for i in range(self.num_domains):
            sh_relu[i] = self.layer1(sh_relu[i])
            sh_relu[i] = self.layer2(sh_relu[i])
            sh_relu[i] = self.layer3(sh_relu[i])
            
            sh_relu[i] = sh_relu[i].reshape(sh_relu[i].size(0), -1)
            #print(sh_relu[i].shape)
            sh_relu[i] = self.reduction(sh_relu[i])
            
         
        
        th_relu = self.layer1(th_relu)
        th_relu = self.layer2(th_relu)
        th_relu = self.layer3(th_relu)
        
        #print(th_relu.shape)
        th_relu = th_relu.reshape(th_relu.size(0), -1)
        #print('th_relu.shape',th_relu.shape)
        th_relu = self.reduction(th_relu)


        # Graph Attentive Transferrable Learning
        x = [sh_relu[i] for i in range(self.num_domains)]# [num_domains, batch_size, num_features]
        x.append(th_relu) # [num_domains+1, batch_size, num_features]
        x = torch.stack(x) 
        x_ = x[0]
        for i in range(1, self.num_domains+1):
            x_ = torch.cat((x_, x[i]), 0)
        
        # compute adjacency matrix, GAT
        adj,_ = self.adj(x_) # N x N , N = (num_domains+1)*batch_size
        for gat_hidden in self.gat_hiddens:
            x_ = gat_hidden(x_, adj) # graph_emb should be [(num_domains+1) x batch_size, num_features]
        graph_emb = x_
        gsh_relu = []
        for i in range(self.num_domains):
            gsh_relu.append(graph_emb[i*self.batch_size:(i+1)*self.batch_size, :])
        
        gth_relu = graph_emb[self.num_domains*self.batch_size:(self.num_domains+1)*self.batch_size, :]



        # Classification probabilities on k source domains.
        logprobs = []
        for i in range(self.num_domains):
            #print(gsh_relu[i].shape)
            logprobs.append(F.log_softmax(self.target_classifier(sh_relu[i]), dim=1))
        # Domain classification accuracies.
        sdomains, tdomains = [], []
        for i in range(self.num_domains):
            sdomains.append(F.log_softmax(self.domains[i](self.grls[i](sh_relu[i])), dim=1))
            tdomains.append(F.log_softmax(self.domains[i](self.grls[i](th_relu)), dim=1))
        return logprobs, sdomains, tdomains


    def inference(self,sinputs, tinputs):
        # logprobs = []
        gth_relu_ = []
        th_relu_ = []
        
        size = tinputs.shape[0]
        
        input_sizes = [data.shape[0] for data in sinputs]
        for begin in range(0, tinputs.shape[0], self.batch_size):
            end = begin + self.batch_size
            end = min([end, size])
            if end-begin < self.batch_size: break

            # preparing batch
            sh_relu = []
            for i in range(self.num_domains):
                s_batch_idx = np.random.choice(input_sizes[i], self.batch_size)
                sh_relu.append( torch.tensor(sinputs[i][s_batch_idx, :], requires_grad=False))
            t_batch_idx = range(begin, end)
            th_relu = tinputs[t_batch_idx, :]

            # feature extractor Stage-I
            for i in range(self.num_domains):
                sh_relu[i] = self.layer1(sh_relu[i])
                sh_relu[i] = self.layer2(sh_relu[i])
                sh_relu[i] = self.layer3(sh_relu[i])
                
                sh_relu[i] = sh_relu[i].reshape(sh_relu[i].size(0), -1)
                sh_relu[i] = self.reduction(sh_relu[i])

            th_relu = self.layer1(th_relu)
            th_relu = self.layer2(th_relu)
            th_relu = self.layer3(th_relu)
            
            #print(th_relu.shape)
            th_relu = th_relu.reshape(th_relu.size(0), -1)
            th_relu = self.reduction(th_relu)

            # feature extractor Stage-II
            # propagation for tranduction
            x = [sh_relu[i] for i in range(self.num_domains)]# [num_domains, batch_size, num_features]
            x.append(th_relu) # [num_domains+1, batch_size, num_features]
            x = torch.stack(x)

            ## construct graph
            x_ = x[0]
            for i in range(1, self.num_domains+1):
                x_ = torch.cat((x_, x[i]), 0)
            self.adj =self.adj.cpu()
            adj_mtr,_ = self.adj(x_)
            for gat_hidden in self.gat_hiddens:
                x_ = gat_hidden(x_, adj_mtr)
            graph_emb = x_
            gth_relu = graph_emb[self.num_domains*self.batch_size:(self.num_domains+1)*self.batch_size, :]
            gth_relu_.append(gth_relu)
            th_relu_.append(th_relu)

            # Classification probabilities on target domains.
        gth_relu_ = torch.stack(gth_relu_).view(len(gth_relu_) * gth_relu_[0].size()[0], -1)
        #th_relu = torch.stack(th_relu_).view(len(th_relu_)*th_relu_[0].size()[0],-1)
        logprobs = F.log_softmax(self.target_classifier(gth_relu_), dim=1)
        # th_relu_ = torch.stack(th_relu).view(len(th_relu) * th_relu[0].size()[0], -1)
        # logprobs = F.log_softmax(self.softmax(F.relu(th_relu_)), dim=1)
        # logprobs = torch.stack(logprobs)
        return logprobs
        
       




if __name__ == '__main__':
    t = torch.rand((10,3,30,30))
    s = [torch.rand((10,3,30,30)) for _ in range(3)]
    model = Mdan(10)
    out1,out2, outt3 = model(s,t)
    out = model.inference(t)
    print(out.shape)
