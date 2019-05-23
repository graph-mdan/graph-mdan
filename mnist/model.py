import torch.nn as nn
import torch
import torch.nn.functional as F
from gatlayer import GraphAttentionLayer,AdjacencyLayer,DistanceLayer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np




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
    
        self.alpha = configs['alpha']
        
        self.batch_size = configs["batch_size"]
     

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


    def inference(self, tinputs):
        
        th_relu = tinputs

        

        th_relu = self.layer1(th_relu)
        th_relu = self.layer2(th_relu)
        th_relu = self.layer3(th_relu)
        
        #print(th_relu.shape)
        th_relu = th_relu.reshape(th_relu.size(0), -1)
        th_relu = self.reduction(th_relu)

          

       
        logprobs = F.log_softmax(self.target_classifier(th_relu), dim=1)
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
