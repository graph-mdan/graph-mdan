import torch.nn as nn
import torch
import torch.nn.functional as F
from gatlayer import GraphAttentionLayer,AdjacencyLayer,DistanceLayer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
import copy




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
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2),
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=2),
            #nn.BatchNorm2d(128),
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
        self.val_task = configs["val_task"]
        self.test_task = configs["test_task"]
        self.data_name = configs['data_name']
        ## meta-learn part 
        self.val_task_id = self.data_name.index(self.val_task)
        self.test_task_id = self.data_name.index(self.test_task)
        self.tr_task_id = list(range(len(self.data_name)))
        self.tr_task_id.remove(self.val_task_id)
        self.tr_task_id.remove(self.test_task_id)

        
        
    def forward(self,sinputs, tinputs, slabels):
        s_feat, s_label = [], []
        for i in range(len(self.data_name)):
            if i in self.tr_task_id:
                s_feat.append(sinputs[i])
                s_label.append(slabels[i])
            elif i == self.val_task_id:
                st_feat = sinputs[i]
                st_label = slabels[i]

        val_losses = []
        val_preds = []

        # compute all the loss
        tr_losses, _ = self.learner(s_feat, st_feat, s_label, is_train=True, counter=0)

        # compute gradients
        
        grad = torch.autograd.grad(tr_losses, list(self.state_dict().values()))
        gvs = dict(zip(self.state_dict().keys(), grad))
        fast_weights = dict(zip(self.state_dict().keys(), [self.state_dict()[key] - self.update_lr * gvs[key] for key in self.state_dict().keys()]))
        # fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.learner.parameters()))) # all the parameters
        self.load_state_dict(fast_weights)
        with torch.no_grad():
            # fast adaption
            val_loss, val_pred = self.learner(s_feat, st_feat, s_label, tlabels=st_label, is_train=False, weights=fast_weights, counter=1)
            val_losses.append(val_loss)
            val_preds.append(val_pred)

        # Inner Loop: continue to build T1-TK steps graph
        for _ in range(1, self.update_step): # i.e., num_updates = 4, update 3 times
            # T_k loss on meta-train
            # we need meta-train loss to fine-tune the task and meta-test loss to update theta
            tr_loss = self.adplearner(s_feat, st_feat, s_label)

            
            adp_weights = dict()
            for key in self.state_dict().keys() :
                if key[:6] == 'target':
                    adp_weights[key] = self.state_dict[key]
            grad = torch.autograd.grad(tr_loss, list(adp_weights.values()))

            # update theta_G and theta_y
            gvs = dict(zip(adp_weights.keys(), grad))
            fast_weights = dict(zip(self.state_dict.keys(), [self.state_dict[key] - self.update_lr * gvs[key] if key[:6] == 'target' else self.state_dict[key] for key in self.state_dict.keys()]))
            # fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.adplearner.parameters()))) # only partial the parameters
            self.load_state_dict(fast_weights)
            # forward on all the parameters Theta
            val_loss, val_pred = self.learner(s_feat, st_feat, s_label, tlabels=st_label, is_train=False,  counter=2)
            val_losses.append(val_loss)
            val_preds.append(val_pred)

        return val_losses[-1], val_preds[-1]




    def learner(self, sinputs, tinputs, slabels, tlabels=None, is_train=False, counter=0): # mdan entrance is forward
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


        # compute losses
        if is_train: # use k-1 sources losses: train_loss
            # Classification probabilities on k-1 source domains.
            logprobs = []
            for i in range(self.num_domains):
                logprobs.append(F.log_softmax(self.target_classifier(sh_relu[i])), dim=1)
            losses = torch.stack([F.nll_loss(logprobs[j].to(self.device), slabels[j].to(self.device)) for j in range(self.num_domains)]).to(self.device) # classification loss

            # Domain classification accuracies.
            sdlabels = torch.ones(self.batch_size, requires_grad=False).type(torch.LongTensor).to(self.device) # print (slabels) # 1
            tdlabels = torch.zeros(self.batch_size, requires_grad=False).type(torch.LongTensor).to(self.device) # print (tlabels) # 0
            sdomains, tdomains = [], []
            for i in range(self.num_domains): # 3 domains
                sdomains.append(F.log_softmax(self.domains[i](self.grls[i](sh_relu[i]))), dim=1)
                tdomains.append(F.log_softmax(self.domains[i](self.grls[i](th_relu))), dim=1)

            domain_losses = torch.stack([F.nll_loss(sdomains[j], sdlabels) +
                                        F.nll_loss(tdomains[j], tdlabels) for j in range(self.num_domains)]).to(self.device) # here is losses of a pair <sdomains, tdomains>

            

            train_losses = torch.log(torch.sum(torch.exp(self.gamma * (losses + self.mu * domain_losses)))) / self.gamma 
            train_pred = None

            return train_losses, train_pred

        else: # use the k-th source losses: val_loss
            # Classification probabilities
            logprobs = F.log_softmax(self.target_classifier(th_relu), dim=1)
            losses = F.nll_loss(logprobs, tlabels)


            val_losses = torch.log(torch.sum(torch.exp(self.gamma * (losses)))) / self.gamma 
            val_pred = torch.max(logprobs, 1)[1]
            return val_losses, val_pred

    def adplearner(self, sinputs, tinputs, slabels): # mdan entrance is forward
        """
        :param sinputs:     A list of inputs from k source domains.
        :param tinputs:     Input from the target domain.
        :param labels:     A list of labels from k source domains.
        :param is_train:    used which labels for loss
        :param weights:     all parameters
        :return:
        """

        sh_relu, th_relu = copy.copy(sinputs), copy.copy(tinputs)

        # oringinal feature extractor
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
            logprobs.append(F.log_softmax(self.target_classifier(sh_relu[i])), dim=1)

        # source domain losses
        losses = torch.stack([F.nll_loss(logprobs[j], slabels[j]) for j in range(self.num_domains)]).to(self.device) # classification loss

        adp_loss = torch.log(torch.sum(torch.exp(self.gamma * (losses)))) / self.gamma 
        return adp_loss

    

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
    configs = {"num_classes": 10,
               "num_epochs":5, "batch_size": 5, "lr": 1e-1, "mu": 10, "num_domains":
                   3,  "gamma": 10.0,  "lambda": 0.01, 'margin':0.1, 'dropout':0,'k':2,'alpha':0.2, 'device':device, 
                 "update_lr": 0.05, "meta_lr": 0.05, "update_step": 4 }
    configs["data_name"] = ["MNIST_M", "MNIST", "SVHN", "SYNTHDIGITS"]
    configs["test_task"] = "SVHN"
    configs["val_task"] = "MNIST_M"
    model = Mdan(configs)
   
    
    print(model.state_dict().keys())
