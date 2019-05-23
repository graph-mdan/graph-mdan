import torch.nn as nn
import torch
import torch.nn.functional as F


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


            

#     def test_shape(self, x):
#         x = self.features(x)
#         print(x.shape)
#         #x = self.avgpool(x)
#         #print(x.shape)
#         x = x.view(x.size(0), 256 * 6 * 6)
#         print(x.shape)
#         x = self.classifier(x)
#         print(x.shape)



    

class OurNet(nn.Module):

    def __init__(self, num_classes=10):
        super(OurNet, self).__init__()
        self.num_domains = 3
        self.domains = nn.ModuleList([nn.Linear(2048, 2) for _ in range(self.num_domains)])
        self.alpha =0.2
        self.grls = [GradientReversalLayer() for _ in range(self.num_domains)]
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
        self.fc = nn.Linear(8*8*32, num_classes)

    def forward(self, sinputs, tinputs):
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



