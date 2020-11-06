import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import utils
# from capsnet import CapsNetWithReconstruction, CapsNet, ReconstructionNet
 
class resnet_extractor(nn.Module):
    def __init__(self, input_channel, 
                    dropout = None, inp = "log-Filterbank-DCT-26-13",
                    nodes = [16, 32, 64, 128], num_layers = 2):
        
        super(resnet_extractor, self).__init__()
        
        import resnet
        self.model = resnet.ResNet(input_channel, resnet.ResidualBlock, nodes, num_layers, 
                                    dropout, inp)
        self.out_features = self.model.out_features

    def forward(self, input):
        output = self.model(input)
        return output

class abn_classifier(nn.Module):
    def __init__(self, in_features, nnClassCount, last_func = "softmax", dropout = None):
        
        super(abn_classifier, self).__init__()
        self.nnClassCount = nnClassCount
        self.in_features = in_features
        self.dropout = dropout
        self.last_func = last_func
        
        self.fc = nn.Linear(self.in_features, self.nnClassCount)
        nn.init.xavier_uniform_(self.fc.weight.data)
        if(self.last_func == "sigmoid"): self.sig = nn.Sigmoid()
        elif(last_func == "softmax"): self.sig = nn.Softmax(dim=1)

    def forward(self, input):
        out = self.fc(input)
        out = self.sig(out)
        return out
