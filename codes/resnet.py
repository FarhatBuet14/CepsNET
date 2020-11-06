import torch.nn as nn

def conv3x3(in_channels, out_channels, stride=1):
    lay =  nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)
    nn.init.xavier_uniform_(lay.weight.data)
    return lay

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout = None):
        super(ResidualBlock, self).__init__()
        self.dropout = dropout
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        nn.init.ones_(self.bn1.weight)
        self.relu = nn.ReLU(inplace=True)
        if(self.dropout): self.drop = nn.Dropout(self.dropout)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        nn.init.ones_(self.bn2.weight)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if(self.dropout): out = self.drop(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        if(self.dropout): out = self.drop(out)
        return out

class ResNet(nn.Module):
    def __init__(self, input_channel, block, nodes, num_layers, 
                dropout = None, inp = "log-Filterbank-DCT-26-13"):
        
        super(ResNet, self).__init__()
        self.input_channel = input_channel
        self.in_channels = 16
        self.dropout = dropout
        self.inp = inp
        self.conv = conv3x3(self.input_channel, 16)
        self.bn = nn.BatchNorm2d(16)
        nn.init.ones_(self.bn.weight)
        self.relu = nn.ReLU(inplace=True)
        if(self.dropout): self.drop1 = nn.Dropout(self.dropout)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
        if("-merged_flat-" not in self.inp):
            self.layer1 = self.make_layer(block, 16, num_layers, stride = 1)
            self.layer2 = self.make_layer(block, 32, num_layers, stride = 2)
            self.layer3 = self.make_layer(block, 64, num_layers, stride = 2)
            self.layer4 = self.make_layer(block, 128, num_layers, stride = 2)
        else:
            self.conv_block_module = []
            for idx, node in enumerate(nodes):
                self.conv_block_module.append(self.make_layer(block, node, num_layers, stride = 1 if(idx==0) else 2))
            self.conv_block_module = nn.Sequential(*self.conv_block_module)
        self.avg_pool = nn.AvgPool2d(2)
        if("_flat" not in inp): self.out_features = 1920 if("13" in inp) else 3840
        else:
            feat = int(inp[len(inp)-1])
            if(inp[:2] == "13"): self.out_features = (1920 * 2) if(feat==2) else (1920 * (feat-1))
            else: self.out_features = (1920 * 5) if(feat==3) else (1920 * (feat+1))
            

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, self.dropout))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels, stride=1, downsample=None, dropout = self.dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        if(self.dropout): out = self.drop1(out)
        out = self.max_pool(out)
        if("-merged_flat-" not in self.inp):
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
        else: out = self.conv_block_module(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        return out