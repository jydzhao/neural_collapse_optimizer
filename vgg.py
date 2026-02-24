import torch.nn as nn

_cfg = {
    'VGG4': [64, 'M', 128, 'M', 32, 'M'],
    'VGG7_small': [64, 'M', 128, 'M', 256, 'M', 128, 'M'],
    'VGG7': [64, 'M', 256, 'M', 512, 'M', 128, 'M'],
    'VGG9': [64, 'M', 128, 'M', 512, 512, 'M', 256, 256, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}



class _VGG(nn.Module):
    """
    VGG module for 3x32x32 input, 10 classes
    """

    def __init__(self, name, last_layer_features, num_classes):

        super(_VGG, self).__init__()
        cfg = _cfg[name]
        self.llBN = nn.BatchNorm2d(num_features=last_layer_features)

        def _make_layers(self,cfg):
            layers = []
            in_channels = 3
            for ind,layer_cfg in enumerate(cfg):
                if layer_cfg == 'M':
                    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                else:
                    layers.append(nn.Conv2d(in_channels=in_channels,
                                            out_channels=layer_cfg,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1,
                                            bias=True))
                    if ind == len(cfg)-2:
                        layers.append(self.llBN)
                    else:
                        layers.append(nn.BatchNorm2d(num_features=layer_cfg))
                    layers.append(nn.ReLU(inplace=True))
                    in_channels = layer_cfg
            return nn.Sequential(*layers)
            
        
        self.layers = _make_layers(self, cfg)
        flatten_features = 512
        self.fc = nn.Linear(flatten_features, num_classes)
        # self.fc2 = nn.Linear(4096, 4096)
        # self.fc3 = nn.Linear(4096, 10)

    def forward(self, x):
        y = self.layers(x)
        y = y.view(y.size(0), -1)
        y = self.fc(y)
        # y = self.fc2(y)
        # y = self.fc3(y)
        return y


def VGG7(num_classes):
    return _VGG('VGG7', 128, num_classes)

def VGG7_small(num_classes):
    return _VGG('VGG7_small', 128, num_classes)

def VGG9(num_classes):
    return _VGG('VGG9', 256, num_classes)

def VGG11(num_classes):
    return _VGG('VGG11', 512, num_classes)


def VGG13(num_classes):
    return _VGG('VGG13', 512, num_classes)


def VGG16(num_classes):
    return _VGG('VGG16', 512, num_classes)


def VGG19(num_classes):
    return _VGG('VGG19', 512, num_classes)
