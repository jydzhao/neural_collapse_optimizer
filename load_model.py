import torch.nn as nn
import torchvision.models as models

from vgg import VGG11, VGG9
from vit import ViTForClassfication
from resnet import ResNet9

def load_model(model_name, dataset_name, device, pretrained_weights='none', finetune='all'):
    '''
    Load model through model_name and load it on the specified device (cpu or cuda)
    '''
    if dataset_name == 'cifar100':
        num_classes = 100
    else:
        num_classes = 10
            
    if model_name == 'ResNet18':
        # ResNet model
        width = 512
        
        if pretrained_weights != 'none':
            model = models.resnet18(num_classes=num_classes, weights=pretrained_weights).to(device)
        else:
            model = models.resnet18(num_classes=num_classes, weights=None).to(device)
    
        # modify ResNet if input has dimensions of 1x28x28
        if dataset_name == 'mnist' or dataset_name == 'fashion':
            model.conv1 = nn.Conv2d(1, model.conv1.weight.shape[0], 3, 1, 1, bias=False)
            # model.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
    elif model_name == 'ResNet9':
        
        if dataset_name == 'mnist' or dataset_name == 'fashion':
            in_channels = 1
        elif dataset_name == 'cifar10' or dataset_name == 'cifar100':
            in_channels = 3
        
        width = 512
        model = ResNet9(in_channels, num_classes=num_classes) 
    elif model_name == 'VGG11':
        width = 512
        model = VGG11(num_classes).to(device)
    
        # modify VGG if input has dimensions of 1x28x28
        if dataset_name == 'mnist' or dataset_name == 'fashion':
            model.layers[0] = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
    elif model_name == 'VGG9':
        width = 1024
        model = VGG9(num_classes).to(device)
    
        # modify VGG if input has dimensions of 1x28x28
        if dataset_name == 'mnist' or dataset_name == 'fashion':
            model.layers[0] = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
    elif model_name == 'ViT_custom':
        width = 512
        if dataset_name == 'mnist' or dataset_name == 'fashion':
            num_channels = 1
        elif dataset_name == 'cifar10' or dataset_name == 'cifar100':
            num_channels = 3
        config = {
            "patch_size": 4,  # Input image size: 32x32 -> 8x8 patches
            "hidden_size": width,
            "num_hidden_layers": 6,
            "num_attention_heads": 8,
            "intermediate_size": width, # 4 * hidden_size
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "initializer_range": 0.02,
            "image_size": 32,
            "num_classes": num_classes, # num_classes of CIFAR10
            "num_channels": num_channels,
            "qkv_bias": False,
            "use_faster_attention": False,
        }
        # These are not hard constraints, but are used to prevent misconfigurations
        assert config["hidden_size"] % config["num_attention_heads"] == 0
        # assert config['intermediate_size'] == 4 * config['hidden_size']
        assert config['image_size'] % config['patch_size'] == 0
        
        model = ViTForClassfication(config)
    elif model_name == 'ViT_B_16':

        if pretrained_weights != 'none':
            model = models.vit_b_16(weights=pretrained_weights).to(device)
            model.heads = nn.Sequential(nn.Linear(model.heads.head.in_features, 10))

            if finetune == 'last_layer':# Freeze all layers
                
                print('Finetuning only last layer, freezing other parameters.')
                for param in model.parameters():
                    param.requires_grad = False
                
                # Unfreeze the last encoder layer and the head
                for param in model.encoder.layers[-1].parameters():
                    param.requires_grad = True
                for param in model.heads.parameters():
                    param.requires_grad = True
        else:
            image_sz = 32
            
            model = models.vit_b_16(num_classes=10, weights=None, image_size=image_sz).to(device)
                    
        print('heads_layer', model.heads[0])
        width = model.heads[0].weight.shape[1]
        print(width)
        model.fc = model.heads[0]
        
    
        if dataset_name == 'mnist' or dataset_name == 'fashion':
    
            model.conv_proj = nn.Conv2d(1, model.conv_proj.out_channels,
                                        kernel_size=model.conv_proj.kernel_size,
                                        stride=model.conv_proj.stride,
                                        padding=model.conv_proj.padding,
                                        bias=model.conv_proj.bias is not None).to(device)
    else:
        raise ValueError(f'Unknown model_name {model_name}')
    
        
    # model.apply(lambda m: init_weights(m, gain=std_gain))
    model.to(device)

    num_param = sum([len(p.flatten()) for p in model.parameters()])
    print(f'num param in {model_name}: {num_param}, on {device}')
    
    return model, width