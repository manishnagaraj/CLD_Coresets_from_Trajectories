from .model_archs.resnet_K import *
from .model_archs.densenet import *
from .model_archs.vgg import *
from .model_archs.alexnet import *

import torchvision.models as models

def get_model(model_arch, dataset):
    if dataset.lower() == 'cifar100' or dataset.lower() == 'cifar10':
        num_classes = 100 if dataset.lower() == 'cifar100' else 10
        if model_arch == 'resnet18':
            return resnet18_k(num_classes)
        elif model_arch == 'resnet34':
            return resnet34_k(num_classes)
        elif model_arch == 'resnet50':
            return resnet50_k(num_classes)
        elif model_arch == 'resnet101':
            return resnet101_k(num_classes)
        elif model_arch == 'resnet152':
            return resnet152_k(num_classes)
        elif model_arch == 'vgg19_bn':
            return vgg19_bn(num_classes)
        elif model_arch == 'densenet':
            return densenet(
                num_classes = 100,
                depth = 100,
                growthRate = 12,
                compressionRate = 2,
                dropRate = 0,
            )
        elif model_arch == 'alexnet':
            return alexnet(num_classes)
        else:
            NotImplementedError
        
    elif dataset.lower() == 'imagenet':
        model = models.__dict__[model_arch]()
        return model
    
    else:
        NotImplementedError


    
    