import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from PIL import Image
import torch
from torch import nn
import model.model as m
import torchvision
from torch.utils.data import Dataset
import pandas as pd
from random import randint
import shutil

sizes={
    "resnet101":224,
    "resnet152":224,
    "efficientnet-b0":220,
    "efficientnet-b4":380,
    "efficientnet-b6":528,
    "swin_tiny":224,
    "swin_base":224,
    "swin_large":224,
    'vit_base':224,
    'vit_large':224,
}

batch_sizes={
    "resnet101":180,
    "resnet152":112,
    "efficientnet_b0":256,
    "efficientnet_b4":32,
    "swin_tiny":200,
    "swin_small":128,
    "swin_base":96,
    "swin_large":60,
    'vit_base':144,
    'vit_large':48,
    'pvt_small':144,
    'deit_small':256,
    'deit_base': 144,
}


def net_init(model, device, freeze=False, pretrained=None, num_classes=2):
    net_name = model.split("net")[0]

    if net_name == "efficient":
        if pretrained is None:
            net = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_'+model.replace("-","_"), pretrained=True)
            if freeze:
                for param in net.parameters():
                    param.requires_grad = False
            net.classifier.fc = nn.Linear(net.classifier.fc.in_features, num_classes)
        else:
            net = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_'+model.replace("-","_"), pretrained=False)
            net.classifier.fc = nn.Linear(net.classifier.fc.in_features, num_classes)
            net.load_state_dict(torch.load(pretrained, map_location=device))

    elif net_name == "res":
        net = eval("m."+model)
        if pretrained is None:
            net = net()
            net.load_state_dict(torch.load(fr"./pretrain/{model}.pth", map_location=device))
            if freeze:
                for param in net.parameters():
                    param.requires_grad = False
            net.fc = nn.Linear(net.fc.in_features, num_classes)
        else:
            net = net(num_classes=num_classes)    
            net.load_state_dict(torch.load(pretrained, map_location=device))
    
    elif net_name[:4] == "swin":
        try:
            net = eval("m."+model)
        except:
            net = eval("m."+model)
        if pretrained is None:
            net = net()
            net.load_state_dict(torch.load(fr"./pretrain/{model}.pth", map_location=device), strict=False)
            if freeze:
                for param in net.parameters():
                    param.requires_grad = False
            net.head = nn.Linear(net.head.in_features, num_classes)
        else:
            net = net(num_classes=num_classes)    
            net.load_state_dict(torch.load(pretrained, map_location=device))

    elif net_name[:3] == "vit":
        net = eval("m."+model+"_16")
        if pretrained is None:
            net = net()
            net.load_state_dict(torch.load(fr"./pretrain/{model}.pth", map_location=device), strict=False)
            if freeze:
                for param in net.parameters():
                    param.requires_grad = False
            net.head = nn.Linear(net.head.in_features, num_classes)
        else:
            net = net(num_classes=num_classes)    
            net.load_state_dict(torch.load(pretrained, map_location=device))

    elif net_name == "res152_Swinb":
        net = eval("m."+model)
        if pretrained is None:
            net = net()
            net.load_state_dict(torch.load(r"./pretrain/resnet152-pre.pth", map_location=device), strict=False)
            if freeze:
                for param in net.parameters():
                    param.requires_grad = False
            net.fc = nn.Linear(net.fc.in_features, num_classes)
        else:
            net = net(num_classes=num_classes)    
            net.load_state_dict(torch.load(pretrained, map_location=device))

    return net.eval().to(device)


def label_get(path, label):
    try:
        label.insert(0, "ID")
    except:
        label = ["ID", label]
        
    prompt = pd.read_excel(path)
    prompt.replace("", float("NaN"), inplace=True)
    prompt.dropna(how='any', axis=1, inplace=True)

    return np.array(prompt[label])

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_paths(path):

    paths = []
    listdir(path, paths)
    paths = np.array(paths)

    steps = len(paths)
    index = np.arange(steps)
    np.random.shuffle(index)
    paths_val = paths[index[:paths.shape[0]//10]]
    paths = paths[index[paths.shape[0]//10:]]
    return paths, paths_val, steps


def listdir(path, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)

def trainset_sample(*args, mode='down'):
    data = [list(i) for i in args]
    l = np.array([len(i) for i in data])
    dataset = np.array([])

    if mode=='down':
        for i in range(len(data)):
            idx = np.arange(l[i])
            np.random.shuffle(idx)
            dataset = np.append(dataset, np.array(data[i])[idx[:np.min(l)]])
        return dataset
    
    elif mode=='up':
        m = max(l)
        for i in range(len(data)):
            dataset = np.append(dataset, data[i]*(m//l[i]))
            num = m % l[i]
            if num:
                idx = np.arange(l[i])
                np.random.shuffle(idx)
                dataset = np.append(dataset, np.array(data[i])[idx[:num]])
        return dataset
        
def get_it(paths:np.ndarray, bs:int, idx:int):
    try:
        paths = paths[bs*idx:bs*(idx+1)]
    except:
        paths = paths[bs*idx:]
    
    image = []
    mask = []
    for path in paths:
        data = np.load(path)
        img, msk = data['image'], data['mask']
        img = np.stack([img,img,img], axis=0)
        image.append(img)
        mask.append(1 if np.sum(msk)>20 else 0)

    image = torch.tensor(image).type(torch.LongTensor)
    image = image/255

    return image, mask


def create_folder(path: str, empty: bool = True):
    if empty and os.path.exists(path):
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.makedirs(path)