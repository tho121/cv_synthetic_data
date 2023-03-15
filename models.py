import numpy as np
import torch
import torchvision
import torch.nn as nn
import os
from torchvision import datasets, transforms
from enum import Enum

num_classes = 3

def getModel(model_index, weights_path=None):
    model_funcs = [efficientNet, vit, convNext, swinV2]

    #don't throw error, calling  function doesn't know the size of model_funcs, just check for None
    if model_index >= len(model_funcs):
        return None

    return model_funcs[model_index](weights_path)

def efficientNet(weights_path=None):
    model = torchvision.models.efficientnet_v2_s(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path))

    print(f"Model loaded: efficientNet - weights_path: {weights_path}")
    return model

def convNext(weights_path=None):
    model = torchvision.models.convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path))

    print(f"Model loaded: convNext - weights_path: {weights_path}")
    return model

def vit(weights_path=None):

    weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    if weights_path is not None:
        weights = None

    model = torchvision.models.vit_b_16(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(
        nn.Linear(model.heads[0].in_features, model.heads[0].in_features),
        nn.ReLU(True),
        nn.Linear(model.heads[0].in_features, num_classes)
    )

    model.heads = classifier

    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path))

    print(f"Model loaded: vit - weights_path: {weights_path}")

    return model

def swinV2(weights_path=None):

    weights = torchvision.models.Swin_V2_T_Weights.DEFAULT
    if weights_path is not None:
        weights = None

    model = torchvision.models.swin_v2_t(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(
        nn.Linear(model.head.in_features, model.head.in_features),
        nn.ReLU(True),
        nn.Linear(model.head.in_features, num_classes)
    )

    model.head = classifier

    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path))

    print(f"Model loaded: swinV2 - weights_path: {weights_path}")

    return model