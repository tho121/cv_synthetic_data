import numpy as np
import torch
import torchvision
import torch.nn as nn
import os
from torchvision import datasets, transforms

num_classes = 3

def allModels():
    models = []

    models.append(efficientNet("output_eff_1000/best_checkpoint.pt"))
    models.append(convNext())
    models.append(vit("output_vit_1000/best_checkpoint.pt"))
    models.append(swinV2("output_swin_1000/best_checkpoint.pt"))

    return models

def efficientNet(weights_path=None):
    model = torchvision.models.efficientnet_v2_s(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path))

    return model

def convNext(weights_path=None):
    model = torchvision.models.convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path))

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

    return model