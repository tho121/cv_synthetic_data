import numpy as np
import torch
import torchvision
import torch.nn as nn
import os
import torchshow
from torchvision import datasets, transforms
from PIL import Image
from cut_model import CUT_model

inf_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

output_transforms = transforms.Compose([
        transforms.Resize(224),
        #transforms.ToPILImage(),
    ])

def get_subset(dataset, num_classes=3):
    indices = []

    #get random imgs based on train_class_size for each class
    for i in range(num_classes):
        c = torch.tensor([i])
        c_indices = torch.nonzero(torch.tensor(dataset.targets) == c).squeeze()
        c_indices = (torch.tensor(dataset.targets)[..., None] == c).any(-1).nonzero(as_tuple=True)[0]
        indices.extend(c_indices.numpy())

    return torch.utils.data.Subset(dataset, indices)


def load_data(dataset_paths, num_classes=3, batch_size=1):
    train_image_datasets = datasets.ImageFolder(os.path.join(dataset_paths, "train"), inf_transforms)
    val_image_datasets = datasets.ImageFolder(os.path.join(dataset_paths, "val"), inf_transforms)
    test_image_datasets = datasets.ImageFolder(os.path.join(dataset_paths, "test"), inf_transforms)
 
    #get random imgs based on train_class_size for each class
    train_image_subset = get_subset(train_image_datasets, num_classes=num_classes)
    train_loader = torch.utils.data.DataLoader(train_image_subset,
                                             batch_size=batch_size, shuffle=False,
                                             num_workers=2)
    
    val_image_subset = get_subset(val_image_datasets, num_classes=num_classes)
    val_loader = torch.utils.data.DataLoader(val_image_subset,
                                                batch_size=batch_size, shuffle=False,
                                                num_workers=2)
    
    test_image_subset = get_subset(test_image_datasets, num_classes=num_classes)
    test_loader = torch.utils.data.DataLoader(test_image_subset,
                                                batch_size=batch_size, shuffle=False,
                                                num_workers=2)
    
    loaders = {}
    loaders["train"] = train_loader
    loaders["val"] = val_loader
    loaders["test"] = test_loader

    return loaders

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_it = 60

    cut_models = {
        0: ["bg", str(model_it)],
        1: ["dog", str(model_it)],
        2: ["turtle", str(model_it)],
        }

    cm = CUT_model(device=device)

    loaders = load_data("dataset")

    for c in range(len(cut_models)):
        cut_class, it = cut_models[c] 
        cm.load(f"cut_output_{cut_class}", f"e_{it}")

        for l in loaders:
            path = os.path.join("gan_dataset", l, cut_class)
            os.makedirs(path, exist_ok=True)
                         
            for i, (x,y) in enumerate(loaders[l]):
                if c == y.item():
                    x = x.to(device)

                    y_hat = cm.forward(x, None, is_train=False)
                    y_hat = output_transforms(y_hat) #.squeeze(dim=0)
                    img_path = os.path.join(path, f"gan_{i}.jpg")
                    torchshow.save(y_hat, img_path) #always saves image to 450x450, why?


if __name__ == '__main__':
    main()