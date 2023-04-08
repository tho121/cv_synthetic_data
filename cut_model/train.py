import numpy as np
import torch
import torchvision
import torch.nn as nn
import os
import torchshow
from torchvision import datasets, transforms
from cut_model import CUT_model

EPOCHS = 150
load_model = False

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cm = CUT_model(device=device)

    if load_model:
        cm.load("cut_output")

    test_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        #transforms.RandomCrop((256, 256),pad_if_needed=True, padding_mode="reflect"),
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    resize_transform = transforms.Resize(224)

    dataset = datasets.ImageFolder("syn_bg", test_transforms)
    c = torch.tensor([0])
    c_indices = torch.nonzero(torch.tensor(dataset.targets) == c).squeeze()
    x_dataset = torch.utils.data.Subset(dataset, c_indices)

    c = torch.tensor([1])
    c_indices = torch.nonzero(torch.tensor(dataset.targets) == c).squeeze()
    y_dataset = torch.utils.data.Subset(dataset, c_indices)
    
    x_loader = torch.utils.data.DataLoader(x_dataset, batch_size=1, shuffle=True)
    y_loader = torch.utils.data.DataLoader(y_dataset, batch_size=1, shuffle=True)

    #rand_x = torch.rand((1, 3, 256, 256), device=device)
    #rand_y = torch.rand((1, 3, 256, 256), device=device)

    for e in range(EPOCHS):
        epoch_loss_d = 0.0
        epoch_loss_g = 0.0

        for i, (x, y) in enumerate(zip(x_loader, y_loader)):
            x = x[0].to(device)
            y = y[0].to(device)

            y_hat, y_idt, z_x, z_y, z_y_hat, z_y_idt = cm.forward(x, y, is_train=True)
            losses = cm.optimize(y, y_hat, z_x, z_y, z_y_hat, z_y_idt)

            epoch_loss_d += losses[0]
            epoch_loss_g += losses[1]

        epoch_loss_d /= (i + 1)
        epoch_loss_g /= (i + 1)

        print(f"Epoch: {e+1} - Losses: {epoch_loss_d} - {epoch_loss_g}")

        if (e+1) % 5 == 0:
            x = resize_transform(x)
            y = resize_transform(y)
            y_hat = resize_transform(y_hat)
            y_idt = resize_transform(y_idt)
            torchshow.save(torch.concat([x, y, y_hat, y_idt], dim=0),
                                os.path.join("cut_output", f"test_{e+1}.jpg"))
            
        if(e+1) % 20 == 0:
            cm.save("cut_output", f"e_{e+1}")

    cm.save("cut_output", f"e_{e+1}")


if __name__ == '__main__':
    main()