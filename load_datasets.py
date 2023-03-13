import torch
from torchvision import datasets, transforms
import os

train_transforms = transforms.Compose([
        #transforms.RandomSizedCrop(224),
        transforms.RandomCrop(224, 32),
        transforms.RandomRotation((-90,90)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def get_subset(dataset, num_classes=3, class_size=100):
    indices = []

    #get random imgs based on train_class_size for each class
    for i in range(num_classes):
        c = torch.tensor([i])
        c_indices = torch.nonzero(torch.tensor(dataset.targets) == c).squeeze()

        c_indices = (torch.tensor(dataset.targets)[..., None] == c).any(-1).nonzero(as_tuple=True)[0]
        perm = torch.randperm(c_indices.size(0))
        indices.extend(c_indices[perm][:class_size].numpy())

    return torch.utils.data.Subset(dataset, indices)


def load_data(dataset_paths, num_classes=3, batch_size=24, train_class_size=100, eval_class_size=100):
    train_image_datasets = datasets.ImageFolder(os.path.join(dataset_paths, "train"), train_transforms)
    val_image_datasets = datasets.ImageFolder(os.path.join(dataset_paths, "val"), test_transforms)
    test_image_datasets = datasets.ImageFolder(os.path.join(dataset_paths, "test"), test_transforms)
    real_test_image_datasets = datasets.ImageFolder(os.path.join(dataset_paths, "real_test"), test_transforms)

    #get random imgs based on train_class_size for each class
    train_image_subset = get_subset(train_image_datasets, num_classes=num_classes, class_size=train_class_size)
    train_loader = torch.utils.data.DataLoader(train_image_subset,
                                             batch_size=batch_size, shuffle=True,
                                             num_workers=2)
    
    val_image_subset = get_subset(val_image_datasets, num_classes=num_classes, class_size=eval_class_size)
    val_loader = torch.utils.data.DataLoader(val_image_subset,
                                                batch_size=batch_size, shuffle=False,
                                                num_workers=2)
    
    test_image_subset = get_subset(test_image_datasets, num_classes=num_classes, class_size=eval_class_size)
    test_loader = torch.utils.data.DataLoader(test_image_subset,
                                                batch_size=batch_size, shuffle=False,
                                                num_workers=2)
    
    real_test_image_subset = get_subset(real_test_image_datasets, num_classes=num_classes, class_size=eval_class_size)
    real_test_loader = torch.utils.data.DataLoader(real_test_image_subset,
                                                batch_size=batch_size, shuffle=False,
                                                num_workers=2)

    loaders = {}
    loaders["train"] = train_loader
    loaders["val"] = val_loader
    loaders["test"] = test_loader
    loaders["real_test"] = real_test_loader

    return loaders

def load_data_mix():
    #TODO: make datasets with real data mixed in
    pass