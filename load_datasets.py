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

def get_real_data_subset_splits(dataset, num_classes=3, train_size=30, val_size=30, test_size=30):
    train_indices = []
    val_indices = []
    test_indices = []
    real_test_indices = []

    sum_train_val = train_size + val_size
    sum_train_val_test = sum_train_val + test_size

    #get random imgs based on train_class_size for each class
    for i in range(num_classes):
        c = torch.tensor([i])
        c_indices = torch.nonzero(torch.tensor(dataset.targets) == c).squeeze()

        c_indices = (torch.tensor(dataset.targets)[..., None] == c).any(-1).nonzero(as_tuple=True)[0]
        perm = torch.randperm(c_indices.size(0))

        train_indices.extend(c_indices[perm][:train_size].numpy())
        val_indices.extend(c_indices[perm][train_size:sum_train_val].numpy())
        test_indices.extend(c_indices[perm][sum_train_val:sum_train_val_test].numpy())
        real_test_indices.extend(c_indices[perm][sum_train_val_test:].numpy())

    subsets = {}
    subsets['train'] = torch.utils.data.Subset(dataset, train_indices)
    subsets['val'] = torch.utils.data.Subset(dataset, val_indices)
    subsets['test'] = torch.utils.data.Subset(dataset, test_indices)
    subsets['real_test'] = torch.utils.data.Subset(dataset, real_test_indices)

    return subsets

def load_data_mixed(dataset_paths, num_classes=3, batch_size=24, train_class_size=100, eval_class_size=100, real_train_class_size=30, real_eval_class_size=30):
    train_image_datasets = datasets.ImageFolder(os.path.join(dataset_paths, "train"), train_transforms)
    val_image_datasets = datasets.ImageFolder(os.path.join(dataset_paths, "val"), test_transforms)
    test_image_datasets = datasets.ImageFolder(os.path.join(dataset_paths, "test"), test_transforms)
    real_test_image_datasets = datasets.ImageFolder(os.path.join(dataset_paths, "real_test"), test_transforms)

    real_data_subsets = get_real_data_subset_splits(real_test_image_datasets, num_classes, 
                                                    train_size=real_train_class_size, 
                                                    val_size=real_eval_class_size,
                                                    test_size=real_eval_class_size)

    #get random imgs based on train_class_size for each class
    train_image_subset = get_subset(train_image_datasets, num_classes=num_classes, class_size=train_class_size)
    train_image_subset = torch.utils.data.ConcatDataset([train_image_subset, real_data_subsets['train']])

    train_loader = torch.utils.data.DataLoader(train_image_subset,
                                             batch_size=batch_size, shuffle=True,
                                             num_workers=2)
    
    val_image_subset = get_subset(val_image_datasets, num_classes=num_classes, class_size=eval_class_size)
    val_image_subset = torch.utils.data.ConcatDataset([val_image_subset, real_data_subsets['val']])

    val_loader = torch.utils.data.DataLoader(val_image_subset,
                                                batch_size=batch_size, shuffle=False,
                                                num_workers=2)
    
    test_image_subset = get_subset(test_image_datasets, num_classes=num_classes, class_size=eval_class_size)
    test_image_subset = torch.utils.data.ConcatDataset([test_image_subset, real_data_subsets['test']])

    test_loader = torch.utils.data.DataLoader(test_image_subset,
                                                batch_size=batch_size, shuffle=False,
                                                num_workers=2)
    
    #real_test_image_subset = get_subset(real_test_image_datasets, num_classes=num_classes, class_size=eval_class_size)
    real_test_loader = torch.utils.data.DataLoader(real_data_subsets['real_test'],
                                                batch_size=batch_size, shuffle=False,
                                                num_workers=2)

    loaders = {}
    loaders["train"] = train_loader
    loaders["val"] = val_loader
    loaders["test"] = test_loader
    loaders["real_test"] = real_test_loader

    return loaders