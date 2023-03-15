import numpy as np
import torch
import torchvision
import torch.nn as nn
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import yaml

import models
import load_datasets

def train(model, output_dir, dataloaders, tb_writer, model_name, lr=0.0005, num_epochs=200, device="cuda"):
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr) #effnet, vit = 0.0005
    criteron = nn.CrossEntropyLoss()
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    #scaler = torch.cuda.amp.GradScaler()
    #https://discuss.pytorch.org/t/optimizer-step-before-lr-scheduler-step-error-using-gradscaler/92930/7

    #early stopping
    val_loss = 99999.0

    early_stopping_patience = 5 #number of validation epochs that don't beat the best validation
    current_early_stopping_count = 0
    early_stop = False

    best_acc = 0
    best_epoch = 0

    for epoch in range(num_epochs):

        if early_stop:
            break

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                if (epoch+1) % 5 > 0:  #only every 5 epochs
                    continue

                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            count = 0
            dataset_size = 0
            for inputs, labels in dataloaders[phase]:
                dataset_size += inputs.size(0)

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad(set_to_none=True) #set_to_none=True maybe?

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    #with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criteron(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step(loss.item)

                # statistics
                running_loss += loss.item() #* inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                count += 1
                
            
            epoch_loss = running_loss / max(count, 1) #dataset_sizes[phase]
            epoch_acc = running_corrects.double().item() / max(dataset_size, 1) #dataset_sizes[phase]

            if phase == 'train':
                tb_writer.add_scalar('training loss', epoch_loss, epoch)
                tb_writer.add_scalar('training acc', epoch_acc, epoch)
                lr_scheduler.step(epoch_loss)
            elif phase == 'val':
                tb_writer.add_scalar('val loss', epoch_loss, epoch)
                tb_writer.add_scalar('val acc', epoch_acc, epoch)

            print(f'{model_name}: {phase} Epoch: {epoch} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            tb_writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch)

            # save best model
            if epoch > 10 and phase == 'val':

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    print(f"Saving at epoch: {epoch}")
                    best_epoch = epoch
                    torch.save(model.state_dict(), os.path.join(output_dir, 'best_checkpoint.pt'))

                if epoch_loss < val_loss:
                    val_loss = epoch_loss
                    current_early_stopping_count = 0
                else:
                    current_early_stopping_count += 1

                if current_early_stopping_count > early_stopping_patience:
                    early_stop = True
                    print(f"Early stopping at epoch: {epoch}")
                    break

    print("finished")
    
    torch.save(model.state_dict(), os.path.join(output_dir, 'final_checkpoint.pt'))
    print(f"Best epoch: {best_epoch}")
    print(f"Best acc: {best_acc}")

def test(model, output_dir, dataloaders, tb_writer, model_name, device="cuda"):

    state_dict = torch.load(os.path.join(output_dir, 'best_checkpoint.pt'))
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()   # Set model to evaluate mode
    
    with torch.no_grad():
        for phase in ['test', 'real_test']:

            running_corrects = 0

            # Iterate over data.
            count = 0
            dataset_size = 0
            for inputs, labels in dataloaders[phase]:
                dataset_size += inputs.size(0)

                inputs = inputs.to(device)
                labels = labels.to(device)

                #with torch.cuda.amp.autocast():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                # statistics
                running_corrects += torch.sum(preds == labels.data)
                count += 1
                
            epoch_acc = running_corrects.double() / max(dataset_size, 1)

            if phase == 'test':
                tb_writer.add_scalar('test acc', epoch_acc)
            elif phase == 'real_test':
                tb_writer.add_scalar('real_test acc', epoch_acc)

            print(f'{model_name}: {phase} Acc: {epoch_acc:.4f}')

def reset(model):

    if model is not None:
        del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    #TODO: gc.collect?
    

def main():
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    suffix = config["suffix"]
    train_class_size = config["train_class_size"]
    eval_class_size = config["eval_class_size"]
    num_classes = config["num_classes"]
    num_epochs = config["num_epochs"]
    real_train_class_size = config['real_train_class_size']
    real_eval_class_size = config['real_eval_class_size']

    is_mixed_datasets = (suffix == "mixed")

    for c in config["model_configs"]:

        if c['skip']:
            continue

        name = c['name']
        model_index = c['model_index']
        batch_size = c['batch_size']
        lr = c['lr']
        load_weights = c['load_weights']

        print(f"Config loaded: Name: {name} Learning Rate: {lr} Batch Size: {batch_size} Model Index: {model_index} Load Weights: {load_weights}")

        output_dir = os.path.join('output', f"{name}_{train_class_size}_{suffix}")

        isExist = os.path.exists(output_dir)
        if not isExist:
            os.makedirs(output_dir)

        if load_weights:
            model = models.getModel(model_index, os.path.join(output_dir, "best_checkpoint.pt"))
        else:
            model = models.getModel(model_index)

        dataloaders = None

        if is_mixed_datasets:
            dataloaders = load_datasets.load_data_mixed("dataset", 
                                num_classes=num_classes, 
                                batch_size=batch_size,
                                train_class_size=train_class_size,
                                eval_class_size=eval_class_size,
                                real_train_class_size=real_train_class_size,
                                real_eval_class_size=real_eval_class_size)
        else:
            dataloaders = load_datasets.load_data("dataset", 
                                num_classes=num_classes, 
                                batch_size=batch_size,
                                train_class_size=train_class_size,
                                eval_class_size=eval_class_size)
        
        writer = SummaryWriter(os.path.join(output_dir, 'output'))

        train(model, output_dir, dataloaders, writer, name, lr, num_epochs, device)
        test(model, output_dir, dataloaders, writer, name, device)

        writer.close()

        reset(model)


if __name__ == '__main__':
    main()