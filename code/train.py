import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from tempfile import TemporaryDirectory
from ultralytics import YOLO
from os import mkdir
from os.path import join, exists
from pathlib import Path
from collections import defaultdict


def calculate_data_moments(data_dir):

    '''
    Calculate mean and standard deviation of retinal image dataset.

    Args:
        data_dir (str): Absolute path to dataset directory containing 0 (healthy) and 1 (glaucoma) folders of retinal images.
    
    Returns:
        means (torch Tensor): RGB channel means of image dataset.
        std_devs (torch Tensor): RGB channel standard deviations of image dataset.
    '''

    # resize images to 224 x 224 and convert to tensor
    basic_transforms = [transforms.Resize((224,224)), transforms.ToTensor()]

    # create torch image folder for dataset
    dataset = ImageFolder(data_dir, transforms.Compose(basic_transforms))

    # get channel means and standard deviations
    sum = torch.zeros(3)
    sum_sq = torch.zeros(3)
    total_pixels = 0

    # loop through images
    for img, _ in dataset:
        C, H, W = img.shape
        img = img.view(C, -1)  # flatten to channel dim
        sum += img.sum(dim=1)  # sum of intensities
        sum_sq += (img ** 2).sum(dim=1)  # sum of squared intensities
        total_pixels += H * W  # total pixels in image

    # calculate mean and std dev from sums
    means = sum / total_pixels
    std_devs = torch.sqrt(sum_sq / total_pixels - means ** 2)  # sqrt(E[X^2] - E[X]^2)

    return means, std_devs


def load_data_from_directory(
    data_dir, 
    means, 
    std_devs, 
    batch_size=16, 
    shuffle=True
):

    '''
    Load data from directory into torch ImageFolder.

    Args:
        data_dir (str): Absolute path to dataset directory containing 0 (healthy) and 1 (glaucoma) folders of retinal images.
        means (torch Tensor): RGB channel means to normalize dataset.
        std_devs (torch Tensor): RBG channel standard deviations to normalize dataset.
        batch_size (int or 'full'): Batch size for DataLoader.
        shuffle (bool): Whether to shuffle data in DataLoader.

    Returns:
        data_loader (torch DataLoader): DataLoader object containing transformed image batches.
    '''

    # resize, convert image to tensor, and normalize
    transforms = [
        transforms.Resize((224,224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=means, std=std_devs)
    ]

    # create image folder with transforms
    data_folder = ImageFolder(data_dir, transforms.Compose(transforms))

    # create data loader from image folder
    if batch_size == 'full':
        batch_size = len(data_folder)
    data_loader = DataLoader(dataset=data_folder, batch_size=batch_size, shuffle=shuffle)

    return data_loader


def initialize_resnet101_for_training(
    n_cls=2,
    lr=0.001,
    betas=(0.9,0.999)
):

    '''
    Initialize ResNet101 model for training.

    Args:
        n_cls (int): Number of classes in the training dataset.
        lr (float): Learning rate for Adam optimizer.
        betas (tuple[float,float]): Betas for Adam optimizer.

    Returns:
        model (torch Module): ResNet101 module initialized for fine-tuning.
        criterion (torch Module): Loss function to evaluate model accuracy during training.
        optimizer (torch Optimizer): Optimizer to update training step.
    '''

    # load model from PyTorch
    model = models.resnet101(weights='DEFAULT')

    # re-initialize final fully-connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, n_cls)

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=None)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas)

    return model, criterion, optimizer


def train_model(
    dataloaders,
    model,
    criterion,
    optimizer,
    output_dir,
    n_epochs=100,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
):
    
    '''
    Function from PyTorch (2017), "Transfer Learning for Computer Vision Tutorial".
    URL: https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html 

    Train deep learning model.

    Args:
        dataloaders (dict(torch DataLoader)): Dictionary of training and validation dataloaders.
        model (torch Module): Model initialized with default weights.
        criterion (torch Module): Loss function to evaluate model accuracy during training.
        optimizer (torch Optimizer): Optimizer to update training step.
        output_dir (str): Path to save model checkpoints.
        n_epochs (int): Number of epochs to train model.
        device (torch device): Device to train on (CPU or GPU).
    '''

    # send model to device
    model = model.to(device)

    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(n_epochs):
            print(f'Epoch {epoch}/{n_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc >= best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # save best model weights
        model.load_state_dict(torch.load(best_model_params_path))
        torch.save(model.state_dict(), output_dir)


def finetune_yolov8(
    data_dir,
    output_dir,
    batch_size=32,
    n_epochs=100,
    dim=224
):

    '''
    Initialize and train YOLOv8 model.

    Args:
        data_dir (str): Path to data folder.
        output_dir (str): Path to save model checkpoints.
        batch_size (int): Batch size for training.
        n_epochs (int): Number of epochs to train model.
        dim (int): Dimension of square training images.
    '''

    # initialize YOLOv8 model
    model = YOLO('yolov8n-cls.pt')

    # train model and save checkpoints to directory
    model.train(
        data=data_dir, 
        batch=batch_size, 
        epochs=n_epochs, 
        imgsz=dim,
        project=output_dir
    )


def finetune(
    data_dir,
    dataset_name,
    output_dir=None,
    model_type='ResNet101'
):
    
    '''
    Prepare data and fine-tune selected model.

    Args:
        data_dir (str): Path to data folder.
        dataset_name (str): Name of dataset.
        output_dir (str): Path to save model checkpoints.
        model_type (str): Type of deep learning model.
    '''

    # dataset directories
    dataset_dir = join(data_dir, dataset_name)

    if output_dir is None:
        output_dir = join(Path(__file__).parent, 'model_weights')
    if not exists(output_dir):
        mkdir(output_dir)

    dataset_output_dir = join(output_dir, dataset_name)

    if model_type == 'YOLOv8':

        # fine-tune YOLOv8
        finetune_yolov8(dataset_dir, dataset_output_dir)

    elif model_type == 'ResNet101':

        # get train and val paths
        train_dir = join(dataset_dir, 'train')
        val_dir = join(dataset_dir, 'val')

        # get training data channel means and std. devs.
        # load existing moments
        moments_path = join(output_dir, 'training_moments.yaml')
        with open(moments_path, "r") as f:
            training_moments = yaml.safe_load(f)

        # get dataset moments
        if dataset_name in training_moments:
            train_mean = training_moments[dataset_name]['mean']
            train_std_dev = training_moments[dataset_name]['std dev']

        # calculate training data moments if not in yaml
        else: 
            train_mean, train_std_dev = calculate_data_moments(train_dir)

            # save dataset moments to yaml
            training_moments[dataset_name]['mean'] = train_mean
            training_moments[dataset_name]['std dev'] = train_std_dev
            with open(moments_path, "w") as f:
                yaml.dump(training_moments, f, default_flow_style=False, sort_keys=False)


        # create train and val dataloaders
        # normalized by training data moments
        dataloaders = defaultdict()
        dataloaders['train'] = load_data_from_directory(train_dir, train_mean, train_std_dev)
        dataloaders['val'] = load_data_from_directory(val_dir, train_mean, train_std_dev)

        # initialize ResNet101
        model, criterion, optimizer = initialize_resnet101_for_training()

        # train model
        train_model(
            dataloaders,
            model,
            criterion,
            optimizer,
            dataset_output_dir
        )