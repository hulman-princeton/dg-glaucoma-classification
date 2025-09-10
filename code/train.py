import numpy as np
import matplotlib.pyplot as plt
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import sklearn.metrics

from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
from tempfile import TemporaryDirectory


def calc_data_moments(data_dir):

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


def load_data_from_dir(
        data_dir, 
        means, 
        std_devs, 
        batch_size=16, 
        shuffle=True
    ):

    '''
    Load data from directory into torch ImageFolder

    Args:
        data_dir (str): Absolute path to dataset directory containing 0 (healthy) and 1 (glaucoma) folders of retinal images.
        means (torch Tensor): RGB channel means to normalize dataset.
        std_devs (torch Tensor): RBG channel standard deviations to normalize dataset.
        batch_size (int): Batch size for DataLoader.
        shuffle (boolean): Whether to shuffle data in DataLoader.

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
    data_loader = DataLoader(dataset=data_folder, batch_size=batch_size, shuffle=shuffle)

    return data_loader


def train_model(
    model,
    criterion,
    optimizer,
    num_epochs,
    outpath
):
    
    '''
    Function from PyTorch (2017), "Transfer Learning for Computer Vision Tutorial".
    URL: https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html 

    Train deep learning model.

    Args:
        model (ModelClass): Model architecture with initial weights.
        criterion
        optimizer
        num_epochs (int): Number of epochs to train model.
        outpath (str): Where to save best model state dict.
    '''

    # TODO


def initialize_model():

    '''
    Initialize deep learning model to prepare for finetuning.
    '''

    # TODO