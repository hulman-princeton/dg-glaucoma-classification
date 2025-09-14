import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import yaml
import pandas as pd

from torchvision import models
from ultralytics import YOLO
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from os.path import join, exists
from os import mkdir, path, listdir
from pathlib import Path

from code import train


def initialize_resnet101_for_prediction(
    model_path,
    n_cls=2,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
):
    
    '''
    Initialize ResNet101 model for prediction.

    Args:
        model_path (str): Complete path to trained model state dictionary.
        n_cls (int): Number of classes in the training dataset.
        device (torch device): Device to predict on.

    Returns:
        model (torch Module): ResNet101 module initialized for prediction.
    '''

    # load model from PyTorch
    model = models.resnet101()

    # re-initialize final fully-connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, n_cls)

    # load fine-tuned state dict to device
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    return model


def predict_resnet101_batch(
    dataloader,
    model,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
):
    
    '''
    Predict glaucoma from image batch with PyTorch model.

    Args:
        dataloader (torch DataLoader): Testing dataloader.
        model (torch Module): Model initialized with fine-tuned weights.
        device (torch device): Device to predict on.

    Returns:
        labels (np array): True image labels.
        preds (np array): Hard class predictions.
        probs (np array): Softmax probabilities of classes.
    '''

    was_training = model.training

    # evaluation mode
    model.eval()

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # model predicts on minibatch
            outputs = model(inputs)

            # convert outputs to softmax predictions
            probs = F.softmax(outputs, dim=1)

            # convert outputs to class predictions
            _, preds = torch.max(outputs, 1)

        model.train(mode=was_training)

    # convert to numpy
    labels = labels.cpu().numpy()
    preds = preds.cpu().numpy()
    probs = probs.cpu().numpy()

    return labels, preds, probs


def predict_yolov8_batch(
    test_dir,
    model
):
    
    '''
    Predict glaucoma from image batch with YOLO model.

    Args:
        test_dir (str): Path to dataset test folder.
        model (YOLO model): Model initialized with fine-tuned weights.

    Returns:
        labels (np array): True image labels.
        preds (np array): Hard class predictions.
        probs (np array): Softmax probabilities of classes.
    '''

    labels = []
    preds = []
    probs = []

    for class_label in ['0', '1']:

        # path to class images
        img_path = join(test_dir, class_label)

        # iterate through images
        for filename in listdir(img_path):

            # get model prediction
            results = model.predict(join(img_path, filename), imgsz=224)
            result = results[0]

            # compute softmax prediction
            prob = result.probs.data.tolist()

            # compute class prediction
            class_names = result.names
            pred = class_names[np.argmax(probs)].upper()

            # append predictions to lists
            labels.append(class_label)
            preds.append(pred)
            probs.append(prob)

    return np.array(labels), np.array(preds), np.array(probs)


def predict(
    data_dir,
    dataset_names,
    model_path,
    training_dataset_name=None,
    show_plot=False,
    model_type='ResNet101'
):
    
    '''
    Predict glaucoma from datasets and save performance metrics and ROC curves.

    Args: 
        data_dir (str): Complete path to data folder containing test datasets.
        dataset_names (list): List of dataset names in data_dir to predict glaucoma.
        model_path (str): Complete path to fine-tuned model state dict.
        training_dataset_name (str): Name of dataset used to fine-tune model.
        show_plot (bool): Whether to display ROC curves plot in addition to saving.
        model_type (str): Type of deep learning model.
    '''

    if model_type == 'ResNet101':
        # get training data channel means and std. devs.
        moments_path = join(Path(__file__).parent, 'model_weights', 'training_moments.yaml')
        with open(moments_path, "r") as f:
            training_moments = yaml.safe_load(f)

        # get training dataset moments if in folder
        if training_dataset_name is not None and training_dataset_name in training_moments:
            train_mean = training_moments[training_dataset_name]['mean']
            train_std_dev = training_moments[training_dataset_name]['std dev']
        # otherwise get imagenet moments
        else:
            train_mean = training_moments['IMAGENET']['mean']
            train_std_dev = training_moments['IMAGENET']['std dev']
        
        # initialize model for prediction
        model = initialize_resnet101_for_prediction(model_path)  
    
    elif model_type == 'YOLOv8':
        model = YOLO(model_path)

    # iterate through datasets
    df_rows = []
    plt.figure()
    for dataset_name in dataset_names:

        test_dir = join(data_dir, dataset_name, 'test')

        if model_type == 'ResNet101':
            # create test data loader
            test_loader = train.load_data_from_directory(
                test_dir, 
                train_mean, 
                train_std_dev, 
                batch_size='full',
                shuffle=False
            )

            # predict image classes
            labels, preds, probs = predict_resnet101_batch(test_loader, model)
        
        elif model_type == 'YOLOv8':
            # predict image classes
            labels, preds, probs = predict_yolov8_batch(test_dir, model)

        # compute accuracy, confusion matrix values, auc score
        acc = accuracy_score(labels, preds)
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel().tolist()
        auc = roc_auc_score(labels, probs[:,1])

        # save dataset row for dataframe
        df_rows.append([dataset_name, acc, tp, tn, fp, fn, auc])

        # compute variables for roc curve
        fpr, tpr, _ = roc_curve(labels, probs[:,1])

        # add roc curve to plot with AUC score
        plt.plot(fpr, tpr, label=f'{dataset_name} (AUC: {auc:.2f})')

    # directory to save results
    results_dir = join(Path(__file__).parent,'results')
    if not exists(results_dir):
        mkdir(results_dir)

    model_name = path.basename(model_path)

    # create prediction stats dataframe and save to results folder
    df = pd.DataFrame(df_rows, columns=['dataset','accuracy','tp','tn','fp','fn','auc'])
    df.to_csv(join(results_dir, f'{model_name}_prediction_statistics.csv'), index=False)

    # add plot titles and dotted line for random ROC curve
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} ROC Curves")
    plt.legend()

    # save and (optionally) display plot
    plt.savefig(join(results_dir, f'{model_name}_roc_plot.png'))
    if show_plot:
        plt.show()
