import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import yaml

from sklearn.manifold import TSNE
from torchvision import models
from torchvision.datasets import ImageFolder
from ultralytics import YOLO
from os.path import join, exists
from os import mkdir, path
from pathlib import Path
from collections import defaultdict


def get_vector(img, layer, model):

    '''
    Function adapted from:
    
        1. Medium (2017), "Extract a feature vector for any image with PyTorch".
        URL: https://becominghuman.ai/extract-a-feature-vector-for-any-image-with-pytorch-9717561d1d4c 

        2. Stack Overflow (2020), "How to extract feature vector from single image in Pytorch?".
        URL: https://stackoverflow.com/questions/63552044/how-to-extract-feature-vector-from-single-image-in-pytorch 

    Extract feature vector of PyTorch model from single image.

    Args:
        img (torch image): Image to extract feature vector from.
        layer (torch Module): Model layer to extract feature vector on.
        model (torch Module): Model to use for feature vector extraction.
    
    Returns:
        my_embedding (torch Tensor): Feature vector tensor.
    '''

    # 1. Create a PyTorch Variable with the transformed image
    t_img = img
    # 2. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of *2048*
    my_embedding = torch.zeros([2048])
    # 3. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.flatten())
    # 4. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # 5. Run the model on our transformed image
    with torch.no_grad():
        model(t_img.unsqueeze(0))
    # 6. Detach our copy function from the layer
    h.remove()
    # 7. Return the feature vector
    return my_embedding


def get_resnet101_embeddings(data_folder, layer, model):
  
    '''
    Get feature embeddings for images in data folder.

    Args:
        data_folder (torch ImageFolder): Images to extract feature vectors from.
        layer (torch Module): Model layer to extract feature vector on.
        model (torch Module): Model to use for feature vector extraction.

    Returns:
        embed_arr (np array): Array of feature vector embeddings for images in folder.
    '''

    size = len(data_folder)
    embed_arr = np.zeros((size, 2048))

    # get embedding for each image in dataset
    for i in range(size):
        img, label = data_folder[i]
        emb = get_vector(img, layer, model)
        embed_arr[i] = emb

    return embed_arr


def get_yolov8_embeddings(data_folder, model, layer=8):

    '''
    Get feature embeddings for images in data folder.

    Args:
        data_folder (torch ImageFolder): Images to extract feature vectors from.
        layer (int): Number of layer (0-8) to extract feature vectors from.
        model (YOLO model): Model to use for feature vector extraction.

    Returns:
        embed_arr (np array): Array of feature vector embeddings for images in folder.
    '''

    size = len(data_folder)
    embed_arr = np.zeros((size, 256))

    # get embedding for each image in dataset
    for i in range(size):
        img, label = data_folder[i]
        results = model.predict(img, embed=[layer])
        embed_arr[i] = results[0]

    return embed_arr


def initialize_resnet101_for_feature_extraction(
    model_path,
    n_cls=2
):
    
    '''
    Initialize ResNet101 model for feature vector extraction.

    Args:
        model_path (str): Complete path to trained model state dictionary.
        n_cls (int): Number of classes in the training dataset.

    Returns:
        model (torch Module): ResNet101 module initialized for prediction.
    '''

    # load model from PyTorch
    model = models.resnet101()

    # re-initialize final fully-connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, n_cls)

    # load fine-tuned state dict to device
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # get final layer to retrieve feature embeddings
    layer = model._modules.get('avgpool')

    return model, layer


def get_feature_vectors(
    data_dir,
    dataset_names,
    model_path,
    training_dataset_name=None,
    show_plot=False,
    model_type='ResNet101'
):
    
    '''
    Extract and visualize feature embeddings from dataset images.

    Args: 
        data_dir (str): Complete path to data folder containing datasets.
        dataset_names (list): List of dataset names in data_dir to extract embeddings.
        model_path (str): Complete path to fine-tuned model state dict.
        training_dataset_name (str): Name of dataset used to fine-tune model.
        show_plot (bool): Whether to display TSNE features plot in addition to saving.
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

        # resize, convert image to tensor, and normalize
        transforms = [
            transforms.Resize((224,224)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=train_mean, std=train_std_dev)
        ]

        # initialize model for feature extraction
        model, layer = initialize_resnet101_for_feature_extraction(model_path)

    elif model_type == 'YOLOv8':
        model = YOLO(model_path)

    # iterate through datasets
    embeddings_dict = defaultdict()
    for dataset_name in dataset_names:

        embeddings_dict[dataset_name] = defaultdict()

        test_dir = join(data_dir, dataset_name, 'test')

        # iterate through class labels
        for class_label in ['0', '1']:

            class_dir = join(test_dir, class_label)
            
            if model_type == 'ResNet101':

                # create image folder
                data_folder = ImageFolder(class_dir, transforms.Compose(transforms))

                # get embeddings
                image_embeddings = get_resnet101_embeddings(data_folder, layer, model)
            
            elif model_type == 'YOLOv8':

                # create image folder
                data_folder = ImageFolder(class_dir)

                # get embeddings
                image_embeddings = get_yolov8_embeddings(data_folder, model)

            # save feature embeddings by dataset and class
            embeddings_dict[dataset_name][class_label] = image_embeddings

    # perform TSNE to reduce embeddings dimensions separately for each dataset
    # other option: perform TSNE on all embeddings together
    tsne = TSNE(n_components=2)

    # create subplot for each dataset
    fig, axes = plt.subplots(1, len(dataset_names), figsize=(20,4), squeeze=False)
    axes = axes.ravel()
    plt.subplots_adjust(wspace=0.15)

    # make plot for each dataset
    for data_idx, (dataset_name, dataset_dict) in enumerate(embeddings_dict.items()):

        ax = axes[data_idx]

        # get array of embeddings from both classes
        embeddings_0 = dataset_dict['0']
        embeddings_1 = dataset_dict['1']
        dataset_embeddings = np.vstack((embeddings_0, embeddings_1))

        # reduce embeddings to 2 dims with TSNE
        tsne_embeddings = tsne.fit_transform(dataset_embeddings)

        # plot each class embeddings on dataset plot
        size_0 = embeddings_0.shape[0]

        ax.scatter(
            tsne_embeddings[:size_0,0], 
            tsne_embeddings[:size_0,1], 
            label='healthy', 
            color='limegreen'
        )

        ax.scatter(
            tsne_embeddings[size_0:,0], 
            tsne_embeddings[size_0:,1], 
            label='glaucoma', 
            color='darkgreen'
        )

        ax.legend()
        ax.set_title(dataset_name)

    # directory to save plot
    results_dir = join(Path(__file__).parent,'results')
    if not exists(results_dir):
        mkdir(results_dir)

    model_name = path.basename(model_path)

    # save and (optionally) display plot
    fig.suptitle(f"t-SNE Projections for {model_name} Feature Embeddings", y=0.99, fontsize=14, fontweight='bold')
    plt.savefig(join(results_dir, f'{model_name}_feature_embeddings_plot.png'))
    if show_plot:
        plt.show()
