import logging
import argparse
import log_setup
import os

#####
#import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from PIL import Image

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


def parse_args(argv=None):
    """
        Parse command line args
    """
    parser = argparse.ArgumentParser(description="Image Classifier Trainer")

    # positional arguments
    parser.add_argument('data_directory', 
                        action="store", 
                        help="Path to root folder containing and train/ test/ valid/ image folders")

    # optionals
    parser.add_argument('-v', 
                        action="store_true",
                        dest="verbose",
                        required=False,
                        help="Debug output")

    parser.add_argument('--gpu',
                        action="store_true",
                        dest="gpu_mode",
                        required=False, 
                        help="Use GPU for training")

    parser.add_argument('--arch "vgg13"',
                        dest="pretrainined model architecture",
                        default="vgg13",
                        required=False,
                        choices=['vgg13', 'aaa'],
                        help="Pretrained model architecture")

    parser.add_argument('--learning_rate',
                        dest="learning_rate",
                        default=0.01,
                        type=float,
                        required=False,
                        help="Use GPU for inference")

    parser.add_argument('--hidden_units',
                        dest="hidden_units",
                        default=512,
                        type=int,
                        required=False,
                        help="Use GPU for inference")

    parser.add_argument('--epochs',
                        dest="epochs",
                        default=3,
                        type=int,
                        required=False,
                        help="Number of times to train against the complete data sets")

    parser.add_argument('--save_dir',
                        dest="save_dir",
                        required=False,
                        help="Set directopry to save checkpoints")

    results = parser.parse_args(argv)
    return results

class Trainer(object):

    def __init__(self, data_dir):
        self._logger = logging.getLogger()
        self._data_dir = data_dir


    def _create_data_loader(self, categorized_path, batch_size, with_randomization):
        
        self._logger.info("Creating data set from folder '%s' random=%s", categorized_path, with_randomization)
        if with_randomization:
            transform_list = [transforms.RandomRotation(30),
                              transforms.RandomResizedCrop(224),
                              transforms.RandomHorizontalFlip()]
                              
        else:
            transform_list = [transforms.Resize(256),
                              transforms.CenterCrop(224)]

        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize([0.485, 0.456, 0.406], 
                                                   [0.229, 0.224, 0.225]))


        transform_set = transforms.Compose(transform_list)

        # Load the dataset with ImageFolder
        dataset = datasets.ImageFolder(categorized_path,
                                       transform=transform_set)
        self._logger.debug(dataset)
        # define the data loader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=with_randomization)
        self._logger.debug(dataloader)

        return dataloader

    def _create_data_loaders(self):

        validate_path = os.path.join(self._data_dir, "valid")
        validate_data_loader = self._create_data_loader(validate_path, batch_size=32,  with_randomization=False)

        test_path = os.path.join(self._data_dir, "valid")
        test_data_loader = self._create_data_loader(test_path, batch_size=32, with_randomization=False)

        train_path = os.path.join(self._data_dir, "valid")
        train_data_loader = self._create_data_loader(train_path, batch_size=64, with_randomization=True)
         

def main():

    arg_object = parse_args()

    if arg_object.verbose:
        log_setup.init(loglevel=logging.DEBUG)
    else:
        log_setup.init(loglevel=logging.INFO)

    logger = logging.getLogger()

    trainer = Trainer(arg_object.data_directory)
    trainer._create_data_loaders()

if __name__ == "__main__":
    main()
