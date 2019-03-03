import logging
import argparse
import log_setup

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

def train():
    pass

def main():

    arg_object = parse_args()

    if arg_object.verbose:
        log_setup.init(loglevel=logging.DEBUG)
    else:
        log_setup.init(loglevel=logging.INFO)

    logger = logging.getLogger()

    train()

if __name__ == "__main__":
    main()
