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
                        dest="pretrained_arch",
                        default="densenet121",
                        required=False,
                        choices=['densenet121', 'vgg13', 'vgg16'],
                        help="Pretrained model architecture")

    parser.add_argument('--learning_rate',
                        dest="learning_rate",
                        default=0.001,
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

    def __init__(self, data_dir, hidden_layer_units, learning_rate):
        self._logger = logging.getLogger()
        self._data_dir = data_dir
        self._hidden_layer_units = hidden_layer_units
        self._criterion = nn.NLLLoss()
        self._learning_rate = learning_rate
        self._model = None
        self._optimizer = None
        self._device = "cpu"

    def set_device(self, device):
        """
            mode should be cuda or cpu
        """
        assert device in ('cpu', 'cuda')
        self._device = device

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
        self._validate_data_loader = self._create_data_loader(validate_path, batch_size=32, with_randomization=False)

        test_path = os.path.join(self._data_dir, "test")
        self._test_data_loader = self._create_data_loader(test_path, batch_size=32, with_randomization=False)

        train_path = os.path.join(self._data_dir, "train")
        self._train_data_loader = self._create_data_loader(train_path, batch_size=64, with_randomization=True)

    def _create_classifier(self, in_count, out_count):
        classifier = nn.Sequential(OrderedDict([
            ('fullyconnected1', nn.Linear(in_count, self._hidden_layer_units)),
            ('relu', nn.ReLU()),
            ('fullyconnected2', nn.Linear(self._hidden_layer_units, out_count)),
            ('output', nn.LogSoftmax(dim=1))
            ]))
        self._logger.debug(classifier)
        return classifier

    def _create_model(self, pretrained_arch, out_features):
        self._logger.info("Loading pretrained %s", pretrained_arch)
        if pretrained_arch == "densenet121":
            self._model = models.densenet121(pretrained=True)
            in_count = 1024
        elif pretrained_arch == "vgg13":
            self._model = models.vgg13(pretrained=True)
            in_count = 25088
        elif pretrained_arch == "vgg16":
            self._model = models.vgg16(pretrained=True)
            in_count = 25088

        # Freeze parameters
        for parameter in self._model.parameters():
            parameter.requires_grad = False

        # Replace classifier
        self._model.classifier = self._create_classifier(in_count=in_count, out_count=out_features)
        self._optimizer = optim.Adam(self._model.classifier.parameters(),
                                     lr=self._learning_rate)

        self._logger.debug(self._model)
        self._logger.debug(self._optimizer)


    def validate_test_data(self):
        self._validate(self._test_data_loader)

    def validate_validate_data(self):
        self._validate(self._validate_data_loader)

    def _validate(self, dataloader):
        self._model.eval()
        with torch.no_grad():
            test_loss = 0
            accuracy = 0
            for images, labels in dataloader:

                images, labels = images.to(self._device), labels.to(self._device)

                self._logger.debug("forward pass")
                output = self._model.forward(images)
                test_loss += self._criterion(output, labels).item()

                ps = torch.exp(output)
                equality = (labels.data == ps.max(dim=1)[1])
                accuracy += equality.type(torch.FloatTensor).mean()
                if self._device == "cpu":
                    break
            self._logger.debug("loss: %s accuracy: %s", test_loss, accuracy)
            return test_loss, accuracy


def main():

    arg_object = parse_args()

    if arg_object.verbose:
        log_setup.init(loglevel=logging.DEBUG)
    else:
        log_setup.init(loglevel=logging.INFO)

    trainer = Trainer(arg_object.data_directory,
                      hidden_layer_units=int(arg_object.hidden_units),
                      learning_rate=float(arg_object.learning_rate))

    if arg_object.gpu_mode:
        trainer.set_device(device='cuda')

    trainer._create_data_loaders()
    trainer._create_model(pretrained_arch=arg_object.pretrained_arch, out_features=102)
    trainer.validate_test_data()

if __name__ == "__main__":
    main()
