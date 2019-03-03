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

DEVELOPMENT_MODE=False

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
                        default="./",
                        help="Set directopry to save checkpoints")

    parser.add_argument('--checkpoint',
                        dest="checkpoint",
                        required=False,
                        help="Resume Training a Checkpoint")

    results = parser.parse_args(argv)
    return results

class Trainer(object):

    def __init__(self):
        self._logger = logging.getLogger()
        self._criterion = nn.NLLLoss()
        self._model = None
        self._device = "cpu"
        self._validate_data_loader = None
        self._test_data_loader = None
        self._train_data_loader = None

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

    def _create_data_loaders(self, data_dir):

        validate_path = os.path.join(data_dir, "valid")
        self._validate_data_loader = self._create_data_loader(validate_path, batch_size=32, with_randomization=False)
        self._logger.debug(self._validate_data_loader)

        test_path = os.path.join(data_dir, "test")
        self._test_data_loader = self._create_data_loader(test_path, batch_size=32, with_randomization=False)
        self._logger.debug(self._test_data_loader)

        train_path = os.path.join(data_dir, "train")
        self._train_data_loader = self._create_data_loader(train_path, batch_size=64, with_randomization=True)
        self._logger.debug(self._train_data_loader)

    def _create_classifier(self, in_count, hidden_count, out_count):
        classifier = nn.Sequential(OrderedDict([
            ('fullyconnected1', nn.Linear(in_count, hidden_count)),
            ('relu', nn.ReLU()),
            ('fullyconnected2', nn.Linear(hidden_count, out_count)),
            ('output', nn.LogSoftmax(dim=1))
            ]))
        self._logger.debug(classifier)
        return classifier

    def _create_model(self, pretrained_arch, hidden_features, out_features):
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
        self._model.classifier = self._create_classifier(in_count=in_count,
                                                         hidden_count=hidden_features,
                                                         out_count=out_features)
        # store params for saving the model
        self._arch = pretrained_arch
        self._hidden_features = hidden_features
        self._out_features = out_features

    def validate_test_data(self):
        self._logger.info("validating model against images in test/")
        loss, accuracy = self._validate(self._test_data_loader)
        self._logger.debug("Test loss: %s Test accuracy: %s",
                           loss/len(self._test_data_loader),
                           accuracy/len(self._test_data_loader))

    def validate_validate_data(self):
        self._logger.info("validating model against images in valid/")
        loss, accuracy = self._validate(self._validate_data_loader)
        self._logger.debug("Validation loss: %s Validation accuracy: %s",
                           loss/len(self._validate_data_loader),
                           accuracy/len(self._validate_data_loader))
        return loss, accuracy

    def _validate(self, dataloader):

        # Make sure network is in eval mode for inference
        # to ensure dropout is turned off
        self._model.eval()

        self._logger.debug("moving model to %s", self._device)
        self._model.to(self._device)
        # Turn off all gradients for all tensors during
        # validation, saves memory and speedsd up computations
        with torch.no_grad():
            loss = 0
            accuracy = 0
            for images, labels in dataloader:

                self._logger.debug("moving validation images and labels to %s", self._device)
                images, labels = images.to(self._device), labels.to(self._device)

                self._logger.debug("forward pass")
                output = self._model.forward(images)
                loss += self._criterion(output, labels).item()

                probabilities = torch.exp(output)
                # max() returns two tensors.
                # tensor[0] contains the highest probabilities
                # of each image.
                # tensor[1] contains the predicted class associated
                # with the highest probability
                #
                # labels.data contains the class that each image
                # actually is
                #
                # equality gives us 1's and 0's for images where
                # predicted class is correct
                equality = (labels.data == probabilities.max(dim=1)[1])
                # how many predictions were correct / number of predictions made
                # We also have to convert equality from ByteTensor to FloatTensor
                # because ByteTensor does not support mean()
                accuracy += equality.type(torch.FloatTensor).mean()

                # local testing - just do it once
                if DEVELOPMENT_MODE and self._device == "cpu":
                    break
            return loss, accuracy


    def train(self, num_epochs, learning_rate):

        print_every = 40
        steps = 0

        # Adam uses momentum for gradient descent
        optimizer = optim.Adam(self._model.classifier.parameters(),
                                     lr=learning_rate)

        self._logger.info(optimizer)

        # local testing - just do it once
        if DEVELOPMENT_MODE and self._device == "cpu":
            print_every = 1

        # put model in training mode
        # so dropout is enabled
        self._model.train()

        for current_epoch in range(num_epochs):
            self._logger.info("starting epoch %s/%s", current_epoch + 1, num_epochs)
            running_loss = 0
            for _, (inputs, labels) in enumerate(self._train_data_loader):
                steps += 1

                # move inputs / labels to cpu / cuda
                self._logger.debug("moving training images and labels to %s", self._device)
                inputs, labels = inputs.to(self._device), labels.to(self._device)

                # reset gradients
                optimizer.zero_grad()

                # Forward and backward passes
                outputs = self._model.forward(inputs)
                loss = self._criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:

                    validation_loss, validation_accuracy = self.validate_validate_data()

                    print("Epoch: {}/{}.. ".format(current_epoch + 1, num_epochs),
                          "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                          "Validation Loss: {:.3f}.. ".format(validation_loss/len(self._validate_data_loader)),
                          "Validation Accuracy: {:.3f}".format(validation_accuracy/len(self._validate_data_loader)))

                    running_loss = 0

                    # Make sure training is back on
                    # so droupout is back on
                    self._model.train()

                    # local testing - just do it once
                    if DEVELOPMENT_MODE and self._device == "cpu":
                        break

    def _get_class_to_index(self, dataloader):
        print(help(dataloader.dataset))
        print(dir(dataloader.dataset))


    def load_model(self, checkpoint_path):
        """
        checkpoint = {
            'class_to_idx': class_to_idx,
            'state_dict': self._model.state_dict(),
            'pretrained_arch': self._arch,
            'hidden_features': self._hidden_features,
            'out_features': self._out_features,
            }
        """
        self._logger.info("Loading model from %s", checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        self._logger.info("Arch: %s, Hidden Features: %s, Output Features: %s",
                          checkpoint.get('pretrained_arch'),
                          checkpoint.get('hidden_features'),
                          checkpoint.get('out_features'))
        self._create_model(pretrained_arch=checkpoint.get('pretrained_arch', 'densenet121'),
                           hidden_features=int(checkpoint.get('hidden_features', 500)),
                           out_features=int(checkpoint.get('out_features', 102)))

        self._model.load_state_dict(checkpoint['state_dict'])
        return self._model, checkpoint['class_to_idx']

    def save_model(self, save_dir, epochs):
        self._model.to('cpu')

        # grab the class_to_idx data from the training model
        class_to_idx = self._train_data_loader.dataset.class_to_idx
        checkpoint = {
            'class_to_idx': class_to_idx,
            'state_dict': self._model.state_dict(),
            'pretrained_arch': self._arch,
            'hidden_features': self._hidden_features,
            'out_features': self._out_features,
        }

        filename = "checkpoint_{}_ft{}_ep{}.pth".format(self._arch,
                                                        self._hidden_features,
                                                        epochs)

        filepath = os.path.join(save_dir, filename)
        self._logger.info("Saving checkpoint to %s", filepath)
        torch.save(checkpoint, filepath)


def main():

    arg_object = parse_args()

    if arg_object.verbose:
        log_setup.init(loglevel=logging.DEBUG)
    else:
        log_setup.init(loglevel=logging.INFO)

    trainer = Trainer()

    if arg_object.gpu_mode:
        trainer.set_device(device='cuda')

    trainer._create_data_loaders(data_dir=arg_object.data_directory)
    if arg_object.checkpoint:
        trainer.load_model(arg_object.checkpoint)

    else:
        # build from scratch
        trainer._create_model(pretrained_arch=arg_object.pretrained_arch,
                              hidden_features=int(arg_object.hidden_units),
                              out_features=102)
    trainer.validate_test_data()
    trainer.validate_validate_data()
    trainer.train(num_epochs=int(arg_object.epochs), learning_rate=float(arg_object.learning_rate))
    trainer.save_model(save_dir=arg_object.save_dir,
                       epochs=int(arg_object.epochs))

if __name__ == "__main__":
    main()
