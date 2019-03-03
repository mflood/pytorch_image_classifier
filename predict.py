"""
    predict.py
    Matthew Flood
    
    Load saved deep-learning model and classify an image
"""
import torch
import json
from collections import OrderedDict
from torchvision import datasets, transforms, models
from torch import nn
import logging
import argparse
import log_setup
from PIL import Image

def parse_args(argv=None):
    """
        Parse command line args
    """
    parser = argparse.ArgumentParser(description="Image Classifier Predictor")

    # positional arguments
    parser.add_argument('image_path', 
                        action="store", 
                        help="Path to input image")

    parser.add_argument('checkpoint',
                        action="store",
                        help="Path to checkpoint")
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
                        help="Use GPU for inference")

    parser.add_argument('--top_k',
                        dest="top_k",
                        default=1,
                        type=int,
                        required=False,
                        help="Use GPU for inference")

    parser.add_argument('--category_names',
                        dest="category_names",
                        required=False,
                        help="JSON file containing mapping of category numbers to names")

    results = parser.parse_args(argv)
    return results


class Predictor(object):

    def __init__(self):
        self._logger = logging.getLogger()
        self._inference_mode = 'cpu'

    def set_inference_mode(self, new_mode):
        """
            mode should be cuda or cpu
        """
        assert new_mode in ('cpu', 'cuda')
        self._inference_mode = new_mode

    def load_checkpoint(self, checkpoint_path):
        """
            Checkpoint is state_dict over a model
        """

        self._logger.info("Loading checkpoint from %s", checkpoint_path)

        # open the file and load the checkpoint data
        checkpoint = torch.load(checkpoint_path)
        self._logger.debug("Loaded the following keys from checkpoint: %s", checkpoint.keys())

        self._logger.debug("Loading densenet121")
        model = models.densenet121(pretrained=True)

        # Freeze parameters
        self._logger.debug("Freezing pretrained model")
        for parameter in model.parameters():
            parameter.requires_grad = False


        self._logger.debug("Replacing pretrained classifier with custom image classifier")
        classifier = nn.Sequential(OrderedDict([
                              ('fullyconnected1', nn.Linear(1024, 500)),
                              ('relu', nn.ReLU()),
                              ('fullyconnected2', nn.Linear(500, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

        model.classifier = classifier

        self._logger.debug("Loading model state_dict")
        model.load_state_dict(checkpoint['state_dict'])

        class_to_idx = checkpoint['class_to_idx']
        return model, class_to_idx

    def predict(self, image_path, model, class_to_idx, top_k=5):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        self._logger.info("Predicting top %d classification for image '%s'", top_k, image_path)

        image = Image.open(image_path)
        processed_image = process_image(image)

        image_as_torch = torch.from_numpy(processed_image)

        # Move model and image to cpu/gpu
        self._logger.debug("Moving image and model to %s", self._inference_mode)

        try:
            image_as_torch.to(self._inference_mode)
            model.to(self._inference_mode)
        except RuntimeError as error:
            self._logger.error(error)
            raise

        # put model in evaulation mode
        model.eval()

        # perform forward pass
        with torch.no_grad():
            logits = model.forward(image_as_torch.unsqueeze_(0))

        # Output of the network are logits, 
        # need to take softmax for probabilities
        ps = torch.exp(logits)

        # topk finds the topk probabilities along with
        # the corresponding indexes
        probabilities, indexes = ps.topk(top_k)
        self._logger.debug("Probabilities: %s", indexes)
        self._logger.debug("Indexes: %s", indexes)

        # Convert from tensor to numpy
        probilities_as_floats = probabilities[0].numpy()
        numpy_indexes = indexes[0].numpy()

        # convert indexes to class
        #
        idx_to_class = dict((v, k) for k, v in class_to_idx.items())
        classes = [idx_to_class[x] for x in numpy_indexes]

        return probilities_as_floats, classes

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    pimage = preprocess(image)
    #imshow(processed_image)
    return pimage.numpy()


def load_category_mapping(filepath):
    with open(filepath, "r") as handle:
        contents = handle.read()
        mapping = json.loads(contents)
        return mapping


def main():

    arg_object = parse_args()

    if arg_object.verbose:
        log_setup.init(loglevel=logging.DEBUG)
    else:
        log_setup.init(loglevel=logging.INFO)

    predictor = Predictor()
    if arg_object.gpu_mode:
        predictor.set_inference_mode('cuda')

    model, class_to_idx = predictor.load_checkpoint(checkpoint_path=arg_object.checkpoint)

    probs, classes = predictor.predict(image_path=arg_object.image_path,
                                       model=model,
                                       class_to_idx=class_to_idx,
                                       top_k=arg_object.top_k)
    category_mapping = None
    if arg_object.category_names:
        category_mapping = load_category_mapping(arg_object.category_names)
        logging.getLogger().debug("Converting classes: %s", classes)
        classes = ["{} ({})".format(x, category_mapping[x]) for x in classes]
        logging.getLogger().debug("Converted classes: %s", classes)

    for x in range(len(probs)):
        print("Class: {}  Probability: {}".format(classes[x], probs[x]))

if __name__ == "__main__":
    main()
