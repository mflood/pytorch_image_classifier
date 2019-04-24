# pytorch_image_classifier

> Image Classifier using Pytorch

# AWS
> Launching GPU instance

Community AMI
Deep Learning AMI with Conda (Amazon Linux) (ami-895adef3)
p2.xlarge  0.90/hour

# ssh onto box
  git clone https://github.com/mflood/pytorch_image_classifier
  cd pytorch_image_classifier/flowers
  ./1_download_images.sh && ./2_extract_images.sh

# install pytorch and downgrade to match udacity environment
  conda install pytorch torchvision -c pytorch
  pip install pip==9.0.1
  pip install torchvision==0.2.1
  pip install torch==0.4.0
  pip install numpy==1.12.1

# to read:

https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-pytorch.html
https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-across-devices

