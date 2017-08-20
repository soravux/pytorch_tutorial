import os
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import Image

from matplotlib import pyplot as plt


# The preprocessing steps *must* be the same as the neural network training, otherwise it won't work
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([transforms.Scale(224),
                                 transforms.ToTensor(),
                                 normalize])


# Definition here: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
model = models.resnet50(pretrained=True)

# Replace the FC layer by an identity operation
model.fc = nn.Sequential()


def getFeatures(image):
    model.eval()
    data = Variable(image)
    output = model(data)
    return output.data.numpy()


if __name__ == '__main__':
    im_path = "cats_and_dogs/test/dog/dog.9363.jpg"
    im = Image.open(im_path)
    im_preprocess = preprocess(im)
    im_preprocess.unsqueeze_(0)
    features = getFeatures(im_preprocess)

    plt.subplot(121); plt.imshow(im)
    plt.subplot(122); plt.imshow(features.reshape(64, 32)); plt.colorbar()
    plt.show()
