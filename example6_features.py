import os
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms
from torch.autograd import Variable

import matplotlib
matplotlib.use('qt5agg')
from matplotlib import pyplot as plt


BATCH_SIZE = 1
NUM_WORKERS = 1
LR = 1e-3

data_folder = "./cats_and_dogs"

traindir = os.path.join(data_folder, 'train')
testdir = os.path.join(data_folder, 'test')


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_loader = data.DataLoader(
    datasets.ImageFolder(traindir,
                         transforms.Compose([
                             transforms.RandomSizedCrop(224),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             normalize,
                         ])),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS)


# Definition here: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
model = models.resnet50(pretrained=True)

# Replace the FC layer by an identity operation
model.fc = nn.Sequential()


def getFeatures(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = Variable(data)
        output = model(data)
        print(output.data.numpy())


if __name__ == '__main__':
    for epoch in range(1, 2):
        getFeatures(epoch)
