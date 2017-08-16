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


BATCH_SIZE = 4
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
    num_workers=NUM_WORKERS,
    pin_memory=False)


model = models.resnet50(pretrained=True)
import pdb; pdb.set_trace()

# Don't train the normal layers
for param in model.parameters():
    param.requires_grad = False

# Create a new output layer
model.fc = nn.Linear(2048, 2) # New layers has requires_grad = True by default



#regular_input = Variable(torch.randn(1, 3, 227, 227))
#volatile_input = Variable(torch.randn(1, 3, 227, 227), volatile=True)


optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data, requires_grad=True), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        #loss = F.nll_loss(output, target)
        loss = F.cross_entropy(output, target)
        loss.backward()
        import pdb; pdb.set_trace()
        optimizer.step()
        #if batch_idx % args.log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.data[0]))


if __name__ == '__main__':
    for epoch in range(1, 2):
        train(epoch)

