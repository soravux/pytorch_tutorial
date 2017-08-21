import os
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms
from torch.autograd import Variable

from matplotlib import pyplot as plt


BATCH_SIZE = 32
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
test_loader = data.DataLoader(
    datasets.ImageFolder(testdir,
                         transforms.Compose([
                             transforms.RandomSizedCrop(224),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             normalize,
                         ])),
    batch_size=50,
    shuffle=True,
    num_workers=NUM_WORKERS)


# Definition here: https://github.com/pytorch/vision/blob/master/torchvision/models/squeezenet.py
model = models.squeezenet1_1(pretrained=True)

# Don't train the normal layers
for param in model.parameters():
    param.requires_grad = False


# Create a new output layer
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(512, 2, kernel_size=1)
        self.avgpool = nn.AvgPool2d(13)

    def forward(self, x):
        x = F.dropout(x, training=self.training)
        x = self.conv(x)
        x = self.avgpool(x)
        x = F.log_softmax(x)
        x = x.squeeze(dim=3).squeeze(dim=2)
        return x

model.classifier = Net()
model.num_classes = 2


optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.data[0]))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    for epoch in range(1, 2):
        train(epoch)
    print("Running test...")
    test()
    # 1 epoch gives 93% in 13 minutes
