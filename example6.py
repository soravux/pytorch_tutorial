import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.models as models
from torchvision import datasets, transforms


BATCH_SIZE = 32
NUM_WORKERS = 1
LR = 1e-3

data_folder = "./cats_and_dogs"

traindir = os.path.join(data_folder, 'train')
testdir = os.path.join(data_folder, 'test')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

torch.manual_seed(31415926)
if 'cuda' in str(device):
    torch.cuda.manual_seed(31415926)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_loader = data.DataLoader(
    datasets.ImageFolder(traindir,
                         transforms.Compose([
                             transforms.RandomResizedCrop(224),
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
                             transforms.RandomResizedCrop(224),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             normalize,
                         ])),
    batch_size=50,
    shuffle=True,
    num_workers=NUM_WORKERS)


# Definition here: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
model = models.resnet50(pretrained=True)

# Don't train the normal layers
for param in model.parameters():
    param.requires_grad = False


# Create a new output layer
model.fc = nn.Linear(2048, 2) # New layers has requires_grad = True by default

model = model.to(device)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    print("Using device:", device)
    for epoch in range(1, 2):
        train(epoch)
    print("Running test...")
    test()
    # 95% accuracy for 1 epoch
