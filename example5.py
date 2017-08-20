import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms
from torch.autograd import Variable


BATCH_SIZE = 16
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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(20)
        self.conv3 = nn.Conv2d(20, 30, kernel_size=5)
        self.bn3 = nn.BatchNorm2d(30)
        self.conv4 = nn.Conv2d(30, 30, kernel_size=5, stride=2)
        self.bn4 = nn.BatchNorm2d(30)
        self.fc1 = nn.Linear(750, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = F.elu(F.max_pool2d(self.conv1(x), 2))
        x = F.elu(F.max_pool2d(self.bn2(self.conv2(x)), 2))
        x = F.elu(F.max_pool2d(self.bn3(self.conv3(x)), 2))
        x = F.elu(F.max_pool2d(self.bn4(self.conv4(x)), 2))

        x = x.view(BATCH_SIZE, 750)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


model = Net()
optimizer = optim.Adam(model.parameters(), lr=LR)


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


if __name__ == '__main__':
    for epoch in range(1, 2):
        train(epoch)
