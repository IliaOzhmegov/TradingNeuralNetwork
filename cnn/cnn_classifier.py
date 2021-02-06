import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.manual_seed(2021)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CATEGORIES = {
    0: 'Price goes UP',
    1: 'Price goes DOWN',
    2: 'No changes in price',
}
# hyperparameters
batch_size = 32
num_epochs = 20
learning_rate = 0.001
momentum = 0.9

loader_params = {
    'batch_size': batch_size,
    'num_workers': 0  # increase this value to use multiprocess data loading
}

transform = transforms.Compose([transforms.Resize(255),
                                transforms.ToTensor()])

# load train and test data
root = './data'

train_set = datasets.ImageFolder(root=root + "/train",
                                 transform=transform)
test_set = datasets.ImageFolder(root=root + "/test",
                                 transform=transform)

train_loader = DataLoader(dataset=train_set, shuffle=True, **loader_params)
test_loader = DataLoader(dataset=test_set, shuffle=False, **loader_params)

loader_params = {
    'batch_size': batch_size,
    'num_workers': 0  # increase this value to use multiprocess data loading
}

# Build Neural Network
class StockCNN(nn.Module):

    def __init__(self):
        super(StockCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.20)

        self.fc1 = nn.Linear(64 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout(self.pool(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.dropout(self.pool(x))

        x = x.view(-1, x_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def name(self):
        return "StockCNN"

# Feature dimensions
def x_features(x):
    size = x.size()[1:]
    x_features = 1
    for dim in size:
        x_features *= dim
    return x_features


# Training the model
def training(model, data_loader, optimizer, criterion, device):
    model.train()

    running_loss = 0.0
    running_corrects = 0

    for batch_idx, (inputs, labels) in enumerate(data_loader):

        # zero the parameter gradients
        optimizer.zero_grad()

        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)

        # backward
        loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        if batch_idx % 10 == 0:
            print(f'Training Batch: {batch_idx:4} of {len(data_loader)}')

    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = running_corrects.double() / len(data_loader.dataset)

    print('-' * 10)
    print(f'Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')

    return epoch_loss, epoch_acc


# testing
def test(model, data_loader, criterion, device):
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    # do not compute gradients
    with torch.no_grad():

        for batch_idx, (inputs, labels) in enumerate(data_loader):

            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            if batch_idx % 10 == 0:
                print(f'Test Batch: {batch_idx:4} of {len(data_loader)}')

        epoch_loss = running_loss / len(data_loader.dataset)
        epoch_acc = running_corrects.double() / len(data_loader.dataset)

    print('-' * 10)
    print(f'Test Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')

    return epoch_loss, epoch_acc

# plot the results
def plot(train_history, test_history, metric, num_epochs):

    plt.title(f"Validation/Test {metric} vs. Number of Training Epochs")
    plt.xlabel(f"Training Epochs")
    plt.ylabel(f"Validation/Test {metric}")
    plt.plot(range(1, num_epochs + 1), train_history, label="Train")
    plt.plot(range(1, num_epochs + 1), test_history, label="Test")
    plt.ylim((0, 1.))
    plt.xticks(np.arange(1, num_epochs + 1, 1.0))
    plt.legend()
    plt.savefig(f"{metric}.png")
    plt.show()


# model initialisation
model = StockCNN().to(device)

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
criterion = nn.CrossEntropyLoss()

train_acc_history = []
test_acc_history = []

train_loss_history = []
test_loss_history = []

best_acc = 0.0
since = time.time()

for epoch in range(num_epochs):

    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    # train
    training_loss, training_acc = training(model, train_loader, optimizer,
                                           criterion, device)
    train_loss_history.append(training_loss)
    train_acc_history.append(training_acc)

    # test
    test_loss, test_acc = test(model, test_loader, criterion, device)
    test_loss_history.append(test_loss)
    test_acc_history.append(test_acc)

    # overall best model
    if test_acc > best_acc:
        best_acc = test_acc

# timing
time_elapsed = time.time() - since
print(
    f'Training complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s'
)
print(f'Best val Acc: {best_acc:4f}')

# plot loss and accuracy curves
train_acc_history = [h.cpu().numpy() for h in train_acc_history]
test_acc_history = [h.cpu().numpy() for h in test_acc_history]

plot(train_acc_history, test_acc_history, 'accuracy_NN3', num_epochs)
plot(train_loss_history, test_loss_history, 'loss_NN3', num_epochs)
