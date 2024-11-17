
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载CIFAR-10数据集
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

# 定义网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net().to(device)

# 训练函数
def train(net, device, trainloader, optimizer, epoch):
    net.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = net(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(trainloader.dataset)} ({100. * batch_idx / len(trainloader):.0f}%)]\tLoss: {loss.item():.6f}')

# 测试函数
def test(net, device, testloader):
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(testloader.dataset)
    accuracy = 100. * correct / len(testloader.dataset)
    return test_loss, accuracy

# 画图函数
def plot_metrics(optimizer_name, train_losses, test_losses, test_accuracies, epoch):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title(f'Loss - {optimizer_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title(f'Accuracy - {optimizer_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# 评估函数
def evaluate(net, device, testloader, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.show()

    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

# 主函数
def main():
    epochs = 10
    optimizers = {
        'SGD': optim.SGD(net.parameters(), lr=0.01, momentum=0.9),
        'Adam': optim.Adam(net.parameters(), lr=0.001),
        'AdamW': optim.AdamW(net.parameters(), lr=0.001),
        'Momentum': optim.SGD(net.parameters(), lr=0.01, momentum=0.9),
        'RMSprop': optim.RMSprop(net.parameters(), lr=0.001),
        'Adagrad': optim.Adagrad(net.parameters(), lr=0.01),
        'Adadelta': optim.Adadelta(net.parameters(), lr=1.0)
    }

    for optimizer_name, optimizer in optimizers.items():
        print(f'Training with {optimizer_name}')
        train_losses = []
        test_losses = []
        test_accuracies = []
        y_true = []
        y_pred = []

        for epoch in range(epochs):
            train(net, device, trainloader, optimizer, epoch)
            test_loss, test_accuracy = test(net, device, testloader)
            train_losses.append(test_loss)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

            # 收集测试数据的预测结果
            net.eval()
            with torch.no_grad():
                for data, target in testloader:
                    data, target = data.to(device), target.to(device)
                    output = net(data)
                    _, predicted = torch.max(output.data, 1)
                    y_true.extend(target.cpu().numpy())
                    y_pred.extend(predicted.cpu().numpy())

        plot_metrics(optimizer_name, train_losses, test_losses, test_accuracies, epochs)
        evaluate(net, device, testloader, y_true, y_pred)

if __name__ == '__main__':
    main()
