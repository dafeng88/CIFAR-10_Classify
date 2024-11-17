# 导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import os

# 加载和预处理数据
transform = transforms.Compose([  # 训练用
    transforms.RandomHorizontalFlip(),  # 以50%的概率对图像进行随机水平翻转
    transforms.RandomGrayscale(p=0.1),  # 以10%的概率将图像随机转换为灰度图像
    transforms.ToTensor(),  # 将PIL图像或NumPy数组转换为PyTorch张量。
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 对张量进行标准化
])

transform_test = transforms.Compose([  # 测试验证用
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 训练集加载
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=2)
# 测试集加载
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10000, shuffle=False, num_workers=2)


# 定义神经网络模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 定义损失函数和优化器
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    timestart = time.time()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()  # 梯度清零

        outputs = net(inputs)  # 前向传播

        loss = criterion(outputs, labels)  # 计算损失

        loss.backward()  # 反向传播

        optimizer.step()  # 更新参数

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if i % 1000 == 999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0
            print('Accuracy on the train images: %d %%' % (100 * correct / total))
            correct = 0
            total = 0
    print('epoch %d cost %3f sec' % (epoch, time.time() - timestart))
print('Finished Training')

# 评估模型
correct = 0
total = 0
with torch.no_grad():  # 在评估阶段不需要计算梯度
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)  # 前向传播
        _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy on the test images: %d %%' % (100 * correct / total))