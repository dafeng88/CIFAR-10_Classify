import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from net.lenet import LeNet
# 设置设备（使用GPU如果可用，否则使用CPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据预处理，转为Tensor并归一化
transform = transforms.Compose([
    transforms.ToTensor(),  # 转为Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化
])

# 加载CIFAR-10训练集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=8, shuffle=True, num_workers=2)

# 加载CIFAR-10测试集
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=8, shuffle=False, num_workers=2)

def train_and_test(device,trainloader,optimizer,num_epochs):
    criterion = nn.CrossEntropyLoss()
    net = LeNet().to(device)  # 初始化
    # 训练和验证
    best_acc = 0.0  # 初始化最佳准确率
    train_losses = []  # 用于保存训练损失
    test_losses = []  # 用于保存测试损失
    train_accuracies = []  # 用于保存训练准确率
    test_accuracies = []  # 用于保存测试准确率
    for epoch in range(num_epochs): #一个epoch跑完一遍测试集
        net.train()  # 设置网络为训练模式
        running_loss = 0.0
        correct = 0
        total = 0

        # 使用tqdm显示训练进度条
        train_bar = tqdm(trainloader, desc=f'Training Epoch {epoch + 1}/{num_epochs}')
        for inputs, labels in train_bar:    #每一轮训练一个批次,每个批次大小为设置的8
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # 梯度清零
            outputs = net(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 优化器更新参数

            running_loss += loss.item()
            _, predicted = outputs.max(1)  # 获取预测结果
            total += labels.size(0)    #注意是labels.size
            correct += predicted.eq(labels).sum().item()

            # 更新进度条信息
            train_bar.set_postfix(loss=running_loss / (len(train_bar) * trainloader.batch_size),
                                  acc=100. * correct / total)

        train_loss = running_loss / len(trainloader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        net.eval()  # 设置网络为评估模式
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():  # 禁用梯度计算
            # 使用tqdm显示验证进度条
            test_bar = tqdm(testloader, desc=f'Validating Epoch {epoch + 1}/{num_epochs}')
            for inputs, labels in test_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                test_bar.set_postfix(loss=test_loss / (len(test_bar) * testloader.batch_size),
                                     acc=100. * correct / total)

        test_loss = test_loss / len(testloader)
        test_acc = 100. * correct / total
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        # 定义保存模型的数据
        save_path = './model'
        # 保存最好的模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(net.state_dict(), "./model./test.path")
    return train_losses,train_accuracies,net


# 定义训练和验证的主函数
def main():
    # Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.错误解决方法
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # 实例化网络并移动到设备（GPU或CPU）
    net = LeNet().to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss() #交叉熵损失函数
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # 训练和验证
    num_epochs = 4  # 训练的轮数
    best_acc = 0.0  # 初始化最佳准确率
    train_losses = []  # 用于保存训练损失
    test_losses = []  # 用于保存测试损失
    train_accs = []  # 用于保存训练准确率
    test_accs = []  # 用于保存测试准确率

    # 定义保存模型的数据
    save_path = './model'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for epoch in range(num_epochs): #一个epoch跑完一遍测试集
        net.train()  # 设置网络为训练模式
        running_loss = 0.0
        correct = 0
        total = 0

        # 使用tqdm显示训练进度条
        train_bar = tqdm(trainloader, desc=f'Training Epoch {epoch + 1}/{num_epochs}')
        for inputs, labels in train_bar:    #每一轮训练一个批次,每个批次大小为设置的8
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # 梯度清零
            outputs = net(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 优化器更新参数

            running_loss += loss.item()
            _, predicted = outputs.max(1)  # 获取预测结果
            total += labels.size(0)    #注意是labels.size
            correct += predicted.eq(labels).sum().item()

            # 更新进度条信息
            train_bar.set_postfix(loss=running_loss / (len(train_bar) * trainloader.batch_size),
                                  acc=100. * correct / total)

        train_loss = running_loss / len(trainloader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        net.eval()  # 设置网络为评估模式
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():  # 禁用梯度计算
            # 使用tqdm显示验证进度条
            test_bar = tqdm(testloader, desc=f'Validating Epoch {epoch + 1}/{num_epochs}')
            for inputs, labels in test_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                test_bar.set_postfix(loss=test_loss / (len(test_bar) * testloader.batch_size),
                                     acc=100. * correct / total)

        test_loss = test_loss / len(testloader)
        test_acc = 100. * correct / total
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # 保存最好的模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(net.state_dict(), os.path.join(save_path, "test.pth"))

    # 绘制损失和准确率曲线并保存
    # train_loss, train_acc, net = train_and_test(device,trainloader,optimizer,num_epochs)
    # train_losses=train_loss
    # train_accuracies=train_acc
    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(num_epochs), train_losses, label='Training Loss')
    #plt.plot(range(num_epochs), test_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(num_epochs), train_accs, label='Training Accuracy')
    #plt.plot(range(num_epochs), test_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')

    # 保存图表
    if not os.path.exists('./results/plots'):
        os.makedirs('./results/plots')
    plt.savefig('./results/plots/test_loss_accuracy_curves.png')
    plt.show()


if __name__ == "__main__":
    main()