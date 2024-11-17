import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, \
    auc
from sklearn.metrics import roc_curve
from tqdm import tqdm

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

# 训练函数
def train( net, device,trainloader,optimizer_name,optimizer,epoch_num):
    criterion = nn.CrossEntropyLoss()
    net.train()
    train_loss=[]  #用于保存每一轮的数据
    train_acc=[]

    for epoch in range(epoch_num):  # 训练10个epoch
        total_loss = 0
        total = 0
        correct = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  #梯度清零
            outputs = net(inputs)   #前向传播
            loss = criterion(outputs, labels) #计算损失
            loss.backward()   #反向传播
            optimizer.step()   #优化器更新参数

            total_loss += loss.item()
            _, predicted = outputs.max(1)  # 获取预测结果
            total += labels.size(0)  # 注意是labels.size,获取批次大小
            correct += predicted.eq(labels).sum().item()  # 准确的数量
            if i % 100 == 0:
                print(
                    f'Train Epoch: {epoch} [{i * len(data)}/{len(trainloader.dataset)} ({100. * i/ len(trainloader):.0f}%)]\tLoss: {loss.item():.6f}\tCorrect: {correct}/{total}')
        print(f'total:{total}')
        print(len(trainloader))
        train_loss.append(total_loss/len(trainloader))
        train_acc.append(correct/total)
    torch.save(net.state_dict(), f"./model./{optimizer_name}_model.pth")  # 保存模型参数
    return train_loss,train_acc, net

# 测试函数
def test_model(net, device,testloader):
    net.eval()
    correct = 0
    total=0
    all_preds = []
    all_labels = []
    all_probs=[]
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            total += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
    accuracy = correct / total
    print(f'Accuracy: {accuracy:.3f}')
    return accuracy, all_labels, all_preds,all_probs

# 评估函数
def evaluate(net, device, optimizer_name,testloader, y_true, y_pred):
    #混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d',cmap='Blues')
    plt.title(f"{optimizer_name}_Confusion Matrix")
    plt.savefig(f"./results/plots/{optimizer_name}_confusion_matrix.png")
    plt.show()
    #精确率
    precision = precision_score(y_true, y_pred, average='weighted')
    #召回率
    recall = recall_score(y_true, y_pred, average='weighted')
    #F1分数
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    return precision, recall, f1
# 绘图函数
def plot_loss_acc(train_losses,train_accs):
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    for name, loss_values in train_losses.items():
        plt.plot(loss_values, label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Curves for Different Algorithms')
    plt.legend()

    plt.subplot(1, 2, 2)
    for name, correct_values in train_accs.items():
        plt.plot(correct_values, label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curves for Different Algorithms')
    plt.legend()
    # 保存图表
    plt.savefig('./results/plots/loss_accuracy_curves.png')
    plt.show()
def plot_auc_roc(all_labels,all_probs,optimizer_name):
    # 计算AUC-ROC曲线
    y_test_binarized = label_binarize(all_labels, classes=np.arange(10))
    #print(y_test_binarized)
    n_classes = y_test_binarized.shape[1]
    #print(n_classes)
    y_score = np.array(all_probs)[:, :n_classes]
    #print(y_score)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # 计算每个类别的FPR、TPR和AUC
    for i in range(n_classes):
        # 计算第i个类别的FPR、TPR和AUC
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 绘制AUC-ROC曲线
    for i in range(n_classes):
        plt.figure()
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic for class {}'.format(i))
        plt.legend(loc="lower right")
        plt.show()
    plt.savefig(f'./results/roc_curves/{optimizer_name}roc_curve.png')


#主函数
def main():
    # Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.错误解决方法
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    num_epochs = 4
    net = LeNet().to(device)
    optimizers = {
        # 'SGD': optim.SGD(net.parameters(), lr=0.01, momentum=0.9),
        # 'RMSprop': optim.RMSprop(net.parameters(), lr=0.001),
        'Adam': optim.Adam(net.parameters(), lr=0.001),
        # 'AdamW': optim.AdamW(net.parameters(), lr=0.001),
        # 'Adagrad': optim.Adagrad(net.parameters(), lr=0.01),
        # 'Adadelta': optim.Adadelta(net.parameters(), lr=1.0)
    }
    # 保存图表路径
    if not os.path.exists('./results/plots'):
        os.makedirs('./results/plots')
    if not os.path.exists('./model'):
        os.makedirs('./model')
    if not os.path.exists('./results/roc_curves'):
        os.makedirs('./results/roc_curves')
    #得到各个算法的损失度、准确率、精确率、召回率、f1分数
    all_train_losses = {name: [] for name in optimizers.keys()}
    all_train_accs = {name: [] for name in optimizers.keys()}
    all_test_losses = {name: [] for name in optimizers.keys()}
    all_test_corrects = {name: [] for name in optimizers.keys()}
    precisions={name: 0.0 for name in optimizers.keys()}
    recalls={name: 0.0 for name in optimizers.keys()}
    f1s={name: 0.0 for name in optimizers.keys()}
    for optimizer_name, optimizer in optimizers.items():
        print(f'Training with {optimizer_name}')
        criterion = nn.CrossEntropyLoss()
        net = LeNet().to(device)
        train_losses = []  # 用于保存每一轮的数据
        train_accs = []
        test_losses = []
        test_accs = []
        best_acc = 0.0  # 初始化最佳准确率
        for epoch in range(num_epochs):  # 一个epoch跑完一遍测试集
            net.train()  # 设置网络为训练模式
            running_loss = 0.0
            correct = 0
            total = 0

            # 使用tqdm显示训练进度条
            train_bar = tqdm(trainloader, desc=f'Training Epoch {epoch + 1}/{num_epochs}')
            for inputs, labels in train_bar:  # 每一轮训练一个批次,每个批次大小为设置的8
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()  # 梯度清零
                outputs = net(inputs)  # 前向传播
                loss = criterion(outputs, labels)  # 计算损失
                loss.backward()  # 反向传播
                optimizer.step()  # 优化器更新参数

                running_loss += loss.item()
                _, predicted = outputs.max(1)  # 获取预测结果
                total += labels.size(0)  # 注意是labels.size
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

            save_path="./model"
            # 保存最好的模型
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(net.state_dict(), os.path.join(save_path, "test.pth"))

       # train_loss, train_correct ,net= train_and_test(device,trainloader, optimizer_name,optimizer, epochs)  # 训练网络
        all_train_losses[optimizer_name]= train_losses
        all_train_accs[optimizer_name]= train_accs

        #评估模型
        accuracy, all_labels,all_preds,all_probs = test_model(net,device,testloader)
        precision, recall, f1 = evaluate(net,device,optimizer_name,testloader,all_labels, all_preds)
        precisions[optimizer_name] = precision  #精确度
        recalls[optimizer_name] = recall  #召回率
        f1s[optimizer_name] = f1   #F1分数
        plot_auc_roc(all_labels,all_probs,optimizer_name)

    plot_loss_acc(all_train_losses,all_train_accs)



if __name__ == '__main__':
    main()
