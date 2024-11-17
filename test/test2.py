
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, ConfusionMatrix, Precision, Recall, F1Score, AUROC
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义模型
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

model = Net().to(device)

# 定义优化器和损失函数
optimizers = {
    'SGD': optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
    # 'Adam': optim.Adam(model.parameters(), lr=0.001),
    # 'AdamW': optim.AdamW(model.parameters(), lr=0.001),
    # 'Momentum': optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
    # 'RMSprop': optim.RMSprop(model.parameters(), lr=0.01),
    # 'Adagrad': optim.Adagrad(model.parameters(), lr=0.01),
    # 'Adadelta': optim.Adadelta(model.parameters(), lr=0.01)
}

loss_fn = nn.CrossEntropyLoss()

# 训练和评估模型
def train_and_evaluate(optimizer_name):
    model.train()
    optimizer = optimizers[optimizer_name]
    accuracy = Accuracy(task="classification")
    confusion_matrix = ConfusionMatrix(num_classes=10)
    precision = Precision(num_classes=10, average="macro", threshold=0.5)
    recall = Recall(num_classes=10, average="macro", threshold=0.5)
    f1_score = F1Score(num_classes=10, average="macro", threshold=0.5)
    auroc = AUROC(num_classes=10, compute_on_step=False)  # AUROC需要在测试时计算

    for epoch in range(10):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            accuracy.update(outputs, labels)
            confusion_matrix.update(outputs, labels)
            precision.update(outputs, labels)
            recall.update(outputs, labels)
            f1_score.update(outputs, labels)

    model.eval()
    with torch.no_grad():
        true_labels = []
        pred_labels = []
        pred_probs = []
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())
            pred_probs.extend(torch.nn.functional.softmax(outputs, dim=1).cpu().numpy())

    # 计算AUC-ROC
    fpr, tpr, _ = roc_curve(true_labels, np.max(pred_probs, axis=1))
    roc_auc = auc(fpr, tpr)

    # 可视化
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 4, 1)
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    plt.subplot(2, 4, 2)
    plt.bar(range(10), precision.compute())
    plt.xticks(range(10), range(10))
    plt.title('Precision')

    plt.subplot(2, 4, 3)
    plt.bar(range(10), recall.compute())
    plt.xticks(range(10), range(10))
    plt.title('Recall')

    plt.subplot(2, 4, 4)
    plt.bar(range(10), f1_score.compute())
    plt.xticks(range(10), range(10))
    plt.title('F1 Score')

    plt.subplot(2, 4, 5)
    plt.bar(range(10), accuracy.compute())
    plt.xticks(range(10), range(10))
    plt.title('Accuracy')

    plt.subplot(2, 4, 6)
    plt.imshow(confusion_matrix.compute().numpy(), cmap='hot', interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.show()

    # 重置指标
    accuracy.reset()
    precision.reset()
    recall.reset()
    f1_score.reset()
    confusion_matrix.reset()

# 运行所有优化器
for optimizer_name in optimizers.keys():
    train_and_evaluate(optimizer_name)

