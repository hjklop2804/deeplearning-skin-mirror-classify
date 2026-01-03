import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import time

# ================= 核心配置区域 =================
# 显卡只有6G显存，BatchSize建议16或32，不要太大
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 5  # 先跑5轮试试水，看看代码能不能通
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据集路径 (对应你刚才生成的文件夹名，如果不叫这个名字请手动修改)
DATA_DIR = 'dataest_final'


# ===============================================

def train_model():
    # 1. 定义数据预处理 (Resize到224是ResNet的标准输入)
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),  # 简单的数据增强
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("正在加载数据，请稍候...")

    # 2. 读取数据集
    # 检查路径是否存在
    if not os.path.exists(DATA_DIR):
        print(f"错误：找不到文件夹 {DATA_DIR}，请检查名字是否写对！")
        return

    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
                      for x in ['train', 'val']}

    # Windows下 num_workers 建议设为 0，避免多线程报错
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                 shuffle=True, num_workers=0)
                   for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    print(f"数据加载完成！训练集: {dataset_sizes['train']} 张, 验证集: {dataset_sizes['val']} 张")
    print(f"检测到的类别: {class_names}")

    # 3. 构建模型 (ResNet-50)
    print("\n正在下载/加载 ResNet50 预训练模型...")
    # weights='IMAGENET1K_V1' 等同于原来的 pretrained=True
    model = models.resnet50(weights='IMAGENET1K_V1')

    # 修改最后一层 (fc层)，从1000分类改成7分类 (因为我们有7种病)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 7)

    # 搬到 GPU 上
    model = model.to(DEVICE)

    # 4. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    # 使用 SGD 优化器，动量设为 0.9
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    # 5. 开始训练循环
    print(f"\n使用设备: {DEVICE}")
    print(f"开始训练，共 {EPOCHS} 轮...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        # 每个epoch包括训练和验证两个阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 训练模式
            else:
                model.eval()  # 验证模式

            running_loss = 0.0
            running_corrects = 0

            # 遍历数据
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # 梯度清零
                optimizer.zero_grad()

                # 正向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 反向传播 (只在训练阶段)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计损失
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # 每50个batch打印一次进度
                if i % 50 == 0 and phase == 'train':
                    print(f"   [Batch {i}] Loss: {loss.item():.4f}")

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    time_elapsed = time.time() - start_time
    print(f'\n训练完成！总耗时: {time_elapsed // 60:.0f}分 {time_elapsed % 60:.0f}秒')


if __name__ == '__main__':
    train_model()