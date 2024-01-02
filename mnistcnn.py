import torch
from matplotlib import pyplot as plt
from torch import nn, optim
# from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator

 
# 超参数
batch_size = 256  # 批大小
learning_rate = 0.0001  # 学习率
epochs = 30  # 迭代次数
channels = 1  # 图像通道大小
 
# 数据集下载和预处理
transform = transforms.Compose([transforms.ToTensor(),  # 将图片转换成PyTorch中处理的对象Tensor,并且进行标准化0-1
                                transforms.Normalize([0.5], [0.5])])  # 归一化处理
path = './data/'  # 数据集下载后保存的目录
# 下载训练集和测试集
trainData = datasets.MNIST(path, train=True, transform=transform, download=True)
testData = datasets.MNIST(path, train=False, transform=transform)
# 处理成data loader
trainDataLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=batch_size, shuffle=True)  # 批量读取并打乱
testDataLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=batch_size)
 
 
# 开始构建cnn模型
class cnn(torch.nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        #卷积，激励函数RELU，池化
        #4层卷积+池化
        self.model = torch.nn.Sequential(
            # The size of the picture is 28*28*1
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),#输入图片通道，卷积核通道，卷积核大小，步长，边界填充
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),#池化 4->1
 
            # The size of the picture is 14*14
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),#池化 4->1
            #
            # The size of the picture is 7*7
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2, stride=2),不需要池化了
 
            # The size of the picture is 7*7
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2, stride=2),
            #卷积结束，特征图组7*7*256

            torch.nn.Flatten(),#默认将第0维保留下来，其余拍成一维
            torch.nn.Linear(in_features=7 * 7 * 256, out_features=512),#全连接NN，输入层7*7*256->隐层512
            torch.nn.ReLU(),#激励函数
            torch.nn.Dropout(0.2),  # 抑制过拟合 随机失活
            torch.nn.Linear(in_features=512, out_features=10),#全连接NN，隐层512->输出层10
            # torch.nn.Softmax(dim=1) # pytorch的交叉熵函数其实是softmax-log-NLL 所以这里的输出就不需要再softmax了
        )
    
    #传播路线
    def forward(self, input):
        output = self.model(input)
        return output
 
 
# 选择模型
model = cnn()
# GPU可用时转到cuda上执行
if torch.cuda.is_available():
    model = model.cuda()
    print("Cuda is available!")
    print("CNN Model has been loaded to GPU")#输出提醒
 
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 选用交叉熵函数作为损失函数 目标标签是one-hotted
optimizer = optim.Adam(model.parameters(), lr=learning_rate)#优化Adam
# optimizer = optim.Adam(model.parameters()) #
 
# 训练模型并存储训练时的指标
#当前迭代次数
epoch = 1
#记录当前迭代损失和正确率
history = {'Train Loss': [],
           'Test Loss': [],
           'Train Acc': [],
           'Test Acc': []}

#early_stopping 参数
hist_tst_acc = 0.0
epoch_c = 1
epoch_count = 0

#迭代
while(epoch_c):
#for epoch in range(1, epochs+1):

    #迭代进度可视化 初始化
    processBar = tqdm(trainDataLoader, unit='step')
    #开始训练
    model.train(True)
    train_loss, train_correct = 0, 0
    for step, (train_imgs, labels) in enumerate(processBar):
        # GPU可用
        if torch.cuda.is_available():  
            train_imgs = train_imgs.cuda()#将数据装载进GPU
            labels = labels.cuda()
            #print("Data has been loaded into GPU")#输出提醒

        #BP
        model.zero_grad()  # 梯度清零
        outputs = model(train_imgs)  # 输入训练集
        loss = criterion(outputs, labels)  # 计算损失函数
        predictions = torch.argmax(outputs, dim=1)  # 得到预测值
        correct = torch.sum(predictions == labels)
        accuracy = correct / labels.shape[0]  # 计算这一批次的正确率
        loss.backward()  # 反向传播
        optimizer.step()  # 更新优化器参数

        # 可视化训练进度条设置 训练数据
        processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f" %  
                                   (epoch, epochs, loss.item(), accuracy.item()))

 
        # 记录下训练的指标
        train_loss = train_loss + loss
        train_correct = train_correct + correct
 
        # 当所有训练数据都进行了一次训练后，在验证集进行验证
        if step == len(processBar) - 1:
            tst_correct, totalLoss = 0, 0
            model.train(False)  # 开始测试
            model.eval()  # 固定模型的参数并在测试阶段不计算梯度
            with torch.no_grad():
                for test_imgs, test_labels in testDataLoader:
                    
                    #数据装载进GPU
                    if torch.cuda.is_available():
                        test_imgs = test_imgs.cuda()
                        test_labels = test_labels.cuda()

                    #测试
                    tst_outputs = model(test_imgs)
                    tst_loss = criterion(tst_outputs, test_labels)#loss_fn
                    predictions = torch.argmax(tst_outputs, dim=1)
 
                    totalLoss += tst_loss
                    tst_correct += torch.sum(predictions == test_labels)
 
                train_accuracy = train_correct / len(trainDataLoader.dataset) # 训练集正确率
                train_loss = train_loss / len(trainDataLoader)  # 累加loss后除以步数即为平均loss值
 
                test_accuracy = tst_correct / len(testDataLoader.dataset)  # 累加正确数除以样本数即为验证集正确率
                test_loss = totalLoss / len(testDataLoader)  # 累加loss后除以步数即为平均loss值
 
                history['Train Loss'].append(train_loss.item())  # 记录loss和acc，输出折线图
                history['Train Acc'].append(train_accuracy.item())
                history['Test Loss'].append(test_loss.item())
                history['Test Acc'].append(test_accuracy.item())
 
                processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f" %
                                           (epoch, epochs, train_loss.item(), train_accuracy.item(), test_loss.item(),
                                            test_accuracy.item()))
    epoch += 1
    processBar.close()

    if (test_accuracy.item() > hist_tst_acc):
        hist_tst_acc = test_accuracy.item()
        epoch_count = 0
    else:
        epoch_count += 1
    
    
    if(epoch == epochs):epoch_c = 0
    if(epoch_count == 3):epoch_c = 0



# 对测试Loss进行可视化
plt.plot(history['Test Loss'], color='red', label='Test Loss')
plt.plot(history['Train Loss'], label='Train Loss')
plt.legend(loc='best')
plt.grid(True)
plt.xlabel('Epoch')
plt.xlim([0, epoch])
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.ylabel('Loss')
plt.title('Train and Test LOSS')
plt.legend(loc='upper right')
plt.savefig('LOSS')
plt.show()
 
# 对测试Acc进行可视化
plt.plot(history['Test Acc'], color='red', label='Test Acc')
plt.plot(history['Train Acc'], label='Train Acc')
plt.legend(loc='best')
plt.grid(True)
plt.xlabel('Epoch')
plt.xlim([0, epoch])
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.ylabel('Accuracy')
plt.title('Train and Test ACC')
plt.legend(loc='lower right')
plt.savefig('ACC')
plt.show()

torch.save(model, './model.pth')# 保存pth


