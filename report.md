# 使用深度学习对 CIFAR-10 进行分类

### 实验目的

使用机器学习和深度学习方法对 CIFAR-10 数据集进行分析，针对不同的方法进行分析、测试。

### 实验说明

本次实验深度学习的方法使用了：全连接神经网络、循环神经网络和卷积神经网络分别构建模型从`CIFAR-10`数据集进行训练和测试。同时使用了机器学习的逻辑斯谛回归(Logistic Regression)，同样针对`CIFAR-10`数据集进行训练和测试。
`CIFAR-10`数据集中共包含 50000 张训练图片和 10000 张测试图片，共 10 个类别，分别是：_airplane_, _automobile_, _bird_, _cat_, _dog_, _frog_, _horse_, _ship_, _truck_。
图像大小为$32×32$的彩色 RGB 图片，如图：
![cifar](./pic/cifar.png)

### 实验步骤

1. 加载训练集
2. 使用`transforms`对数据进行预处理
3. 定义神经网络模型
4. 使用定义的网络模型进行训练
5. 对训练好的模型进行测试，测试模型的整体准确性和针对不同类型的准确性。

### 模型构建

#### 卷积神经网络

构建卷积神经网络模型，使用多层次的$卷积层+池化层$。
这里模仿了 VGG 的网络模型，使用了多个$3\times3$的卷积核。经检验，这样的模型的识别效果表现还不错。
| Input（32×32×3）color image|
|:----:|
|Conv1 3×3, 64|
|Conv2 3×3, 64|
|Maxpool 2×2, strides = 2|
|Batch Normalization|
|Relu|
|-|
|Conv3 3×3, 128|
|Conv4 3×3, 128|
|Maxpool 2×2, strides = 2|
|Batch Normalization|
|Relu|
|-|
|Conv5 3×3, 128|
|Conv6 3×3, 128|
|Conv7 1×1, 128|
|Maxpool 2×2, strides = 2|
|Batch Normalization|
|Relu|
|-|
|Conv8 3×3, 256|
|Conv9 3×3, 256|
|Conv10 1×1, 256|
|Maxpool 2×2, strides = 2|
|Batch Normalization|
|Relu|
|-|
|Conv11 3×3, 512|
|Conv12 3×3, 512|
|Conv13 1×1, 512|
|Maxpool 2×2, strides = 2|
|Batch Normalization|
|Relu|
|-|
|FC14 (8192, 1024)|
|Dropout|
|Relu|
|FC15 (1024, 1024)|
|Dropout|
|Relu|
|FC16 (1024, 10)|

```python
class Net(nn.Module):

    def __init__(self, trainloader, testloader):
        super(Net, self).__init__()
        self.trainloader = trainloader
        self.testloader = testloader

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv7 = nn.Conv2d(128, 128, 1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        self.conv8 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv9 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv10 = nn.Conv2d(256, 256, 1, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()

        self.conv11 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv13 = nn.Conv2d(512, 512, 1, padding=1)
        self.pool5 = nn.MaxPool2d(2, 2, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU()

        self.fc14 = nn.Linear(512*4*4, 1024)
        self.drop1 = nn.Dropout2d()
        self.fc15 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout2d()
        self.fc16 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.pool5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = x.view(-1, 512*4*4)
        x = F.relu(self.fc14(x))
        x = self.drop1(x)
        x = F.relu(self.fc15(x))
        x = self.drop2(x)
        x = self.fc16(x)

        return x
```

#### 全连接神经网络

全连接神经网络设计了 8 层全连接层，将原来的$32\times32\times3$的数据逐步降维，最终分为 10 类。
| Input（32×32×3）color image|
|:----:|
|$Linear$ $32\times32\times3$, $32\times32$|
|$Linear$ $32\times32$, $512$|
|$Linear$ $512$, $256$|
|$Linear$ $256$, $128$|
|$Linear$ $128$, $64$|
|$Linear$ $64$, $32$|
|$Linear$ $32$, $16$|
|$Linear$ $16$, $10$|

```python
class Net(nn.Module):
    def __init__(self, trainloader, testloader):
        super(Net, self).__init__()
        self.trainloader = trainloader
        self.testloader = testloader

        self.l1 = nn.Linear(32*32*3, 32*32)
        self.l2 = nn.Linear(32*32, 512)
        self.l3 = nn.Linear(512, 256)
        self.l4 = nn.Linear(256, 128)
        self.l5 = nn.Linear(128, 64)
        self.l6 = nn.Linear(64, 32)
        self.l7 = nn.Linear(32, 16)
        self.l8 = nn.Linear(16, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))
        x = F.relu(self.l6(x))
        x = F.relu(self.l7(x))
        x = self.l8(x)
        return x
```

#### 循环神经网络

循环神经网络使用长短期记忆(Long Short-Term Memory，LSTM)这样的时间循环神经网络，虽然其强项在于处理和预测时间序列中间隔和延迟非常长的重要事件，但在这里也可以用来对图像进行分类。
LSTM 模型定义如下：

```python
class Net(nn.Module):
    def __init__(self, trainloader, testloader):
        self.trainloader = trainloader
        self.testloader = testloader

        super(Net,self).__init__()
        self.LSTM = nn.LSTM(32 * 3, 128, num_layers=3, batch_first=True)
        self.line=nn.Linear(128,128)
        self.output = nn.Linear(128,10)
    def forward(self,x):
        out,(h_n,c_n) = self.LSTM(x)
        out=self.line(out[:,-1,:])
        return self.output(out)
```

#### Logistic Regression

在 Pytorch 中，`nn.Linear`使用的就是 Logistic Regression，所以这里只需要设计一层由 $32\times32\times3$到$10$的线性回归即可。
| Input（32×32×3）color image|
|:----:|
|$Linear \ 32\times32\times3$, $10$|

```python
class Logistic_Regression(nn.Module):
    def __init__(self, trainloader, testloader):
        super(Logistic_Regression, self).__init__()
        self.trainloader = trainloader
        self.testloader = testloader

        self.logistic = nn.Linear(32*32*3, 10)

    def forward(self, x):
        x = self.logistic(x)
        return x
```

### 运行结果与分析

#### 卷积神经网络

使用卷积神经网络作为模型进行训练，经过 100 轮过后，得到训练集中准确率达到 100%。
可以看到在下图中，使用 CNN 训练过程中函数梯度下降比较快，在 5-10 个 Epoch 内就可以收敛到一个比较小的值，在训练集的准确率已经超过 90%。
![CNN](./pic/CNN.png)

使用卷积神经网络根据 CIFAR-10 数据进行分类，经过 100 轮的训练，在 10000 个图片的测试集上，最终准确率为：**84.120%**

```
Accuracy of the network on the 10000 test images: 84.120 %
Accuracy of plane : 89 %
Accuracy of   car : 88 %
Accuracy of  bird : 78 %
Accuracy of   cat : 72 %
Accuracy of  deer : 87 %
Accuracy of   dog : 74 %
Accuracy of  frog : 89 %
Accuracy of horse : 85 %
Accuracy of  ship : 89 %
Accuracy of truck : 88 %
```

#### 全连接神经网络

使用全连接神经网络作为模型进行训练，经过 100 轮的训练过后，在训练集的准确率也可以达到 95%左右。
由图中可以看到，相比卷积神经网络，其损失函数的梯度下降速度较慢，但是相比 LSTM 以及机器学习方法的 Logistic Regression，这样的全连接网络的整体表现也可以说是差强人意。
![FCNN](./pic/FCNN.png)
使用全连接神经网络训练模型对 CIFAR-10 的数据进行分分类，最终得到在 10000 个测试集上的准确率为 **56.32%**

```
Accuracy of the network on the 10000 test images: 56.320 %
Accuracy of plane : 71 %
Accuracy of   car : 77 %
Accuracy of  bird : 30 %
Accuracy of   cat : 50 %
Accuracy of  deer : 44 %
Accuracy of   dog : 45 %
Accuracy of  frog : 60 %
Accuracy of horse : 64 %
Accuracy of  ship : 64 %
Accuracy of truck : 55 %
```

#### 循环神经网络

使用循环神经网络进行训练，可以看到，在 10 轮过后，loss 下降比较缓慢，但是仍处在一个持续下降的趋势。这里猜测可能需要增加循环神经网络的训练批数，可以获得更好的表现。
在训练集上的准确率最终也只达到了 80%左右。
![RNN](./pic/RNN.png)
使用循环神经网络训练模型对 CIFAR-10 的数据进行分类，最终得到在 10000 个测试集上的准确率为 **55.14%**

```
Accuracy of the network on the 10000 test images: 55.140 %
Accuracy of plane : 66 %
Accuracy of   car : 58 %
Accuracy of  bird : 39 %
Accuracy of   cat : 32 %
Accuracy of  deer : 51 %
Accuracy of   dog : 51 %
Accuracy of  frog : 71 %
Accuracy of horse : 61 %
Accuracy of  ship : 68 %
Accuracy of truck : 63 %
```

#### Logistic Regression

同时使用机器学习方法与上述的深度学习方法进行对比，机器学习在经过 10 轮左右的训练后就保持在稳定状态，loss 稳定在 0.0035 左右的同时，训练集的准确率也在 40%左右。
![logistic](./pic/logistic.png)
使用 Logistic Regression 这种二分类模型对 CIFAR-10 进行分类，最终可以看到在 10000 个测试集上的准确率为 **41.68%**

```
Accuracy of the network on the 10000 test images: 41.680 %
Accuracy of plane : 46 %
Accuracy of   car : 45 %
Accuracy of  bird : 25 %
Accuracy of   cat : 24 %
Accuracy of  deer : 32 %
Accuracy of   dog : 31 %
Accuracy of  frog : 60 %
Accuracy of horse : 41 %
Accuracy of  ship : 62 %
Accuracy of truck : 48 %
```

### 结论

深度学习在对于特征分类方面的表现明显好于机器学习方法，同时在深度学习中，卷积神经网络更适合处理图像的问题，所以在结果的准确率和训练速度上都要略胜一筹。
从下表中可以看出，对于其他模型分类识别准确率较低，难度较高的类型，使用卷积神经网络仍然可以得到一个比较好的结果。
| Type | CNN | FCNN | RNN | Logistics Regression |
|:-: | :-: | :-: | :-: | :-: |
|plane| 89% | 71% | 66% | 46% |
| car | 88% | 77% | 58% | 45% |
| bird| 78% | 30% | 39% | 25% |
| cat | 72% | 50% | 32% | 24% |
| deer| 87% | 44% | 51% | 32% |
| dog | 74% | 45% | 51% | 31% |
| frog| 89% | 60% | 71% | 60% |
|horse| 85% | 64% | 61% | 41% |
| ship| 89% | 64% | 68% | 62% |
|truck| 88% | 55% | 63% | 48% |
|**total**| **84.12%**| **56.32%** | **55.14%** | **41.68%** |

### Reference

[1] [training a classifier — pytorch tutorials 1.3.1 documentation](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py)
[2] [快速入门pytorch —— 训练一个图片分类器和多 gpus 训练](https://juejin.im/post/5cea7c9351882501c773089d)
[3] [用 pytorch 从零创建 cifar-10 的图像分类器神经网络](https://juejin.im/entry/5bf51d35e51d454049668d57)
[4] [logistic regression](https://houxianxu.github.io/implementation/LogisticRegression.html)
