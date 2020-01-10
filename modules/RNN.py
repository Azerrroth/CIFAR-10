import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import torchvision
import torchvision.transforms as transforms
import os
from torch.autograd import Variable

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

    def train(self, device):
        optimizer = optim.Adam(self.parameters(), lr=0.0001)

        lossss = []
        accccc = []

        path = 'RNN_module.tar'
        initepoch = 0

        if os.path.exists(path) is not True:
            loss = nn.CrossEntropyLoss()
            # optimizer = optim.SGD(self.parameters(),lr=0.01)

        else:
            checkpoint = torch.load(path)
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            initepoch = checkpoint['epoch']
            loss = checkpoint['loss']
        for epoch in range(initepoch, 100):
            timestart = time.time()

            running_loss = 0.0
            total = 0
            correct = 0
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                inputs = Variable(inputs)
                labels = Variable(labels)
                inputs = inputs.view(-1, 32, 32 * 3)
                outputs = self(inputs)

                l = loss(outputs, labels)
                l.backward()
                optimizer.step()

                # print statistics
                running_loss += l.item()
                # print("i ",i)
                if i % 500 == 0:  # print every 500 mini-batches
                    print('[%d, %5d] loss: %.4f' %
                          (epoch, i, running_loss / 500))
                    lossss.append(running_loss / 500)

                    running_loss = 0.0
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    print('Accuracy of the network on the %d tran images: %.3f %%' % (total,
                                                                                      100.0 * correct / total))
                    accccc.append(100.0 * correct / total)
                    total = 0
                    correct = 0
                    torch.save({'epoch': epoch,
                                'model_state_dict': self.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': loss
                                }, path)

            print('epoch %d cost %3f sec' % (epoch, time.time()-timestart))
        with open('./rnnAns.csv', 'w') as fw:
            for i in range(len(lossss)):
                fw.write(str(lossss[i])+','+str(accccc[i])+'\n')
        print('Finished Training')

    def test(self, device, classes):
        correct = 0
        total = 0
        class_correct = list(0. for i in range(len(classes)))
        class_total = list(0. for i in range(len(classes)))

        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                images = images.view(-1, 32, 32 * 3)
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(4):
                  label = labels[i]
                  class_correct[label] += c[i].item()
                  class_total[label] += 1

        print('Accuracy of the network on the 10000 test images: %.3f %%' % (
            100.0 * correct / total))
        for i in range(len(classes)):
            print('Accuracy of %5s : %2d %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))
