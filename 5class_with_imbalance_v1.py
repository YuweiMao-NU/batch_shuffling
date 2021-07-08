# same class in each batch
import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.optim as optim

import random
import numpy as np

from collections import Counter
from Tkinter import _flatten

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),  # 3*32*32 -> 16*32*32
            torch.nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            torch.nn.Conv2d(16, 32, 3, padding=1),  # 16*32*32 -> 32*32*32
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)  # 32*32*32-> 32*16*16
        )
        self.conv3 = nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, padding=1),  # 32*16*16 -> 64*16*16
            torch.nn.ReLU(),
        )

        # self.conv4 = nn.Sequential(
        #     torch.nn.Conv2d(64, 128, 3, padding=1),  # 64*16*16 -> 128*16*16
        #     torch.nn.ReLU(),
        #     torch.nn.MaxPool2d(2, 2)  # 128*16*16 -> 128*8*8
        # )

        # self.conv5 = nn.Sequential(
        #     torch.nn.Conv2d(128, 256, 3, padding=1),  # 128*8*8 -> 256*8*8
        #     torch.nn.ReLU(),
        #     torch.nn.MaxPool2d(2, 2)  # 256*8*8 -> 256*4*4
        # )
        #
        # self.gap = nn.AvgPool2d(4, 4)
        # self.fc = torch.nn.Linear(256, 10)

        self.fc1 = torch.nn.Linear(64*16*16, 1024)
        self.fc2 = torch.nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.conv5(x)
        # x = self.gap(x)
        x = x.view(-1, 64*16*16)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def run(k, trainset, testloader, epochs, batch_size):
    seed = 34
    # print("seed: ", seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    model = Net()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_loss_list = []
    test_loss_list = []
    best_r = 0
    best_epoch = 0
    best_model = None
    for epoch in range(epochs):  # loop over the dataset multiple times

        shuffle_seed = k*N+epoch
        seeds.append(shuffle_seed)
        # print('shuffle_seed: ', shuffle_seed)

        np.random.seed(shuffle_seed)  # Numpy module.
        random.seed(shuffle_seed)  # Python random module.

        indices = list(i for i in range(27500))
        random.shuffle(indices)

        indices = CreateIndices(indices, trainset, batch_size)

        new_trainset = torch.utils.data.Subset(trainset, indices)

        trainloader = torch.utils.data.DataLoader(new_trainset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=0)

        running_loss = 0.0

        # train
        model.train()
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size()[0]
        train_loss = running_loss / 27500
        train_loss_list.append(train_loss)

        # test
        model.eval()
        correct = 0
        total = 0
        running_loss = 0.0
        with torch.no_grad():

            for i, data in enumerate(testloader, 0):
                # get the inputs
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * labels.size()[0]

                # get correct
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            test_loss = running_loss / 5500
        test_loss_list.append(test_loss)

        # print('epoch: {}, train loss: {}, test loss: {}'.format(epoch, train_loss, test_loss))
        # print('Accuracy of the network on the 10000 test images: %.5f %%' % (
        #         100 * correct / float(total)))

        result = 100 * correct / float(total)

        if result>best_r:
            best_r =result
            best_epoch = epoch
            best_model = model

    return best_r, best_epoch, train_loss_list, test_loss_list

def CreateIndices(init_indices, trainset, batch_size):
    datasets = []
    labelsets = []
    for data, label in trainset:
        datasets.append(data)
        labelsets.append(label)

    label_indices = [[] for _ in range(10)]
    for i in range(27500):
        indice = init_indices[i]
        label_indices[labelsets[indice]].append(indice)

    indices = []
    for label_start in range(int(10/classes_num)):
        indices_tmp = list(map(list, zip(*label_indices[label_start*classes_num:label_start*classes_num+classes_num])))
        indices_tmp = list(_flatten(indices_tmp))
        indices.extend(indices_tmp)

    i = 0
    final_indices = []
    while i<27500:
        tmp = indices[i:i+batch_size]
        random.shuffle(tmp)
        final_indices.extend(tmp)
        i += batch_size

    # print(len(final_indices))
    #
    # new_trainset = torch.utils.data.Subset(trainset, indices)
    #
    # trainloader = torch.utils.data.DataLoader(
    #     new_trainset, batch_size=batch_size)
    #
    # # batch_lists = [[] for _ in range(49)]
    # for step, (data, label) in enumerate(trainloader):
    #     print("Step: ", step)
    #     label = label.numpy()
    #     print(Counter(label))

    return indices

def construct_imbalance_data(trainset):
    print(len(trainset))
    minority = []
    for i in range(5):
        data = [x for x in trainset if (x[-1] == i)]
        minority.extend(data[:int(len(trainset)/100)])
    print(len(minority))

    majority = [x for x in trainset if (x[-1] > 4)]
    print(len(majority))

    trainset = minority + majority
    print(len(trainset))

    return trainset

if __name__ == '__main__':
    # load datasets
    batch_size_list = [512]
    classes_num = 5
    epochs = 100

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data',
                                            train=True,
                                            download=True,
                                            transform=transform)

    testset = torchvision.datasets.CIFAR10(root='./data',
                                           train=False, download=True,
                                           transform=transform)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    trainset = construct_imbalance_data(trainset)
    testset = construct_imbalance_data(testset)

    for batch_size in batch_size_list:

        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=batch_size, shuffle=False,
                                                 num_workers=2)

        N = 20  # running times
        seeds = []
        results = []

        train_loss_ret = []
        test_loss_ret = []

        for k in range(N):

            best_r, best_epoch, train_loss_list, test_loss_list = run(k, trainset, testloader, epochs, batch_size)
            print('running time: {}, best results: {}, best epoch: {}'.format(k, best_r, best_epoch))
            results.append(best_r)

            # ax1 = plt.subplot(range(epochs), loss_list, '-')

            ax1 = plt.subplot(2, 1, 1)
            ax2 = plt.subplot(2, 1, 2)
            plt.sca(ax1)
            plt.plot(range(epochs), train_loss_list, '-')

            plt.sca(ax2)
            plt.plot(range(epochs), test_loss_list, '-')

            train_loss_ret.append(train_loss_list)
            test_loss_ret.append(test_loss_list)

        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('model name=imbalance_v6')
        plt.show()

        print('batch_size: ', batch_size)
        # print("seeds: ", seeds)
        print("results: ", results)
        print("mean: ", np.mean(results))
        print("variance: ", np.var(results))
        print('std: ', np.std(results))

        print(train_loss_ret)
        print(test_loss_ret)
        test_loss_ret = np.array(test_loss_ret)
        test_min = test_loss_ret.min(axis=0)
        test_max = test_loss_ret.max(axis=0)

        plt.plot(range(epochs), test_min, '-')
        plt.plot(range(epochs), test_max, '-')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('model name=imbalance_v6_test')
        plt.show()

