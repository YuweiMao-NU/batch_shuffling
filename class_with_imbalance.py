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

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

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

    majority_label_indices = [[] for _ in range(5)]
    minority_label_indices = [[] for _ in range(5)]
    for i in range(27500):
        indice = init_indices[i]
        l = labelsets[indice]
        if l<5:
            minority_label_indices[l].append(indice)
        else:
            majority_label_indices[l-5].append(indice)

    majority_indices = []
    for label_start in range(5/classes_num):
        majority_indices_tmp = list(map(list, zip(*majority_label_indices[label_start * classes_num:label_start * classes_num + classes_num])))
        majority_indices_tmp = list(_flatten(majority_indices_tmp))
        majority_indices.extend(majority_indices_tmp)
    if label_start * classes_num + classes_num < len(majority_label_indices):
        majority_indices_tmp = list(
            map(list, zip(*majority_label_indices[label_start * classes_num + classes_num:])))
        majority_indices_tmp = list(_flatten(majority_indices_tmp))
        majority_indices.extend(majority_indices_tmp)

    minority_indices = []
    for label_start in range(5/classes_num):
        minority_indices_tmp = list(map(list, zip(*minority_label_indices[label_start * classes_num:label_start * classes_num + classes_num])))
        minority_indices_tmp = list(_flatten(minority_indices_tmp))
        minority_indices.extend(minority_indices_tmp)

    # print(label_start)
    if label_start * classes_num + classes_num < len(minority_label_indices):
        minority_indices_tmp = list(
            map(list, zip(*minority_label_indices[label_start * classes_num + classes_num:])))
        minority_indices_tmp = list(_flatten(minority_indices_tmp))
        minority_indices.extend(minority_indices_tmp)

    # print(len(majority_indices), len(minority_indices))
    indices = []
    majority_i = 0
    minority_i = 0
    minority_size = batch_size/10
    majority_size = batch_size-minority_size
    while majority_i<len(majority_indices) and minority_i<len(minority_indices):

        tmp = majority_indices[majority_i: majority_i+majority_size]
        tmp.extend(minority_indices[minority_i: minority_i + minority_size])
        # random.shuffle(tmp)
        indices.extend(tmp)

        majority_i += majority_size
        minority_i += minority_size

    # print(majority_i)
    # print(minority_i)
    tmp = majority_indices[majority_i:]
    tmp.extend(minority_indices[minority_i:])
    # random.shuffle(tmp)
    indices.extend(tmp)
    #
    # print(len(indices))
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
        plt.title('model name=imbalance_v1_5')
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
        plt.title('model name=imbalance_v1_5_test')
        plt.show()

