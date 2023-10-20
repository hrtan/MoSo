

'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models
import os
import argparse
import model
from tqdm import tqdm  
import numpy as np
from utils import progress_bar
from utils import *
from models import *
from torch.utils.data import DataLoader
from torch.autograd import grad
import random
import copy
torch.manual_seed(3407)

class ResNetT(nn.Module):
    def __init__(self, cfg="resnet50"):
        super().__init__()
        
        if cfg == "resnet18":
            self.base = torchvision.models.resnet18(num_classes=200)
        elif cfg == "resnet34":
            self.base = torchvision.models.resnet34(num_classes=200)
        elif cfg == "resnet50":
            self.base = torchvision.models.resnet50(num_classes=200)
        else:
            raise NotImplementedError()
        
        self.base.avgpool =  nn.AdaptiveAvgPool2d((1,1))
        #self.base.fc.apply(weight_init_kaiming)
    def forward(self, x):
        return self.base(x)

def selection_manner_and_epoch(selection_manner, epoch):
    if True:
        if selection_manner[0] == 'i': 
            # for example:  i_10
            interval = int(selection_manner[2:])
            index_list = list(range(0, 10000, interval))
        elif selection_manner.find('to')!=-1: 
            # a to b
            start, end = selection_manner.split('to')
            start, end = int(start), int(end)
            index_list = list(range(start, end))
        else:    
            # for example: 1_2_3                   
            index_list = selection_manner.split('_')
            index_list = [int(i) for i in index_list]
        #print(index_list)
        for i in index_list:
            if epoch == i:
                return True
        return False

def calc_grad_single(model, data, label, criterion):
    z, t = data.unsqueeze(0), label.unsqueeze(0)
    model.eval()
    y = model(z)
    loss = criterion(y, t)
    # Compute sum of gradients from model parameters to loss
    params = [ p for p in model.parameters() if p.requires_grad ]
    g = list(grad(loss, params, create_graph=False))
    #g = torch.tensor(g).detach().cpu().reshape(-1)
    #print(g)
    g = torch.nn.utils.parameters_to_vector(g)
    #g = g.detach().cpu()
    return g

def cal_grad_set(model, data, device, criterion):
    model.eval()
    features,labels = data
    overall_grad = 0
    for i in tqdm(range(len(features))):
        data, label = features[i].to(device), torch.tensor(labels[i]).to(device)
        grad = calc_grad_single(model, data, label, criterion)
        grad = grad.detach().cpu()
        overall_grad = overall_grad * i/(i+1) + grad / (i+1)
    return overall_grad

def cal_AOSP(classifier, sup_grad, lr, data, device, criterion):
    AOSP_score_list = []
    features,labels = data
    grad_list = []
    sup_grad = sup_grad.to(device)
    # cal gradient for each sample
    for i in tqdm(range(len(features))):
        data, label = features[i].to(device), torch.tensor(labels[i]).to(device)
        grad = calc_grad_single(classifier, data, label, criterion)
        AOSP_z = sup_grad * (grad - sup_grad)
        AOSP_z = AOSP_z.sum().detach().cpu()
        AOSP_score_list.append(AOSP_z)
        #grad_list.append(grad)
    AOSP_score_list = torch.tensor(AOSP_score_list).detach().cpu()
    return AOSP_score_list

def extract_feature(net, dataloader):
    feature_list = []
    label_list = []
    net.eval()
    for i, (inputs, labels) in tqdm(enumerate(dataloader)):
        # compute output
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs,_ = net(inputs,out_feature=True)
        feature_list.append(outputs.squeeze().cpu().data)
        label_list.append(labels.item())
    return [feature_list,label_list]


def m_guided_opt(S, size):
    # S: dict
    # size: the subset size
    size = int(len(S) * size )
    pool = torch.tensor(S)
    pool = pool.squeeze()
    _, index = pool.topk(size)
    mask = (pool*0).int()
    mask[index] = 1
    return mask

def nopt(S, size, tar):
    # S: dict
    # size: the subset size
    tar = torch.tensor(tar)
    K = tar.max()
    size = int(len(S) * size )
    pool = torch.tensor(S)
    pool = pool.squeeze()
    # normalize
    for i in range(K+1):
        pool[tar == i] = pool[tar == i] - pool[tar == i].mean()
        N = pool[tar == i].std()
        pool[tar == i] = pool[tar == i]/N
    _, index = pool.topk(size)
    #mask = (pool*0).int()
    #mask[index] = 1
    #return mask
    return index


def sample_opt(S, size, tar):
    # S: dict
    # size: the subset size
    tar = torch.tensor(tar)
    K = tar.max()
    size = int(len(S) * size )
    pool = torch.tensor(S)
    pool = pool.squeeze()
    # normalize
    for i in range(K+1):
        pool[tar == i] = pool[tar == i] - pool[tar == i].mean()
        N = pool[tar == i].std()
        pool[tar == i] = pool[tar == i]/N
    #_, index = pool.topk(size)
    #pool = pool.numpy()
    probabilities = (pool - min(pool))
    probabilities = probabilities / probabilities.sum()
    print(probabilities.sum())
    probabilities = probabilities.numpy().tolist()
    s_p = sum(probabilities)
    probabilities = [pp/s_p for pp in probabilities]
    elements = [i for i in range(len(pool))]
    if size == len(S):
        return elements
    index = np.random.choice(elements, size, p=probabilities, replace=False)
    #mask = (pool*0).int()
    #mask[index] = 1
    #return mask
    return index

def nopt2(S, size, tar):
    # S: dict
    # size: the subset size
    tar = torch.tensor(tar)
    K = tar.max()
    size = int(len(S) * size )
    pool = torch.tensor(S)
    pool = pool.squeeze()
    mask = (pool*0).int()
    # normalize
    index_all = []
    for i in range(K+1):
        temp_pool = pool * (tar == i).float()
        _, index = temp_pool.topk(int(size/(K+1)))
        #mask[index] = 1
        index_all = index_all + index.tolist()
    #return mask
    return index_all

def random_opt(S, size):
    # S: dict
    # size: the subset size
    print('random selecting....')
    size = int(len(S) * size )
    pool = torch.tensor(S)
    pool = pool.squeeze()
    pool = torch.rand(pool.shape)
    _, index = pool.topk(size)
    #mask = (pool*0).int()
    #mask[index] = 1
    #return mask
    return index

def save_model(model, acc, epoch, path, lr):
    print('saving......')
    # os.makedirs(ckpt_saving_root)
    # 'lr': scheduler.get_last_lr(),
    state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'lr': lr,
            }
    torch.save(state, path)

def train(net, dataloader, optimizer, criterion):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return net

def test(net, dataloader, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    acc = 100.*correct/total
    return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset Pruning')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset: cifar10/cifar100/tiny')
    parser.add_argument('--model', default='resnet50', type=str, help='model')
    parser.add_argument('--ckptfreq', default=1, type=int, help='saving interval')
    parser.add_argument('--path', default='./MoSo_CIFAR100', type=str, help='the path of this exp')
    parser.add_argument('--bs', default=256, type=int, help='batchsize')
    parser.add_argument('--num_trails', default=8, type=int, help='number of trials')
    parser.add_argument('--maxepoch', default=50, type=int, help='max epoch')
    parser.add_argument('--noise_ratio', default=0.0, type=float, help='noise_ratio')
    parser.add_argument('--trainaug', default=0, type=int, help='0: None, 1: AutoAug (Cifar10), 2: RandAug, 3: AugMix')\
    #ckptfreq , cls_indim, num_classes
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    maxepoch = args.maxepoch

    # Data
    print('==> Preparing data..' + args.dataset)
    if args.dataset == 'tiny':
        mean = [0.4802, 0.4481, 0.3975]
        std = [0.2302, 0.2265, 0.2262]
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(55),
            transforms.Resize(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        transform_train_RandAug = transforms.Compose([
            transforms.RandomResizedCrop(55),
            transforms.Resize(64),
            transforms.RandAugment(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        transform_train_AugMix = transforms.Compose([
            transforms.RandomResizedCrop(55),
            transforms.Resize(64),
            #transforms.AugMix(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(int(64/0.875)),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        if True:
            if args.trainaug == 0:
                transform_train = transform_train
            elif args.trainaug == 3:
                transform_train = transform_train_AugMix
            else:
                transform_train = transform_train_RandAug
    else:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        transform_train_AutoAug = transforms.Compose([
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        transform_train_RandAug = transforms.Compose([
            transforms.RandAugment(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        transform_train_AugMix = transforms.Compose([
            #transforms.AugMix(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        if True:
            if args.trainaug == 0:
                transform_train = transform_train
            elif args.trainaug == 1:
                transform_train = transform_train_AutoAug
            elif args.trainaug == 2:
                transform_train = transform_train_RandAug
            elif args.trainaug == 3:
                transform_train = transform_train_AugMix
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    # Dataset
    print('==> Building model..')
    if args.dataset == 'cifar10': #classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        cls_outdim = 10
        trainset = torchvision.datasets.CIFAR10(root='/mnt/workspace/cifar10/exp1/data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='/mnt/workspace/cifar10/exp1/data', train=False, download=True, transform=transform_test)
        wholedataset = torchvision.datasets.CIFAR10(root='/mnt/workspace/cifar10/exp1/data', train=True, download=True, transform=transform_test)
    elif args.dataset == 'cifar100':
        cls_outdim = 100
        trainset = torchvision.datasets.CIFAR100(root='/earth-nas/datasets/cifar-100-python', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='/earth-nas/datasets/cifar-100-python', train=False, download=True, transform=transform_test)
        wholedataset = torchvision.datasets.CIFAR100(root='/earth-nas/datasets/cifar-100-python', train=True, download=True, transform=transform_test)
    elif args.dataset == 'tiny':
        cls_outdim = 200
        train_set_path = os.path.join('/mnt/workspace/workgroup/tanhaoru.thr/dataset/tiny-imagenet-200', 'train')
        test_set_path = os.path.join('/mnt/workspace/workgroup/tanhaoru.thr/dataset/tiny-imagenet-200', 'val')
        trainset = ImageFolder(root=train_set_path, transform=transform_train)
        testset = ImageFolder(root=test_set_path, transform=transform_test)
        wholedataset = ImageFolder(root=train_set_path, transform=transform_train)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=1)
    wholeloader = torch.utils.data.DataLoader(
        wholedataset, batch_size=1, shuffle=False, num_workers=1)
    
    # Model
    # net = VGG('VGG19')
    cls_indim = -10
    if args.model == 'resnet18':
        net = ResNet18(cls_indim, cls_outdim)
        if args.dataset == 'tiny':
            net = ResNetT('resnet18') #torchvision.models.resnet18(num_classes=200)
    elif args.model == 'resnet50':
        net = ResNet50(cls_indim, cls_outdim)
        if args.dataset == 'tiny':
            net = ResNetT('resnet50') #torchvision.models.resnet50(num_classes=200)
    elif args.model == 'senet50':
        net = SENet18(cls_indim, cls_outdim)
    elif args.model == 'mobilenetv2':
        net = MobileNetV2(cls_indim, cls_outdim)
    else:
        print('no valid network specified')
    
    #if args.dataset == 'tiny':
    #    net.avgpool = nn.AdaptiveAvgPool2d(1)
    #    num_ftrs = net.fc.in_features
    #    net.fc = nn.Linear(num_ftrs, 200)

    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    # net = EfficientNetB0()
    # net = RegNetX_200MF()
    # net = SimpleDLA()
    # net = model.ResNet18()

    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    dataset = args.dataset
    trial_number = args.num_trails
    support_set_size = len(trainset)//trial_number
    trial_mask_support = []
    for i in range(trial_number):
        if i != trial_number-1:
            start = i * support_set_size
            end = (i + 1) * support_set_size
            temp_mask = [int(j>=start and j<end) for j in range(len(trainset))]
            trial_mask_support.append(temp_mask)
        else:
            start = i * support_set_size
            end = len(trainset)
            temp_mask = [int(j>=start and j<end) for j in range(len(trainset))]
            trial_mask_support.append(temp_mask)
    
    if args.noise_ratio != 0.0:
        N = len(trainset)
        # noise_num = int(N * args.noise_ratio)
        noise_mask = [torch.rand(1)<args.noise_ratio for i in range(N)]
        L = max(trainset.targets)
        label_set = [ind for ind in range(L)]
        flag = -1
        for i in noise_mask:
            flag = flag + 1
            if i:
                trainset.targets[flag] = trainset.targets[flag] * 0 + random.sample(label_set, 1)[0]
            else:
                pass
        # saving mask
        noise_mask_path = os.path.join(args.path, 'noise_mask')
        if os.path.exists(noise_mask_path):
            pass
        else:
            os.makedirs(noise_mask_path)
        noise_mask_saving_path = os.path.join(noise_mask_path, 'mask.pth')
        torch.save(noise_mask, noise_mask_saving_path)
        noise_label_saving_path = os.path.join(noise_mask_path, 'label.pth')
        torch.save(trainset.targets, noise_label_saving_path)
    # Optimizer
    #if args.dataset == 'cifar10':
    #    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005, nesterov=True)
    #elif args.dataset == 'cifar100':
    #    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005, nesterov=True)
    #elif args.dataset == 'tiny':
    #    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001, nesterov=True)
    
    # Preparing the working path
    ckpt_path = os.path.join(args.path, 'checkpoint')
    if os.path.exists(ckpt_path):
        pass
    else:
        os.makedirs(ckpt_path)
    # Surrogate training
    trial_numbers = list(range(trial_number))
    for trial_index in trial_numbers:
        best_acc = 0
        print('Now, we are in the trial-' + str(trial_index))
        # prepare for the current trial
        model = copy.deepcopy(net)
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=0.0002, nesterov=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=maxepoch, eta_min=0.0001)
        support_mask = trial_mask_support[trial_index]
        trial_train_set = copy.deepcopy(trainset)
        # get the train set of the current trial
        if args.dataset == 'tiny':
            trial_train_set.samples = [trial_train_set.samples[ind] for ind in range(len(support_mask)) if support_mask[ind] == 1]
        else:
            trial_train_set.data = [trial_train_set.data[ind, :] for ind in range(len(support_mask)) if support_mask[ind] == 1]
            trial_train_set.data = torch.tensor(trial_train_set.data).numpy()
        trial_train_set.targets = [trial_train_set.targets[ind] for ind in range(len(support_mask)) if support_mask[ind]==1]
        trial_train_loader = torch.utils.data.DataLoader(trial_train_set, batch_size=args.bs, shuffle=True, num_workers=2)
        # name it
        trial_name = 'trial_' + str(trial_index) + '_'
        # train it
        for epoch in range(0, maxepoch):
            print('\ntrial_number: ' + str(trial_index) + ', ' + 'Epoch: ' + str(epoch) + '/' + str(maxepoch))
            # train(net, dataloader, optimizer, criterion)
            model = train(model, trial_train_loader, optimizer, criterion)
            if epoch % args.ckptfreq == 0:
                # test(net, dataloader, criterion)
                acc = test(model, testloader, criterion)
                print('the test acc is ' + str(float(acc)))
                file_name = trial_name + str(epoch) + '.pth'
                saving_path = os.path.join(ckpt_path, file_name)
                save_model(model, acc, epoch, saving_path, scheduler.get_last_lr())
            scheduler.step()
    