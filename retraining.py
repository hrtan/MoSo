

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

def sample_opt2(S, size, tar):
    # S: dict
    # size: the subset size
    print('sampling based selection')
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

def sample_opt(S, size, tar):
    tar = torch.tensor(tar)
    K = max(tar)
    size = int(len(S) * size )
    pool = torch.tensor(S)
    pool = pool.squeeze()
    elements = torch.tensor([i for i in range(len(pool))])
    index = []
    for i in range(K+1):
        prob = pool[tar == i]
        prob = (prob).exp()
        prob = prob/prob.sum()
        elem = elements[tar == i]
        temp_index = np.random.choice(elem, int(size/(K+1)), p=prob.numpy(), replace=False)
        index = index + temp_index.tolist()
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
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
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
    # Save checkpoint.
    acc = 100.*correct/total
    return acc

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset Pruning')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--wd', default=0.0001, type=float, help='weight decay for Tiny')
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset: cifar10/cifar100/tiny')
    parser.add_argument('--model', default='resnet50', type=str, help='model')
    parser.add_argument('--pr', default=0.5, type=float, help='pruning ratio')
    parser.add_argument('--evalfreq', default=1, type=int, help='eval interval')
    parser.add_argument('--bs', default=256, type=int, help='batchsize')
    parser.add_argument('--maxepoch', default=200, type=int, help='max epoch')
    parser.add_argument('--random', default=0, type=int, help='wether random pruning')
    parser.add_argument('--noise_ratio', default=0.0, type=float, help='noise_ratio')
    parser.add_argument('--trainaug', default=0, type=int, help='0: None, 1: AutoAug (Cifar10), 2: RandAug, 3: AugMix')
    parser.add_argument('--nest', default=1, type=int, help='0: without nestrove, 1: with nest')
    parser.add_argument('--path', default='./MoSo_CIFAR100', type=str, help='the path of this exp')
    parser.add_argument('--num_trails', default=8, type=int, help='number of trials')
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
    elif args.model == 'senet':
        net = SENet18(cls_outdim)
    elif args.model == 'mobilenetv2':
        net = MobileNetV2(cls_indim, cls_outdim)
    elif args.model == 'EfficientNetB0':
        net = EfficientNetB0(cls_outdim)
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
    
    # load the mask
    if args.noise_ratio != 0.0:
        # find the noise-label path
        noise_mask_path = os.path.join(args.path, 'noise_mask')
        print('loading noise label ' + noise_mask_path)
        assert os.path.exists(noise_mask_path), "the noise-label path not exists"
        noise_label_path = os.path.join(noise_mask_path, 'label.pth')
        temp = torch.load(noise_label_path)
        uneq = 0
        for i,j in zip(trainset.targets, temp):
            if i!=j:
                uneq = uneq + 1
        print(uneq)
        #print((temp!=trainset.targets).float().sum())
        trainset.targets = temp

    # Load the MoSo score
    score_file_path = os.path.join(args.path, 'score/moso_score.pth')
    moso_all = torch.load(score_file_path, map_location='cpu')
    #all_score = all_score + AOSP_score[index]/AOSP_score[index].norm() * ((index + 1) ** selection_power/denominator)
    pruned_dataset = trainset
    
    all_score = moso_all
    if args.random==0:
        #selected_index = nopt(all_score, 1-args.pr, pruned_dataset.targets)
        selected_index = nopt2(all_score, 1-args.pr, pruned_dataset.targets)
        #selected_index = sample_opt(all_score, 1-args.pr, pruned_dataset.targets)
    else:
        length = len(pruned_dataset)#pruned_dataset.data.shape[0]
        #print(max(pruned_dataset.targets))
        all_score = [0 for i in range(length)]
        selected_index = random_opt(all_score, 1-args.pr) 
    #torch.utils.data.Subset
    pruned_dataset = torch.utils.data.Subset(pruned_dataset, selected_index)
    pruned_loader = torch.utils.data.DataLoader(pruned_dataset, batch_size=args.bs, shuffle=True, num_workers=2)
    model = copy.deepcopy(net)
    model = model.to(device)
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0002)
    if args.nest == 0:
        nesterov = False
    else:
        nesterov = True
    if args.dataset == 'cifar10':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005, nesterov=True)
    elif args.dataset == 'cifar100':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005, nesterov=True)
    elif args.dataset == 'tiny':
        #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001, nesterov=True)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd, nesterov=True)
    maxepoch = args.maxepoch
    print('Overall epoch is ' + str(maxepoch))
    ckpt_path = os.path.join(args.path, args.model + '_' + str(args.pr))
    if os.path.exists(ckpt_path):
        pass
    else:
        os.makedirs(ckpt_path)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=maxepoch, eta_min=0.000000001)
    best_acc = 0
    for epoch in range(0, maxepoch):
        print('\nEpoch: ' + str(epoch) + '/' + str(maxepoch))
        model = train(model, pruned_loader, optimizer, criterion)
        if epoch % args.evalfreq == 0:
            acc = test(model, testloader, criterion)
            print('the test acc is ' + str(float(acc)))
            file_name = 'trail_0_' + str(epoch) + '.pth'
            saving_path = os.path.join(ckpt_path, file_name)
            save_model(model, acc, epoch, saving_path, scheduler.get_last_lr())
        scheduler.step()



