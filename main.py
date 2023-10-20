

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

def train(epoch, net, dataloader, optimizer, criterion, args, scheduler, trial_name=None):
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
    saving_root = os.path.join(args.saveroot, args.expname)
    if not os.path.isdir(saving_root):
        os.mkdir(saving_root)
    if True:
        if epoch % args.ckptfreq == 0: 
            print('Saving best..')
            state = {
            'net': net.state_dict(),
            'acc': 0,
            'epoch': epoch,
            'lr': scheduler.get_last_lr(),
            }
            ckpt_saving_root = os.path.join(saving_root, 'checkpoint')
            if not os.path.isdir(ckpt_saving_root):
                os.makedirs(ckpt_saving_root)
            if trial_name == None:
                file_name = 'ckpt_' + str(epoch) + '.pth'
            else:
                file_name = trial_name + str(epoch) + '.pth'
            torch.save(state, os.path.join(ckpt_saving_root, file_name))
    return net

def test(epoch, net, dataloader, criterion, best_acc, args, trial_name=None):
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

    saving_root = os.path.join(args.saveroot, args.expname)
    if not os.path.isdir(saving_root):
        os.mkdir(saving_root)
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving best..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        ckpt_saving_root = os.path.join(saving_root, 'checkpoint')
        if not os.path.isdir(ckpt_saving_root):
            os.makedirs(ckpt_saving_root)
        if trial_name == None:
            file_name = 'best_ckpt' + '.pth'
        else:
            file_name = trial_name + '_best_ckpt' + '.pth'
        torch.save(state, os.path.join(ckpt_saving_root, file_name))
        best_acc = acc
    return best_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset Pruning')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset: cifar10/cifar100/tiny')
    parser.add_argument('--model', default='resnet18', type=str, help='model')
    parser.add_argument('--pr', default=0.5, type=float, help='pruning ratio')
    parser.add_argument('--mode', default=0, type=int, help='0: train; 1: pruning mask; 2: retrain on the pruned dataset')
    parser.add_argument('--ckptfreq', default=1, type=int, help='saving interval')
    parser.add_argument('--evalfreq', default=1, type=int, help='eval interval')
    parser.add_argument('--clsindim', default=50, type=int, help='classification-layer input-dim')
    parser.add_argument('--expname', default='AOSP_EXP1', type=str, help='the name of this exp')
    parser.add_argument('--bs', default=256, type=int, help='batchsize')
    parser.add_argument('--selectionmanner', default='i_10', type=str, help='number with _ indicates the exact ones, 0to2 indicates 0 to 20, i_1 indicates uniform-selection with every 1 interval.')
    parser.add_argument('--weightpower', default=0.0, type=float, help='ind.pow(this) is the weight of the ind-th pool vector')
    parser.add_argument('--maxepoch', default=400, type=int, help='max epoch')
    parser.add_argument('--trial', default=0, type=int, help='specific trial index')
    parser.add_argument('--saveroot', default='/mnt/workspace/workgroup/tanhaoru.thr/AOSP', type=str, help='root for saving')
    parser.add_argument('--ckptroot', default='', type=str, help='for already existed Checkpoint path')
    parser.add_argument('--aosproot', default='', type=str, help='for already existed AOSP score files')
    parser.add_argument('--risk', default=0, type=int, help='0: risk on suppor tset, 1: rish on query set.')#random
    parser.add_argument('--random', default=0, type=int, help='wether random pruning')
    parser.add_argument('--step', default=60, type=int, help='milestones steps.')#random
    parser.add_argument('--gamma', default=0.2, type=float, help='gamma')
    parser.add_argument('--cosscheduler', default=0, type=int, help='cos scheduler')
    parser.add_argument('--trainaug', default=0, type=int, help='0: None, 1: AutoAug (Cifar10), 2: RandAug, 3: AugMix')
    parser.add_argument('--nest', default=1, type=int, help='0: without nestrove, 1: with nest')
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
    if (min(args.pr, 1-args.pr)) == 0:
        trial_number = 1
    else:
        trial_number = int(1.0 / (min(args.pr, 1-args.pr)))
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

    # Pre-train
    if True:
        if args.dataset == 'cifar10':
            optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005, nesterov=True)
        elif args.dataset == 'cifar100':
            optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005, nesterov=True)
        elif args.dataset == 'tiny':
            optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001, nesterov=True)
    if args.mode == 3: #TODO
        maxepoch = maxepoch / (1-args.pr)
        maxepoch = int(maxepoch)
    scheduler_meta = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=maxepoch, eta_min=0.0001)

    if args.mode == 0: 
        if args.trial >= trial_number or args.trial <0 :
            trial_numbers = list(range(trial_number))
        else:
            trial_numbers = [args.trial]
        for trial_index in trial_numbers:
            best_acc = 0
            print('Now, we are in the trial-' + str(trial_index))
            # prepare for the current trial
            scheduler = copy.deepcopy(scheduler_meta)
            model = copy.deepcopy(net)
            optimizer = optim.SGD(model.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=0.0002, nesterov=True)
            support_mask = trial_mask_support[trial_index]
            trial_train_set = copy.deepcopy(trainset)
            # get the train set of the current trial
            trial_train_set.data = [trial_train_set.data[ind, :] for ind in range(len(support_mask)) if support_mask[ind] == 1]
            trial_train_set.data = torch.tensor(trial_train_set.data).numpy()
            trial_train_set.targets = [trial_train_set.targets[ind] for ind in range(len(support_mask)) if support_mask[ind]==1]
            trial_train_loader = torch.utils.data.DataLoader(trial_train_set, batch_size=args.bs, shuffle=True, num_workers=2)
            # name it
            trial_name = 'trial_' + str(trial_index) + '_'
            # train it
            for epoch in range(0, maxepoch):
                print('\nEpoch: ' + str(epoch) + '/' + str(maxepoch))
                model = train(epoch, model, trial_train_loader, optimizer, criterion, args, scheduler, trial_name)
                if epoch % args.evalfreq == 0:
                    best_acc = test(epoch, model, testloader, criterion, best_acc, args, trial_name)
                    print(args.expname + ' ' + trial_name + ': iter = ' + str(epoch) + '/' + str(maxepoch) + ', ' + 'and the best acc is ' + str(best_acc))
                scheduler.step()
            best_acc = test(epoch, model, testloader, criterion, best_acc, args)
            print(args.expname + ': iter = ' + str(epoch) + '/' + str(maxepoch) + ', ' + 'and the best acc is ' + str(best_acc))
    elif args.mode == 1:
        saving_root = os.path.join(args.saveroot, args.expname)
        if not os.path.isdir(saving_root):
            os.mkdir(saving_root)
        AOSP_saving_root = os.path.join(saving_root, 'AOSP_score')
        if not os.path.isdir(AOSP_saving_root):
            os.makedirs(AOSP_saving_root)
        if args.ckptroot == '':
            ckpt_saving_root = os.path.join(saving_root, 'checkpoint')
            # check the ckpt_saving_root
            if not os.path.isdir(ckpt_saving_root):
                print('there is no checkpoint path!!!')
            # check the AOSP_saving_root
        else:
            ckpt_saving_root = args.ckptroot
        # get all saved ckpt file names
        ckpt_file_list = os.listdir(ckpt_saving_root)
        ckpt_file_list = [file_name  for file_name in ckpt_file_list if file_name.find('.pth')!=-1 and file_name.find('best')==-1 and file_name.find('trial')!=-1]
        selection_manner = args.selectionmanner
        trial_number_of_given_ckpts = 0
        for name in ckpt_file_list:
            if name.find('trial_')!=-1:
                temp_number = int(name.split('_')[1])
                if temp_number > trial_number_of_given_ckpts:
                    trial_number_of_given_ckpts = temp_number
        trial_number_of_given_ckpts = trial_number_of_given_ckpts + 1
        if trial_number_of_given_ckpts!=trial_number:
            print('trial_number_of_given_ckpts != trial_number: ', str(trial_number_of_given_ckpts) + ', ' + str(trial_number))
            trial_number = trial_number_of_given_ckpts

        if args.trial >= trial_number or args.trial <0 :
            trial_numbers = list(range(trial_number))
        else:
            trial_numbers = [args.trial]

        for trial_index in trial_numbers:
            print('process trial ' + str(trial_index))
            # define the trial name
            trial_name = 'trial_' + str(trial_index) + '_'
            # get the mask
            support_mask = trial_mask_support[trial_index]
            #AOSP
            query_mask = [1-i for i in support_mask]
            #MOSO-P
            query_mask = support_mask#[1-i for i in support_mask]
            # conrtruct the support set
            support_set = copy.deepcopy(trainset)
            support_set.data = [support_set.data[ind, :] for ind in range(len(support_mask)) if support_mask[ind] == 1]
            support_set.data = torch.tensor(support_set.data).numpy()
            support_set.targets = [support_set.targets[ind] for ind in range(len(support_mask)) if support_mask[ind]==1]
            support_loader = torch.utils.data.DataLoader(support_set, batch_size=1, shuffle=False, num_workers=1)
            # construct the query set
            query_set = copy.deepcopy(trainset)
            query_set.data = [query_set.data[ind, :] for ind in range(len(query_mask)) if query_mask[ind] == 1]
            query_set.data = torch.tensor(query_set.data).numpy()
            query_set.targets = [query_set.targets[ind] for ind in range(len(query_mask)) if query_mask[ind]==1]
            query_loader = torch.utils.data.DataLoader(query_set, batch_size=1, shuffle=False, num_workers=1)
            # get all ckpts of the current trial
            temp_file_list = []
            for file_name in ckpt_file_list:
                if file_name.find(trial_name)!=-1:
                    temp_file_list.append(file_name)
            # cal the AOSP scores for the query set in the current trial
            for ckpt_name in temp_file_list:
                ckpt_path = os.path.join(ckpt_saving_root, ckpt_name)
                ckpt = torch.load(ckpt_path, map_location='cpu')
                current_lr = ckpt['lr']
                current_epoch = ckpt['epoch']
                current_acc = ckpt['acc']
                current_net = ckpt['net']
                if selection_manner_and_epoch(selection_manner, current_epoch):
                    pass
                elif current_epoch == maxepoch:
                    pass
                else:
                    continue
                # check wether the AOSP-score.pth in current trial and epoch is well-calculated~~
                AOSP_file_path = os.path.join(AOSP_saving_root, trial_name + str(current_epoch) + '.pth')
                # TODO
                try:
                    temp_ckpt = torch.load(AOSP_file_path)
                    continue
                except:
                    print('Generating ckpt ' + AOSP_file_path)
                # load the ckpt's parameters
                model = copy.deepcopy(net)
                model.load_state_dict(current_net)
                classifier = nn.Linear(cls_indim, cls_outdim).cuda()
                classifier.load_state_dict(model.linear.state_dict())
                # get the overall gradient on the support set
                if args.risk == 0:
                    features, labels = extract_feature(model, support_loader)
                else:
                    features, labels = extract_feature(model, query_loader)
                sup_grad = cal_grad_set(classifier, [features, labels], device, criterion)
                sup_grad = sup_grad / len(features)
                sup_grad = sup_grad.detach().cpu()
                # cal the AOSP scores for the query set
                features, labels = extract_feature(model, query_loader)
                # cal_AOSP(classifier, sup_grad, lr, data, device, criterion)
                AOSP_query = cal_AOSP(classifier, sup_grad, current_lr, [features,labels], device, criterion)
                AOSP_all = []
                flag = -1
                for query_index in range(len(query_mask)):
                    if query_mask[query_index]==1:
                        flag = flag + 1
                        AOSP_all.append(AOSP_query[flag])
                    else:
                        AOSP_all.append(0)
                AOSP_all = torch.tensor(AOSP_all).cpu()
                # save them .......
                torch.save(AOSP_all, AOSP_file_path)
            # TODO: ImageNet
            #POOL_score_list = cal_pool(classifier, current_lr, [features,labels], device, criterion, current_epoch)
    else:
        # sort and return the top-m points
        #torch.save(pool_score_list, './pool_score/pool_' + str(epoch) + '.pth')
        saving_root = os.path.join(args.saveroot, args.expname)
        if not os.path.isdir(saving_root):
            os.mkdir(saving_root)
        if args.aosproot != '':
            AOSP_saving_root = args.aosproot
        else:
            AOSP_saving_root = os.path.join(saving_root, 'AOSP_score')
        if os.path.isdir(AOSP_saving_root):
            AOSP_file_list = os.listdir(AOSP_saving_root)
        else:
            print('Error: AOSP saving root not found!!!')
        AOSP_score = dict()
        selection_manner = args.selectionmanner

        if args.random == 0:
            for file_name in AOSP_file_list: 
                if file_name.find('pth') == -1 or file_name.find('trial') == -1:
                    continue
                _, trial_index, epoch_index =  file_name.split('_')
                trial_index = int(trial_index)
                epoch_index = int(epoch_index.split('.')[0])
                if selection_manner_and_epoch(selection_manner, epoch_index):
                    kkkkkkkkk=0
                else:
                    continue
                AOSP_file_path = os.path.join(AOSP_saving_root, file_name)
                #print('now, we are loading ' + AOSP_file_path)
                temp_score = torch.load(AOSP_file_path, map_location='cpu')
                if AOSP_score.get(epoch_index)!=None:
                    AOSP_score[epoch_index] = AOSP_score[epoch_index] + temp_score
                else:
                    AOSP_score[epoch_index] = temp_score
            #print(AOSP_score.keys())
            # args.weightpower
            #weight_list = [i**selection_power for i in weight_list]
            #weight_list = [i/sum(weight_list) for i in weight_list]
            # score select manner
            # AOSP_score = list(AOSP_score.values())
            AOSP_epoch_key = list(AOSP_score.keys())
            all_score = 0
            denominator = 0
            selection_power = args.weightpower 
            for index in list(AOSP_epoch_key):
                denominator = denominator + ((index + 1) ** selection_power)
            for index in list(AOSP_epoch_key):
                if True: #selection_manner_and_epoch(selection_manner, index):
                    #AOSP_score.append(AOSP_score[index])
                    all_score = all_score + AOSP_score[index]/AOSP_score[index].norm() * ((index + 1) ** selection_power/denominator)
                    #print(AOSP_score[index].max(), all_score.max()*100, ((index + 1) ** selection_power))
        if True:
            #print(AOSP_score)
            my_transform = transform_train
            print(args.dataset, 'hahahahhhhhhhhubhubuybuyb')
            if args.dataset=='cifar10':
                pruned_dataset = torchvision.datasets.CIFAR10(root='/mnt/workspace/cifar10/exp1/data', train=True, download=True, transform=my_transform)
            elif args.dataset=='cifar100':
                pruned_dataset = torchvision.datasets.CIFAR100(root='/mnt/workspace/cifar10/exp1/data', train=True, download=True, transform=my_transform)
            elif args.dataset=='tiny':
                print('preparing Tiny-ImageNet')
                train_set_path = os.path.join('/mnt/workspace/workgroup/tanhaoru.thr/dataset/tiny-imagenet-200', 'train')
                pruned_dataset = ImageFolder(root=train_set_path, transform=my_transform)
            #selected_index = m_guided_opt(all_score, 1-args.pr)
            #selected_index = nopt2(all_score, 1-args.pr, pruned_dataset.targets)
            if args.random==0:
                #selected_index = nopt(all_score, 1-args.pr, pruned_dataset.targets)
                #selected_index = nopt2(all_score, 1-args.pr, pruned_dataset.targets)
                selected_index = sample_opt(all_score, 1-args.pr, pruned_dataset.targets)
            else:
                length = len(pruned_dataset)#pruned_dataset.data.shape[0]
                #print(max(pruned_dataset.targets))
                all_score = [0 for i in range(length)]
                selected_index = random_opt(all_score, 1-args.pr) 
            #torch.utils.data.Subset
            pruned_dataset = torch.utils.data.Subset(pruned_dataset, selected_index)
            #pruned_dataset.data = [pruned_dataset.data[ind, :] for ind in range(len(selected_index)) if selected_index[ind] == 1]
            #pruned_dataset.data = torch.tensor(pruned_dataset.data).numpy()
            #pruned_dataset.targets = [pruned_dataset.targets[ind] for ind in range(len(selected_index)) if selected_index[ind]==1]
            pruned_loader = torch.utils.data.DataLoader(pruned_dataset, batch_size=args.bs, shuffle=True, num_workers=2)
            #print(len(pruned_dataset))
            #for i in range(max(pruned_dataset.targets)+1):
            #    print(i, (torch.tensor(pruned_dataset.targets)==i).sum())
        #scheduler = copy.deepcopy(scheduler_meta)
        model = copy.deepcopy(net)
        print(model)
        #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0002)
        if args.nest == 0:
            nesterov = False 
        else:
            nesterov = True
        if args.dataset == 'cifar10':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005, nesterov=nesterov)
        elif args.dataset == 'cifar100':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005, nesterov=nesterov)
        elif args.dataset == 'tiny':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001, nesterov=nesterov)
            #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        print('Overall epoch is ' + str(maxepoch))
        step = args.step
        gamma=args.gamma
        milestones = list(range(step, maxepoch, step))
        if args.cosscheduler==1:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=maxepoch, eta_min=0.000000001)
        else:
            #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=maxepoch, eta_min=0.0001)
        best_acc = 0
        for epoch in range(0, maxepoch):
            model = train(epoch, model, pruned_loader, optimizer, criterion, args, scheduler)
            if epoch % args.evalfreq == 0:
                best_acc = test(epoch, model, testloader, criterion, best_acc, args)
                print(args.expname + ': iter = ' + str(epoch) + '/' + str(maxepoch) + ', ' + 'and the best acc is ' + str(best_acc))
            scheduler.step()
        best_acc = test(epoch, model, testloader, criterion, best_acc, args)
        print(args.expname + ': iter = ' + str(epoch) + '/' + str(maxepoch) + ', ' + 'and the best acc is ' + str(best_acc))



