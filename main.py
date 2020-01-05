import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from tensorboardX  import SummaryWriter
from tqdm import tqdm

from quantize import QConv2d, QLinear, CGPACTLayer, DoReFaQuantizeLayer
from PyTransformer.transformers.torchTransformer import TorchTransformer

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

model_names.append('mobilenet')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default='/home/jakc4103/windows/Toshiba/workspace/dataset/ILSVRC/Data/CLS-LOC/',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=110, type=int,
                    metavar='N', help='mini-batch size ')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=4e-5, type=float,
                    metavar='W', help='weight decay (default: 4e-5)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--logdir', default='/home/jakc4103/windows/Toshiba/workspace/imagenet/model/tensorboard', type=str,
                    help='path to save tensorboard logs')

parser.add_argument('--savedir', default='/home/jakc4103/windows/Toshiba/workspace/imagenet/model/vanilla', type=str,
                    help='path to save model weights')

best_prec1 = 0

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(  3,  32, 2), 
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

def set_module_bits(model, num_bits):
        for module_name in model._modules:			
            # has children
            if type(model._modules[module_name]) == QConv2d:
                model._modules[module_name].quant = DoReFaQuantizeLayer(num_bits=num_bits, quant_scale=False)

            elif type(model._modules[module_name]) == QLinear:
                model._modules[module_name].quant = DoReFaQuantizeLayer(num_bits=num_bits, quant_scale=True)

            elif len(model._modules[module_name]._modules) > 0:
                set_module_bits(model._modules[module_name], num_bits)

            else:
                if type(model._modules[module_name]) == CGPACTLayer:
                    model._modules[module_name].__init__(num_bits=num_bits)


def main():
    global args, best_prec1
    args = parser.parse_args()

    # create model
    model = torch.nn.DataParallel(Net())
    model.load_state_dict(torch.load("mobilenet_sgd_rmsprop_69.526.tar")['state_dict'])

    transformer = TorchTransformer()
    transformer.register(torch.nn.ReLU, CGPACTLayer)
    model = transformer.trans_layers(model, False)
    transformer.register(torch.nn.Conv2d, QConv2d)
    transformer.register(torch.nn.Linear, QLinear)
    model = transformer.trans_layers(model, True)
    
    set_module_bits(model, 4)

    model = model.cuda()
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    lr_schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 150, eta_min=0, last_epoch=-1)
   # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=10, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    writer = SummaryWriter(args.logdir)

    for epoch in range(args.start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, writer)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch, writer)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filename=os.path.join(args.savedir, 'checkpoint.pth.tar'))

        lr_schedular.step()


def train(train_loader, model, criterion, optimizer, epoch, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    tbar = tqdm(enumerate(train_loader), total=len(train_loader))
    end = time.time()
    for i, (input, target) in tbar:
        # measure data loading time
        data_time.update(time.time() - end)

        # target = target.cuda(async=True)
        target = target.cuda()
        input_var = input.cuda() #torch.autograd.Variable(input)
        target_var = target #torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tbar.set_description("loss: {}, top1: {}, top5: {}, epoch: {}".format(loss.data, top1.avg, top5.avg, epoch))
        writer.add_scalar("train/loss", loss.data, len(train_loader)*epoch + i + 1)
        writer.add_scalar("train/acc/top1", top1.avg, len(train_loader)*epoch + i + 1)
        writer.add_scalar("train/acc/top5", top5.avg, len(train_loader)*epoch + i + 1)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


def validate(val_loader, model, criterion, epoch, writer):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    tbar = tqdm(enumerate(val_loader), total=len(val_loader))
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in tbar:
            # target = target.cuda(async=True)
            target = target.cuda()
            input_var = input.cuda() #torch.autograd.Variable(input, volatile=True)
            target_var = target #torch.autograd.Variable(target, volatile=True)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            tbar.set_description("loss: {}, top1: {}, top5: {}, epoch: {}".format(loss.data, top1.avg, top5.avg, epoch))

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data, input.size(0))
            top1.update(prec1, input.size(0))
            top5.update(prec5, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        writer.add_scalar("val/acc/top1", top1.avg, epoch)
        writer.add_scalar("val/acc/top5", top5.avg, epoch)

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
