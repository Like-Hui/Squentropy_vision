import argparse
import os
import sys
import shutil
import time

import wandb
import itertools
import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import torchvision.models as models

from torchvision.datasets import SVHN
from torchvision.transforms import ToTensor

# from torchmetrics import HingeLoss
# used for logging to TensorBoard
parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--num_classes', default=10, type=int,
                    help='class number')
parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--optimizer', default='SGD', type=str,
                    help='optimizer')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='WideResNet-28-10', type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--loss_type', default='CE', type=str,
                    help='name of the loss function, choose from CE, MSE, and sqen')
parser.add_argument('--new_loss', action='store_true',
                            help='sc loss or not')
parser.add_argument('--weighted', default=0, type=int, help='reweight the loss at true label by ?')
parser.add_argument('--rescale_factor', default=1, type=int, help='rescale the one hot vector by how much?')
parser.add_argument('--rescale', default=1, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument("--save", type=str, help="model save path")

parser.set_defaults(augment=True)

best_prec1 = 0

seed = 1111
os.environ['PYTHONHASHSEED'] = str(seed)
# Torch RNG
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# Python RNG
np.random.seed(seed)
random.seed(seed)

class _ECELoss(nn.Module):

    def __init__(self, n_bins=10):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        ece = torch.zeros(1, device=logits.device)

        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                print('bin_lower=%f, bin_upper=%f, accuracy=%.4f, confidence=%.4f: ' % (bin_lower, bin_upper, accuracy_in_bin.item(),
                      avg_confidence_in_bin.item()))
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        print('ece = ', ece)
        return ece

def main():
    global args, best_prec1
    args = parser.parse_args()
    print(args)
    t = time.perf_counter()

    wandb.init(project='combined_svhn_vgg', entity='like0902', config = args)

    # Data loading code
    train_dataset = SVHN(
        root='data/', split='train', download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
    )

    val_dataset = SVHN(
        root='data/', split='test', download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
    )

    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=True, **kwargs)

    # create model
    model = models.vgg11_bn(pretrained=False, num_classes=args.num_classes)

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    model = model.cuda()

    # Here is one example to calculate the ece along different confidence interval
    # Load model state dict
    # model_filename = os.path.join(args.save, 'model_best.pth.tar')
    # if not os.path.exists(model_filename):
    #     raise RuntimeError('Cannot find file %s to load' % model_filename)
    # model_name = torch.load(model_filename)
    # model.load_state_dict(model_name['state_dict'])
    #
    # ece_criterion = _ECELoss().cuda()
    # logits_list = []
    # labels_list = []
    # with torch.no_grad():
    #     for i, (input, target) in enumerate(val_loader):
    #         target = target.cuda(non_blocking=True)
    #         input = input.cuda(non_blocking=True)
    #         logits = model(input)
    #         logits_list.append(logits)
    #         labels_list.append(target)
    #     logits = torch.cat(logits_list).cuda()
    #     labels = torch.cat(labels_list).cuda()
    # before_temperature_ece = ece_criterion(logits, labels).item()
    # sys.exit()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum, nesterov=args.nesterov,
                                    weight_decay=args.weight_decay)
    # cosine learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*args.epochs)

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, optimizer, epoch, args.num_classes, scheduler)

        # evaluate on validation set
        prec1 = validate(val_loader, model, epoch, args.num_classes)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)
    print('Best accuracy: ', best_prec1)
    ExecTime = time.perf_counter() - t
    print('Running time: ', ExecTime)

def train(train_loader, model, optimizer, epoch, num_classes, scheduler):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    grad_norm = []
    w_norm = []

    for i, (input, target) in enumerate(train_loader):
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)

        # compute output
        output = model(input)

        if args.loss_type == 'MSE':
            device = target.get_device()
            target_final = torch.zeros([target.size()[0], num_classes], device=device).scatter_(1, target.reshape(target.size()[0], 1), 1)

            if args.weighted != 0:
                mse_weights = target_final * args.weighted + 1
                loss = torch.mean((output - args.rescale_factor * target_final.type(torch.float)) ** 2 * mse_weights)
            else:
                loss = torch.mean((output - args.rescale_factor * target_final.type(torch.float)) ** 2)

        elif args.loss_type == 'sqen':
            device = target.get_device()
            target_final = torch.zeros([target.size()[0], num_classes], device=device).scatter_(1, target.reshape(
                target.size()[0], 1), 1)

            ce_func = nn.CrossEntropyLoss().cuda()
            loss = args.resquare * (torch.sum(output ** 2) - torch.sum((output[target_final == 1]) ** 2)) / (num_classes - 1) / \
                   target_final.size()[0] + ce_func(output, target)


        else:
            if args.new_loss:
                output = output ** 2
                output[:, target] *= args.rescale
            target_final = target
            criterion = nn.CrossEntropyLoss().cuda()
            loss = criterion(output, target_final)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        lin_norm = torch.norm(model.classifier[6].weight.grad, keepdim=True)
        weight_norm = torch.norm(model.classifier[6].weight, keepdim=True)

        grad_norm.append(lin_norm.detach().cpu().numpy())
        w_norm.append(weight_norm.detach().cpu().numpy())

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      loss=losses, top1=top1))
    # log to TensorBoard
    wandb.log({'train_loss': losses.avg})
    wandb.log({'train_acc': top1.avg})
    wandb.log({'grad_norm': np.mean(np.asarray(grad_norm))})
    wandb.log({'w_norm': np.mean(np.asarray(w_norm))})


def validate(val_loader, model, epoch, num_classes):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)

        # compute output
        with torch.no_grad():
            output = model(input)

        if args.loss_type == 'MSE':
            device = target.get_device()
            target_final = torch.zeros([target.size()[0], num_classes], device=device).scatter_(1, target.reshape(
                target.size()[0], 1), 1)

            if args.weighted != 0:
                mse_weights = target_final * args.weighted + 1
                loss = torch.mean((output - args.rescale_factor * target_final.type(torch.float)) ** 2 * mse_weights)
            else:
                loss = torch.mean((output - args.rescale_factor * target_final.type(torch.float)) ** 2)

        elif args.loss_type == 'sqen':
            device = target.get_device()
            target_final = torch.zeros([target.size()[0], num_classes], device=device).scatter_(1, target.reshape(
                target.size()[0], 1), 1)

            ce_func = nn.CrossEntropyLoss().cuda()
            loss = args.resquare * (torch.sum(output ** 2) - torch.sum((output[target_final == 1]) ** 2)) / (
                        num_classes - 1) / \
                   target_final.size()[0] + ce_func(output, target)

        else:
            if args.new_loss:
                output = output ** 2
                output[:, target] *= args.rescale
            target_final = target
            criterion = nn.CrossEntropyLoss().cuda()
            loss = criterion(output, target_final)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # log to TensorBoard
    wandb.log({'val_loss': losses.avg})
    wandb.log({'val_acc': top1.avg})
    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.name) + 'model_best.pth.tar')

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
