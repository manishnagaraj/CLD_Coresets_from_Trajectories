from __future__ import print_function, absolute_import
import time
import os
import errno
import torch
import torch.nn as nn
import torch.nn.init as init
from tqdm import tqdm

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
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
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def _unpack_batch(batch):
    """Support (inputs, targets) and (inputs, targets, idx)."""
    if isinstance(batch, (list, tuple)) and len(batch) == 3:
        inputs, targets, _ = batch
    else:
        inputs, targets = batch
    return inputs, targets

def train_epoch(trainloader, model, criterion, optimizer, epoch):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    progress_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Epoch {epoch}")
    for batch_idx, batch in progress_bar:
        # measure data loading time
        data_time.update(time.time() - end)

        inputs, targets = _unpack_batch(batch)
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        progress_bar.set_postfix({
            'Loss': f"{losses.avg:.4f}",
            'Top1': f"{top1.avg:.4f}",
            'Top5': f"{top5.avg:.4f}",
            'Batch Time': f"{batch_time.avg:.4f}s"
        })

    return (losses.avg, top1.avg)

def test_epoch(testloader, model, criterion, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()

    progress_bar = tqdm(enumerate(testloader), total=len(testloader), desc=f"Epoch {epoch}")
    with torch.no_grad():
        for batch_idx, batch in progress_bar:
            # measure data loading time
            data_time.update(time.time() - end)

            inputs, targets = _unpack_batch(batch)
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            progress_bar.set_postfix({
                'Loss': f"{losses.avg:.4f}",
                'Top1': f"{top1.avg:.4f}",
                'Top5': f"{top5.avg:.4f}",
                'Batch Time': f"{batch_time.avg:.4f}s"
            })

    return (losses.avg, top1.avg)

def save_best_checkpoint(state, manual_seed, savestr ='', checkpoint_path='./Data/Checkpoint', filename=None):
    if filename is None:
        filename = 'checkpoint_' + savestr + f'_seed{manual_seed}.pth.tar'
    os.makedirs(checkpoint_path, exist_ok=True)
    best_filepath = os.path.join(checkpoint_path, filename)
    torch.save(state, best_filepath)
    print('Best model saved to ' + filename)
