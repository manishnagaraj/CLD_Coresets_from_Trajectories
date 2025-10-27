from __future__ import print_function
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import fire
from typing import List
import Utils  
import pickle
from tqdm import tqdm

def main(
        data_path: str = './Data',
        dataset: str = 'CIFAR100',
        model_arch: str = 'resnet18',
        workers: int = 4,
        epochs: int = 164,
        start_epoch: int = 0,
        batch_size: int = 128,
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        disable_nesterov: bool = False,
        schedule: List[int] = [81, 121],
        gamma: float = 0.1,
        checkpoint_path: str = './Data/Checkpoint',
        logpath: str = './Logs',
        resume_path: str = '',
        manual_seed: int = 1234,
        evaluate_only: bool = False,
        score_path: str = 'Scores/CLD_CIFAR100_resnet18_seed1234_scores.pickle',
        topk_per_class: int = 100,
):
    args = locals()

    allowed_datasets = {'cifar100', 'cifar10', 'imagenet'}
    if dataset.lower() not in allowed_datasets:
        raise ValueError(f"Invalid dataset '{dataset}'. Allowed values are: 'CIFAR100', 'CIFAR10', 'ImageNet'.")
    
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)

    best_acc = 0
    if not os.path.isdir(checkpoint_path):
        Utils.mkdir_p(checkpoint_path)
    
    print(f"Loading coreset scores from: {score_path}")
    if not os.path.isfile(score_path):
        raise FileNotFoundError(f"Score file not found at {score_path}. Please run scoring script first.")
    
    with open(score_path, 'rb') as f:
        cld_scores = pickle.load(f)
    
    args['scores'] = cld_scores
    
    train_loader, test_loader = Utils.get_dataloaders_for_coresets(**args)
    print(f"Created coreset with {len(train_loader.dataset)} samples.")

    model = Utils.get_model(model_arch, dataset)
    model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True
    print(f'\tTotal params: {sum(p.numel() for p in model.parameters())/1000000.0:.2f}M')

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(),
                          lr=learning_rate,
                          momentum=momentum,
                          weight_decay=weight_decay,
                          nesterov=not(disable_nesterov))
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=schedule, gamma=gamma)
    
    title = f"{dataset}_{model_arch}_coreset"
    save_str = f"{title}_seed{manual_seed}_topk{topk_per_class}"

    if not os.path.isdir(logpath):
        Utils.mkdir_p(logpath)
    
    if resume_path:
        assert os.path.isfile(resume_path), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(resume_path)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Utils.Logger(os.path.join(logpath, save_str+'.txt'), title=title, resume=True)
    else:
        logger = Utils.Logger(os.path.join(logpath, save_str+'.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Test Loss', 'Train Acc.', 'Test Acc.'])

    if evaluate_only:
        print("Evaluation only mode. Running test epoch.")
        test_loss, test_acc = Utils.test_epoch(test_loader, model, criterion, 0)
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}")
        return

    for epoch_no in range(start_epoch, epochs):
        # Retrieve current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f'\nEpoch: [{epoch_no + 1} | {epochs}] LR: {current_lr}')        
        train_loss, train_acc = Utils.train_epoch(train_loader, model, criterion, optimizer, epoch_no)
        test_loss, test_acc = Utils.test_epoch(test_loader, model, criterion, epoch_no)
        scheduler.step()

        # append logger file
        logger.append([current_lr, train_loss, test_loss, train_acc, test_acc])

        # Save model checkpoint
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        if is_best:
            Utils.save_best_checkpoint({
                'epoch': epoch_no + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, manual_seed, save_str, checkpoint_path=checkpoint_path)

    logger.close()
    print(f'Finished Training. Best Test Accuracy: {best_acc:.2f}')


if __name__ == '__main__':
    fire.Fire(main)