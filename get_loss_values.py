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

def get_train_validation_losses(model, train_loader, validation_loader, epoch):
    criterion_evaluating = nn.CrossEntropyLoss(reduction='none')
    model.eval()
    train_losses, train_labels = [], []
    validation_losses, validation_labels = [], []

    with torch.no_grad():
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}, Train Loss Collection")
        for batch_idx, (train_ip, train_label, _idx) in progress_bar:
            train_ip, train_label = train_ip.cuda(), train_label.cuda()
            outputs = model(train_ip)
            loss = criterion_evaluating(outputs, train_label)
            train_losses.extend(loss.detach().cpu().numpy())
            train_labels.extend(train_label.detach().cpu().numpy())

    with torch.no_grad():
        progress_bar = tqdm(enumerate(validation_loader), total=len(validation_loader), desc=f"Epoch {epoch}, Validation Loss Collection")
        for val_batch_idx, (val_ip, val_label, _idx) in progress_bar:
            val_ip, val_label = val_ip.cuda(), val_label.cuda()
            outputs_val = model(val_ip)
            loss_val = criterion_evaluating(outputs_val, val_label)
            validation_losses.extend(loss_val.detach().cpu().numpy())
            validation_labels.extend(val_label.detach().cpu().numpy())
            
    return train_losses, validation_losses, train_labels, validation_labels


def main(
        data_path: str = './Data',
        dataset: str = 'CIFAR100',
        model_arch: str = 'resnet18',
        workers: int = 4,
        epochs: int = 164,
        start_epoch: int = 0,
        batch_size: int = 128,
        test_batch_size: int = 256,
        val_split_ratio: float = 0.1,
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
):
    args = locals()
    
    allowed_datasets = {'cifar100', 'imagenet'}  # CIFAR10 not implemented in dataset_utils
    if dataset.lower() not in allowed_datasets:
        raise ValueError(f"Invalid dataset '{dataset}'. Allowed values are: 'CIFAR100', 'ImageNet'.")

    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)

    best_acc = 0
    if not os.path.isdir(checkpoint_path):
        Utils.mkdir_p(checkpoint_path)

    train_loader, eval_train_loader, validation_loader, test_loader, meta = Utils.get_loss_collecting_dataloaders(**args)
    
    # Save split for reproducibility
    split_dir = './Splits'
    os.makedirs(split_dir, exist_ok=True)

    # Pick a clear name; matches your run naming
    split_path = os.path.join(split_dir, f'{dataset}_{model_arch}_seed{manual_seed}_split.pkl')

    split_payload = {
        'dataset': dataset,
        'model_arch': model_arch,
        'manual_seed': manual_seed,
        'val_split_ratio': val_split_ratio,   # this is your CLI arg; FYI dataset_utils uses 0.1/0.01 defaults unless you wire it through
        'train_indices': meta['train_indices'],
        'val_indices': meta['val_indices'],
    }

    with open(split_path, 'wb') as f:
        pickle.dump(split_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'[*] Saved train/val split to {split_path}')
    
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
    
    title = f"{dataset}_{model_arch}"
    save_str = f"{title}_seed{manual_seed}"

    os.makedirs(logpath, exist_ok=True)

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
        print("Evaluation only mode. Exiting training loop.")
        return

    stored_results = {
        'epoch': [],
        'tr_loss': [],
        'val_loss': [],
        'tr_lbl': [],
        'val_lbl': [],
    }

    for epoch_no in range(start_epoch, epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f'\nEpoch: [{epoch_no + 1} | {epochs}] LR: {current_lr}')
        
        train_loss, train_acc = Utils.train_epoch(train_loader, model, criterion, optimizer, epoch_no)
        test_loss, test_acc = Utils.test_epoch(test_loader, model, criterion, epoch_no)
        scheduler.step()

        logger.append([current_lr, train_loss, test_loss, train_acc, test_acc])

        storing_train_loss, storing_val_loss, train_labels, val_labels = get_train_validation_losses(
            model, eval_train_loader, validation_loader, epoch_no)
        
        stored_results['epoch'].append(epoch_no)
        stored_results['tr_loss'].append(storing_train_loss)
        stored_results['val_loss'].append(storing_val_loss)
        stored_results['tr_lbl'].append(train_labels)
        stored_results['val_lbl'].append(val_labels)

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

    scores_dir = 'Scores'
    if not os.path.isdir(scores_dir):
        Utils.mkdir_p(scores_dir)
        
    storing_name = os.path.join(scores_dir, f'{save_str}_losses.pickle')
    with open(storing_name, 'wb') as f:
        pickle.dump(stored_results, f, pickle.HIGHEST_PROTOCOL)

    print(f'Best test accuracy: {best_acc:.2f}')
    print(f"Loss trajectories saved to: {storing_name}")


if __name__ == '__main__':
    fire.Fire(main)
