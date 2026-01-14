import numpy as np
import pandas as pd

import torch
from torch import nn
from torchvision import transforms as T
from torchvision.ops import sigmoid_focal_loss

import os
from dataclasses import dataclass
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.metrics import balanced_accuracy_score
import argparse
from tqdm import tqdm
import csv

from dataloader import ISICDataset2020, ISICDataset2018
import models
from loss import FocalLoss

import warnings

def unfreeze_stage(backbone, model, stage):
    """
    Gradually unfreezes layers depending on model type and stage index.
    """

    if isinstance(model, nn.DataParallel):
        model = model.module

    # RESNET
    if backbone in ['resnet50', 'resnet50_224', 'resnet18_224']:
        if stage == 1:
            for p in model.layer4.parameters(): p.requires_grad = True
        elif stage == 2:
            for p in model.layer3.parameters(): p.requires_grad = True
        elif stage == 3:
            for p in model.layer2.parameters(): p.requires_grad = True
        elif stage >= 4:
            for p in model.parameters(): p.requires_grad = True

    # EFFICIENTNET
    elif backbone == 'efficientnet':
        if stage == 1:
            for p in model.blocks[-3:].parameters(): p.requires_grad = True
        elif stage == 2:
            for p in model.blocks[-6:].parameters(): p.requires_grad = True
        elif stage >= 3:
            for p in model.features.parameters(): p.requires_grad = True

    # SWIN TRANSFORMER
    elif backbone == 'swin':
        if stage == 1:
            for p in model.layers[-1].parameters(): p.requires_grad = True
        elif stage == 2:
            for p in model.layers[-2].parameters(): p.requires_grad = True
        elif stage >= 3:
            for p in model.parameters(): p.requires_grad = True

    # CUSTOM CNN
    elif backbone in ['conv_T', 'conv_B']:
        if stage == 1:
            for p in model.features[-1].parameters(): p.requires_grad = True
        elif stage == 2:
            for p in model.features[-2].parameters(): p.requires_grad = True
        elif stage >= 3:
            for p in model.features.parameters(): p.requires_grad = True

    else:
        raise ValueError("Unknown model type — cannot unfreeze stage")


def main(model_id, dataset, args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    MODEL_FACTORY = {
    "resnet50": models.ResNet_50,
    "efficientnet": models.EfficientNet,
    "swin": models.Swin_B,
    "conv_B": models.get_ConvBase,
    "conv_T": models.get_ConvTiny,
    "resnet50_224": models.ResNet_50_224,
    "resnet18_224": models.ResNet18_224
    }

    pos_weights = None
    sampler = None    
    # load the model with correct output classes
    if dataset == 'isic2018':
        in_chnls = 3
        num_classes = 7
        
        # Load model
        if model_id not in MODEL_FACTORY:
            raise ValueError(f"{model_id} is an invalid model name")

        model, transformation = MODEL_FACTORY[model_id](in_channels=in_chnls, num_classes=num_classes)
        # init the dataset
        df_path_train = '/scratch/gssodhi/melanoma/isic2018/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv'
        df_path_val = '/scratch/gssodhi/melanoma/isic2018/ISIC2018_Task3_Validation_GroundTruth/ISIC2018_Task3_Validation_GroundTruth.csv'
        df_path_test = '/scratch/gssodhi/melanoma/isic2018/ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv'

        root_train = '/scratch/gssodhi/melanoma/isic2018/ISIC2018_Task3_Training_Input'
        root_val = '/scratch/gssodhi/melanoma/isic2018/ISIC2018_Task3_Validation_Input'
        root_test = '/scratch/gssodhi/melanoma/isic2018/ISIC2018_Task3_Test_Input'

        df_train = pd.read_csv(df_path_train)
        df_val = pd.read_csv(df_path_val)
        df_test = pd.read_csv(df_path_test)
        
        train_dataset = ISICDataset2018(df_train, root_train, transformation)
        val_dataset = ISICDataset2018(df_val, root_val, transformation)
        test_dataset = ISICDataset2018(df_test, root_test, transformation)

        train_loader = DataLoader(dataset=train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=args.num_worker, pin_memory=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size = args.batch_size, shuffle=False, num_workers=args.num_worker, pin_memory=True)
        # test_loader = DataLoader(dataset=test_dataset, batch_size = args.batch_size, shuffle=False, num_workers=args.num_worker, pin_memory=True)

        print("=> Data Prep isic2018 Done")

        class_cols = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

        train_counts = df_train[class_cols].sum().astype(int)
        total = sum(train_counts)

        pos_weights = torch.tensor(total / train_counts.values, dtype=torch.float32).to(device)

    elif dataset == 'isic2020':
        in_chnls = 3
        num_classes = 2
        
        # load model
        if model_id not in MODEL_FACTORY:
            raise ValueError(f"{model_id} is an invalid model name")

        model, transformation = MODEL_FACTORY[model_id](in_channels=in_chnls, num_classes=num_classes)

        df = pd.read_csv("/scratch/gssodhi/melanoma/ISIC_2020_Training_GroundTruth.csv")

        train_df, val_df = train_test_split(
            df,
            test_size=0.2,
            stratify=df["target"],
            random_state=42,
        )
        root_2020 = '/scratch/gssodhi/melanoma/train'

        if len(val_df['target'].unique()) < 2:
            raise ValueError("Validation set does not contain both classes! Increase test_size or check data.")

        labels = np.array(train_df.target)  # shape [N], values 0/1

        # computing class counts
        class_counts = np.bincount(labels)
        # eg [950 negatives, 50 positives]

        # computing inverse frequency weights
        class_weights = 1. / class_counts
        # eg [1/950, 1/50]

        # assign weight to each sample
        sample_weights = class_weights[labels]

        # create sampler
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=len(sample_weights),  # or len(dataset)
            replacement=True
        )

        if model_id in ['resnet18_224', 'resnet50_224', 'conv_B', 'efficientnet', 'swin']:
            # NOTE: Currently there are separate tranformation splits for resnet18_224, resnet50_224, conv_B only
            train_transform = transformation["train"]
            val_transform   = transformation["val"]

            train_dataset = ISICDataset2020(train_df, root_2020, train_transform)
            val_dataset = ISICDataset2020(val_df, root_2020, val_transform)
        else:
            train_dataset = ISICDataset2020(train_df, root_2020, transformation)
            val_dataset = ISICDataset2020(val_df, root_2020, transformation)

        if sampler is not None and args.loss != 'focal':
            train_loader = DataLoader(dataset=train_dataset, 
                                      batch_size = args.batch_size, 
                                      num_workers=args.num_worker, 
                                      pin_memory=True,
                                      persistent_workers=True,
                                      prefetch_factor=4,
                                      sampler=sampler)
        else:
            train_loader = DataLoader(dataset=train_dataset, 
                                      batch_size = args.batch_size, 
                                      shuffle=True, 
                                      num_workers=args.num_worker, 
                                      pin_memory=True,
                                      persistent_workers=True,
                                      prefetch_factor=4)
        
        val_loader = DataLoader(dataset=val_dataset, 
                                batch_size=args.batch_size, 
                                shuffle=False, num_workers=args.num_worker, 
                                pin_memory=True,
                                persistent_workers=True,
                                prefetch_factor=4)

        print(f"Dataset size: {len(train_dataset)}")
        print(f"Number of batches: {len(train_loader)}")
        print(f"Effective batch size per step: {args.batch_size}")

        # NOTE: To use weighted loss uncommected the two lines below, but make sure to comment out lines for 
        # sampler as well
        
        # counts = train_df['target'].value_counts().sort_index()  # index 0 and 1
        # pos_weights = torch.tensor([counts[1] / counts.sum(), counts[0] / counts.sum()], dtype=torch.float32).to(device)

        print("=> Data Prep isic2020 Done")
        
    else:
        raise ValueError(f"{dataset} is an invalid dataset name")

    
    # move model to device
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"=> Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    else:
        print("=> Using 1 GPU")

    # if freezing layers
    if args.freeze:
        for param in model.parameters():
            param.requires_grad = False

        # selectively unfreeze the head (each model type has differnet name for a head)
        if hasattr(model, "fc"):  # ResNet-style
            for param in model.fc.parameters():
                param.requires_grad = True
        elif hasattr(model, "classifier"):  # EfficientNet or Swin
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif hasattr(model, "MLP"):  # custom CNN (Base/Tiny)
            for param in model.MLP.parameters():
                param.requires_grad = True
        else:
            raise ValueError("Unknown model head — cannot unfreeze classifier.")

    if args.loss == 'CE':
        if pos_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=pos_weights)
        else:
            criterion = nn.CrossEntropyLoss()
    elif args.loss == 'focal':
        if pos_weights is not None:
            warnings.warn("Weighted loss is not implemented for focal loss")
        criterion = FocalLoss()
    else:
        raise ValueError("Not a valid loss fuunction")

    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                     lr=args.lr, 
                     weight_decay=5e-4)
    
    # optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    os.makedirs(args.log_file_path, exist_ok=True)  # make sure directory exists

    csv_file = os.path.join(args.log_file_path, f"metrics_{args.run}.csv")

    print(f"Loggin in {csv_file}")

    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
                "f1_score",
                "auc",
            ])

    START = 0
    resume = args.resume
    if resume:
        if not os.path.isfile(args.resume_model_path):
            raise ValueError(f"{args.resume_model_path} is an invalid model path")

        print(f"Loading model from: {args.resume_model_path}")
        checkpoint = torch.load(args.resume_model_path, map_location=device, weights_only=False)
    
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
    
        START = checkpoint['epoch'] + 1

    print("=> TRAINING STARTED")
    # If using layer freezing: the epochs to unfreeze some portion of the layer
    e1, e2, e3, e4 = 5, 10, 15, 20
    for epoch in range(START, args.epochs):

        if args.freeze and epoch in [e1, e2, e3, e4]:
            stage = {e1:1, e2:2, e3:3, e4:4}[epoch]
            unfreeze_stage(model_id, model, stage)
            
            print(f"Unfreezing Stage: {stage}")
            
            # update optmizer to train unfrozen layers
            optimizer = Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr,
                weight_decay=5e-4
            )

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                }, filename=args.save_model_path)
        
        test_loss, test_acc, f1, auc = test(model, val_loader, criterion, device)
        
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                train_loss,
                train_acc,
                test_loss,
                test_acc,
                f1,
                auc
        ])

        print(f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {test_loss:.4f} | Val Acc: {test_acc:.4f} | "
            f"F1: {f1:.4f} | AUC: {auc:.4f}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0

    all_labels = []
    all_preds = []

    for x, y in tqdm(loader, desc='train'):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        # forward
        out = model(x)
        
        # loss
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # predictions
        preds = out.argmax(dim=1)

        all_labels.extend(y.detach().cpu().numpy())
        all_preds.extend(preds.detach().cpu().numpy())

    avg_loss = running_loss / len(loader)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)

    return avg_loss, balanced_acc



def test(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for x, y in tqdm(loader, desc='test'):
            x, y = x.to(device), y.to(device)

            out = model(x)

            loss = criterion(out, y)
            running_loss += loss.item()

            # Softmax across class dimension
            probs = torch.softmax(out, dim=1)
            preds = probs.argmax(dim=1)

            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(y.detach().cpu().numpy())
            
            # for ROC-AUC: if 2-class, pick class 1 probabilities
            if probs.shape[1] == 2:
                probs_for_auc = probs[:, 1]
            else:
                probs_for_auc = probs  # multi-class (use full matrix later)

            all_probs.extend(probs_for_auc.detach().cpu().numpy())

    avg_loss = running_loss / len(loader)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)

    # compute metrics
    try:
        if len(set(all_labels)) == 2:
            auc = roc_auc_score(all_labels, all_probs)
        else:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
        f1 = f1_score(all_labels, all_preds, average='macro')
    except ValueError:
        # occurs if only one class present in y
        print("WARNING: hit nan in test method")
        auc, f1 = float('nan'), float('nan')

    return avg_loss, balanced_acc, f1, auc

def save_checkpoint(state, filename='model'):
    torch.save(state, filename + '.pth.tar')


@dataclass
class model_config:
    batch_size: int = 128
    num_worker:int = 8
    lr: float = 1e-5
    epochs: int = 50
    resume: bool = False
    resume_model_path: str = '/scratch/gssodhi/melanoma/checkpoint/chkpt_efNet'
    save_model_path: str = '/scratch/gssodhi/melanoma/checkpoint/chkpt_efNet'
    log_file_path: str = '/home/gssodhi/melanoma/baselines/data/'
    run: str = 'run'
    freeze: bool = False
    loss: str = 'BCE'


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='RUN Baseline model of ISIC')

    parser.add_argument('--data_flag',
                        default='isic2020',
                        type=str)
    parser.add_argument('--model_flag',
                        default='resnet50',
                        help='choose backbone from resnet, swin, efficientnet',
                        type=str)
    parser.add_argument('--resume', 
                        action='store_true', 
                        help='resume from checkpoint')
    parser.add_argument('--freeze', 
                        action='store_true', 
                        help='if freezing layer pass this arg. Responsible for setting up optimizer properly')
    parser.add_argument('--batch_size', 
                        default=128,
                        type=int)
    parser.add_argument('--loss', 
                        default='CE',
                        type=str)
    parser.add_argument('--run', 
                        type=str)


    cli_args = parser.parse_args()

    model_id = cli_args.model_flag
    dataset = cli_args.data_flag
    resume_bool = cli_args.resume
    batch_size_cli = cli_args.batch_size
    run = cli_args.run

    args = model_config(
        resume_model_path = f'/scratch/gssodhi/melanoma/checkpoint/chkpt_{model_id}_{dataset}_{run}.pth.tar',
        save_model_path = f'/scratch/gssodhi/melanoma/checkpoint/chkpt_{model_id}_{dataset}_{run}',
        resume = resume_bool,
        batch_size=batch_size_cli,
        log_file_path = f'/home/gssodhi/melanoma/baselines/data/{model_id}_{dataset}',
        run = run, 
        freeze = cli_args.freeze,
        loss = cli_args.loss
    )

    if cli_args.freeze and model_id not in ['resnet50', 'efficientnet', 'conv_T', 'conv_b', 'resnet50_224']:
        raise ValueError(f"Layer freezing not implemented for {model_id}." +
                         " Please change model or don't pass freeze arg")

    if cli_args.loss not in ['CE', 'focal']:
        raise ValueError(f"{cli_args.loss} is not an implemented loss function")
    
    if cli_args.loss == 'focal' and dataset == 'isic2018':
        raise ValueError("The current implementation of focal loss is for binary classification only and isic2018 is mutliclass")
    
    main(model_id, dataset, args)
    
