import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
from argparse import ArgumentParser

from data import CIFAR10Data
from utils import *
from inceptionv4 import Inception4


def main(args):
    TRAIN_LOSS = []
    VAL_LOSS = []
    ACC = []
    best_acc = -1  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Device
    device = "cpu"
    if args.use_gpu and torch.cuda.is_available():
        device = "cuda"

    # Model
    model = Inception4().to(device)

    # Optimizer
    optimizer = optim.RMSprop(model.parameters(),
                    lr=args.learning_rate,
                    weight_decay=args.wd)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer,
                    gamma=args.lr_decay)

    # Dataloader
    data = CIFAR10Data(args)
    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()

    # Create Loss
    criterion = nn.CrossEntropyLoss()

    # Create checkpoint path
    path = os.path.join(
        args.base_dir,
        args.checkpoint_dir,
        "{}_{}".format(args.model_name, args.trial))
    if not os.path.exists(path):
        os.mkdir(path)
    
    if args.resume:
        stat = utils.load_checkpoint(args)
        model.load_state_dict(stat["state_dict"])
        start_epoch = stat["start_epoch"]+1
    
    print("Start Training!")
    for epoch in range(start_epoch, args.max_epochs):
        scheduler.step()
        train_stat = train(train_loader, model, criterion, optimizer, epoch, args)
        
        val_stat = validate(val_loader, model, criterion, optimizer, epoch, args, best_acc)

        TRAIN_LOSS.append(train_stat["train_loss"])
        VAL_LOSS.append(val_stat["val_loss"])
        best_acc = val_stat["best_acc"]

        torch.save({
            "state_dict" : model.state_dict(),
            "start_epoch" : epoch+1,
            "loss" : [train_state["train_loss"], val_state["val_loss"]],
            "acc" : [train_state["train_acc"], val_state["val_acc"]],
        }, path+"/checkpoint_epoch_{}.pt".format(epoch))
        

if __name__ == "__main__":
    parser = ArgumentParser()

    # -------------------------------------------------------------#
    parser.add_argument("--data_dir", type=str, default="/dataset/CIFAR")
    parser.add_argument("--base_dir", type=str, default="/root/volume/Paper/MLVC_Internship")
    parser.add_argument("--checkpoint_dir", type=str,default="checkpoint")
    parser.add_argument("--use_gpu", action="store_true", default=False)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--save_interval", type=int, default=5)
    # -------------------------------------------------------------#
    parser.add_argument("--model_name", required=True, type=str)
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--resume_epoch", type=int, default=None)
    # -------------------------------------------------------------#
    parser.add_argument("--pretrained", type=bool, default=False)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    
    parser.add_argument("--learning_rate", type=float, default=0.045)
    parser.add_argument("--lr_decay", type=float, default=0.94)
    parser.add_argument("--eps", type=float, default=1.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--wd", type=float, default=0.9)
    # -------------------------------------------------------------#

    args = parser.parse_args()
    torch.cuda.empty_cache()
    main(args)