import os
import sys
import torch
import torch.nn as nn

from filelock import FileLock
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

import ray
from ray.util.sgd.torch import TrainingOperator
from ray.util.sgd.utils import override

sys.path.insert(1, '/root/volume/Paper/MLVC_Internship/models')
from models.incep_resnet import IncepResNetV2

class CIFAR10Module(TrainingOperator):
    @override(TrainingOperator)
    def setup(self, args):
        # Create model
        model = IncepResNetV2()

        # Create optimizer
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=args["lr"],
            weight_decay=args["wd"])

        # Load in training and validation data
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))])

        with FileLock(".ray.lock"):
            train_dataset = CIFAR10(
                root=args["data_dir"],
                train=True,
                download=True,
                transform=transform_train)
            validation_dataset = CIFAR10(
                root=args["data_dir"],
                train=False,
                download=False,
                transform=transform_test)
        
        if args["smoke_test"]:
            train_dataset = Subset(train_dataset, list(range(64)))
            validation_dataset = Subset(validation_dataset, list(range(64)))

        train_loader = DataLoader(
            train_dataset, batch_size=args["batch_size"], num_workers=args["num_workers"], shuffle=True)
        validation_loader = DataLoader(
            validation_dataset, batch_size=args["batch_size"], num_workers=args["num_workers"])

        # Create scheduler
        """
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[150, 250, 350], gamma=0.1)
        """
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=args["lr_decay"])
        
        # Create loss
        criterion = nn.CrossEntropyLoss()

        # Register all components
        self.model, self.optimizer, self.criterion, self.scheduler = self.register(
            models=model, optimizers=optimizer, criterion=criterion, schedulers=scheduler)
        self.register_data(
            train_loader=train_loader, validation_loader=validation_loader)