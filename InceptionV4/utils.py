import os
import sys
import time

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

def train(train_loader, model, criterion, optimizer, epoch, args):
    print('Epoch {}/{}'.format(epoch + 1, args.max_epochs))
    print('-' * 10)
    
    start_time = time.time()

    # switch to train mode
    model.train()

    train_loss = 0
    correct = 0
    total = 0

    for i, (input, target) in enumerate(train_loader):
        input = input.to(device)
        target = target.to(device)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()

        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    end_time = time.time()
    print('TrainLoss: %.3f | TrainAcc: %.3f%% (%d/%d) | Time Elapsed %.3f sec' % (train_loss/(i+1), 100.*correct/total, correct, total, end_time-start_time))

    return {"train_loss" : train_loss/(len(train_loader)),
            "train_acc" : 100.*correct/total}

    
def load_checkpoint(args):
    path = os.path.join(args.base_dir, args.checkpoint_dir, "{}_{}".format(args.model_name, args.trial))
    return torch.load(path+"epoch_{}".format(args.resume_epoch))


def validate(val_loader, model, criterion, epoch, args, best_acc):

    model.eval()

    val_loss = 0
    correct = 0
    total = 0
    acc = 0

    for i, (input, target) in enumerate(val_loader):
        input = input.to(device)
        target = target.to(device)

        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        val_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    acc = (100.) * (correct/total)
    print('Val Loss: %.3f | Val Acc: %.3f%% (%d/%d)' % (val_loss/(i+1), acc, correct, total))

    if acc > best_acc:
        best_acc = acc
        print('Saving model..')
        path = os.path.join(args.base_dir, args.checkpoint_dir, "{}_{}".format(args.model_name, args.trial))
        torch.save(model.state_dict(), path+"/Best_model_"+str(epoch)+".pth)
    return {"val_loss" : (val_loss/len(val_loader)),"val_acc" : acc,"best_acc" : best_acc}