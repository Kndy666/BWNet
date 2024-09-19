import os
import time
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from data_hp import Dataset_Pro
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from BWNet import BWNet, summaries


SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.benchmark = False
cudnn.deterministic = True


#################### initializing hyper-parameters ####################
lr = 0.005
ckpt = 50
epochs = 200
start_epoch = 0
batch_size = 32
weight_path = '1000.pth'

model = BWNet().cuda()
summaries(model, grad=True)


################### load pre-trained weights ###########################
BWNet_weight = model.state_dict()
PanNet_weight = torch.load(weight_path)

for key in BWNet_weight.keys():
    if key in PanNet_weight.keys():
        BWNet_weight[key] = PanNet_weight[key]

model.load_state_dict(BWNet_weight)


################### initializing criterion and optimizer ###################
def update(model):
    paras = []
    for name, p in model.named_parameters():
        if name in ['conv1.weight', 'conv1.bias']:
            p.requires_grad = True
            paras.append(p)
        elif name in PanNet_weight.keys():
            p.requires_grad = False
            print(name)
        else:
            p.requires_grad = True
            paras.append(p)
    return paras

criterion = nn.L1Loss(size_average=True).cuda()
#criterion = nn.MSELoss(size_average=True).cuda()
paras = update(model)
optimizer = optim.Adam(paras, lr=lr)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=50, gamma=0.2)


############################# main functions ###############################
def save_checkpoint(model, epoch):
    model_out_path = 'weights' + '/' + "{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)


def train(training_data_loader, validate_data_loader, start_epoch=0):
    t_start = time.time()
    print('Start training...')
    val_loss, train_loss = [], []

    # train
    for epoch in range(start_epoch, epochs, 1):
        epoch += 1
        model.train()
        epoch_train_loss = []
        for iteration, batch in enumerate(training_data_loader, 1):
            gt, lms, pan, pan_hp, ms, ms_hp = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), \
                                              batch[3].cuda(), batch[4].cuda(), batch[5].cuda()
            optimizer.zero_grad()
            sr = model(ms_hp, pan_hp, ms)
            loss = criterion(sr, gt)
            epoch_train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        lr_scheduler.step()
        t_loss = np.nanmean(np.array(epoch_train_loss))
        train_loss.append(t_loss)
        print('Epoch: {}/{}  training loss: {:.7f}'.format(epochs, epoch, t_loss))

        # validate
        with torch.no_grad():
            if epoch % 10 == 0:
                model.eval()
                epoch_val_loss = []
                for iteration, batch in enumerate(validate_data_loader, 1):
                    gt, lms, pan, pan_hp, ms, ms_hp = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), \
                                                      batch[3].cuda(), batch[4].cuda(), batch[5].cuda()
                    sr = model(ms_hp, pan_hp, ms)
                    loss = criterion(sr, gt)
                    epoch_val_loss.append(loss.item())
                v_loss = np.nanmean(np.array(epoch_val_loss))
                val_loss.append(v_loss)
                print('---------------validate loss: {:.7f}---------------'.format(v_loss))
                t_end = time.time()
                print('-------------------time cost: {:.4f}s--------------------'.format(t_end - t_start))

        # save data
        if epoch % ckpt == 0:
            # save parameters
            save_checkpoint(model, epoch)

            # save train loss
            f_train_loss = open("loss_data/train_loss.txt", 'r+')
            f_train_loss.read()
            for i in range(len(train_loss)):
                f_train_loss.write(str(train_loss[i]))
                f_train_loss.write('\n')
            f_train_loss.close()
            train_loss = []

            # save val loss
            f_val_loss = open("loss_data/validation_loss.txt", 'r+')
            f_val_loss.read()
            for i in range(len(val_loss)):
                f_val_loss.write(str(val_loss[i]))
                f_val_loss.write('\n')
            f_val_loss.close()
            val_loss = []


if __name__ == "__main__":
    train_set = Dataset_Pro('D:/Study/pansharpening_new_data/training_wv3/train_wv3.h5')
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)
    validate_set = Dataset_Pro('D:/Study/pansharpening_new_data/training_wv3/valid_wv3.h5')
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)
    train(training_data_loader, validate_data_loader, start_epoch)

