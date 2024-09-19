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
from model.u2net_hp import U2Net, summaries

SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.benchmark = False
cudnn.deterministic = True


#################### initializing hyper-parameters ####################
lr = 0.001
ckpt = 10
epochs = 500
start_epoch = 0
batch_size = 2

model = U2Net(dim=32).cuda()

summaries(model, grad=True)


################### initializing criterion and optimizer ###################
criterion = nn.L1Loss(size_average=True).cuda()
# criterion = nn.MSELoss(size_average=True).cuda()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=100, gamma=0.5)


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
                print('------------------validate loss: {:.7f}---------------'.format(v_loss))
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
