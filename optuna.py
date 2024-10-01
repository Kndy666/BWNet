import os
import time
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from data import Dataset_Pro
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from model.bwnet_lagconv import BWNet, summaries, init_weights
from torch.utils.tensorboard import SummaryWriter
import optuna

SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.benchmark = False
cudnn.deterministic = True

# Hyperparameters to be optimized by Optuna
epochs = 1000
start_epoch = 550
batch_size = 64
ckpt = 10

model = BWNet().cuda()
weight_path = 'weights/BWNET_LAGConv_T3/model_epoch_550.pth'
if os.path.exists(weight_path):
    model.load_state_dict(torch.load(weight_path))
else:
    init_weights(model.raise_dim, model.layers, model.bw_output, model.to_output)
summaries(model, grad=True)

# Criterion
criterion = nn.L1Loss().cuda()

############################# main functions ###############################

def save_checkpoint(model, epoch):
    model_out_path = os.path.join("weights/BWNET_LAGConv_T3", "model_epoch_{}.pth".format(epoch))
    torch.save(model.state_dict(), model_out_path)


def train_optuna(training_data_loader, validate_data_loader, trial):
    t_start = time.time()
    print('Start training...')
    
    # Hyperparameters from Optuna trial
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
    step_size = trial.suggest_int('step_size', 50, 200)
    gamma = trial.suggest_uniform('gamma', 0.1, 0.9)

    optimizer = optim.Adam([{'params': model.parameters(), 'initial_lr': lr}], lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma, last_epoch=start_epoch)

    val_loss, train_loss = [], []
    writer = SummaryWriter('loss_data/lagconv_t3_optuna')

    # train
    for epoch in range(start_epoch, epochs, 1):
        model.train()
        epoch_train_loss = []
        for iteration, batch in enumerate(training_data_loader, 1):
            gt, pan, ms = batch[0].cuda(), batch[3].cuda(), batch[4].cuda()
            optimizer.zero_grad()
            output = model(ms, pan)
            loss = criterion(output, gt)

            epoch_train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        lr_scheduler.step()
        t_loss = np.nanmean(np.array(epoch_train_loss))
        train_loss.append(t_loss)
        print('Epoch: {}/{}  training loss: {:.7f}'.format(epochs, epoch, t_loss))
        
        # Record training loss to TensorBoard
        writer.add_scalar('Loss/train', t_loss, epoch)

        # validate
        with torch.no_grad():
            if epoch % 10 == 0:
                model.eval()
                epoch_val_loss = []
                for iteration, batch in enumerate(validate_data_loader, 1):
                    gt, pan, ms = batch[0].cuda(), batch[3].cuda(), batch[4].cuda()
                    sr = model(ms, pan)
                    loss = criterion(sr, gt)
                    epoch_val_loss.append(loss.item())
                v_loss = np.nanmean(np.array(epoch_val_loss))
                val_loss.append(v_loss)
                print('---------------validate loss: {:.7f}---------------'.format(v_loss))
                t_end = time.time()
                print('-------------------time cost: {:.4f}s--------------------'.format(t_end - t_start))

                # Record validation loss to TensorBoard
                writer.add_scalar('Loss/val', v_loss, epoch)

        # Save model at checkpoints
        if epoch % ckpt == 0:
            save_checkpoint(model, epoch)

    writer.close()  # Close the TensorBoard writer when done
    
    # Return validation loss at the end of the last epoch to be minimized
    return v_loss[-1]


# Optuna objective function
def objective(trial):
    train_set = Dataset_Pro('../../training_data/wv3/train_wv3.h5')
    training_data_loader = DataLoader(dataset=train_set, num_workers=8, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)
    validate_set = Dataset_Pro('../../training_data/wv3/valid_wv3.h5')
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=4, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)

    return train_optuna(training_data_loader, validate_data_loader, trial)


if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')  # We want to minimize validation loss
    study.optimize(objective, n_trials=50)  # Run for 50 trials
    print(f'Best trial: {study.best_trial.value}')
    print(f'Best hyperparameters: {study.best_trial.params}')
