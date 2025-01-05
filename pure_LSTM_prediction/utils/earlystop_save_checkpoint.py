import numpy as np
import torch 
import os


def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj == True:
        if epoch % 10 == 0: # update every 20 epochs
            lr_adjust = {epoch: args.lr * (0.5 ** ((epoch-1) // 20))}
        
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print('\nUpdating learning rate to {}'.format(lr))





class EarlyStopping:
    def __init__(self, args, patience=20, verbose=True):
        self.args = args
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.verbose = verbose
        self.early_stop = False
        self.val_loss_min = np.inf
    
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint_model(model=model, val_loss=val_loss)
        elif score < self.best_score:   # model not better
            self.counter += 1
            print(f'\nEarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:   # model is getting better 
            self.best_score = score
            self.save_checkpoint_model(model=model, val_loss=val_loss)
            self.counter = 0
            

    def save_checkpoint_model(self, model, val_loss):
        if self.verbose:
            print(f'\nValidation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        path = os.path.join(self.args.checkpoints, self.args.setting)
        if not os.path.exists(path):
            os.makedirs(path)
        model_save_path = path + '/' + 'checkpoint.pth'
        torch.save(model.state_dict(), model_save_path)
        self.val_loss_min = val_loss
        
        
 
