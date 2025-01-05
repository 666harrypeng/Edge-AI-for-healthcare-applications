from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'informer':Informer,
            'informerstack':InformerStack,
        }
        if self.args.model=='informer' or self.args.model=='informerstack':
            e_layers = self.args.e_layers if self.args.model=='informer' else self.args.s_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers, # self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
            'ETTm1':Dataset_ETT_minute,
            'ETTm2':Dataset_ETT_minute,
            'WTH':Dataset_Custom,
            'ECL':Dataset_Custom,
            'Solar':Dataset_Custom,
            'custom':Dataset_Custom,
        }   # values are Dataloaders
        
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        
        print(flag, "data_set length = ", len(data_set))
        
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def loss_visualization(self, loss_list, iter_list, setting, root_folder_path, flag):
        # root_folder_path should be like './checkpoints/' + setting +'/'
        if (flag == 'Training') or (flag == 'Train') or (flag == 'training') or (flag == 'train'):  # save the loss curve under the checkpoint folder
            print('>>> Training Visualization >>>')
            plt.figure(figsize=(10, 6))
            plt.plot(iter_list, loss_list, marker='o', linestyle='-', color='b')
            plt.title(f'{flag} Loss Curve')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.grid()
            # plt.text(0, 1, f'Note: {setting}', fontsize=10, color='gray')
            filename = root_folder_path + '/' + f'{flag}_loss_curve.jpg'
            plt.savefig(filename)
            print('<<< Training Visualization FINISH <<<')
        elif (flag == 'test') or (flag == 'testing') or (flag == 'Test') or (flag == 'Testing'):        # save the metrics visualization under the results/<setting> folder
            print('>>> Test Visualization >>>')
            
            ERROR_MAX_BOUND = 1e20
            
            error_values = loss_list
            error_types = iter_list
            error_full_names = ['MAE: Mean Absolute Error', 'MSE: Mean Squared Error', 
                    'RMSE: Root Mean Squared Error', 'MAPE: Mean Absolute Percentage Error', 
                    'MSPE: Mean Squared Percentage Error']
            
            if np.sum(np.array(error_values)[-2:] > ERROR_MAX_BOUND) > 0:   # MAPE or MSPE may be close to inf, only visualize the first three elements
                error_values = loss_list[:3]
                error_types = iter_list[:3]
                error_full_names = ['MAE: Mean Absolute Error', 'MSE: Mean Squared Error', 'RMSE: Root Mean Squared Error']
                
            print('error_values shape = ', error_values.shape, error_values)
            print('error_types shape = ', error_types.shape, error_types)

            plt.figure(figsize=(12, 8))
            bars = plt.bar(error_types, error_values)
            plt.yscale('log')   # avoid large values, especially for the MSPE error when we have the actual error value is near 0
            plt.xlabel('Error Types')
            plt.ylabel('Error Value (log scale)')
            plt.title(f'Error Metrics for {flag}', fontsize=18, fontweight='bold')
            for bar in bars:    # Add value labels on top of bars
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2.0, height * 1.05,
                        f'{height:.2e}', ha='center', va='bottom', fontsize=12, fontweight='bold')
            # label the settings
            # plt.text(0.5, -0.15, f"Note: {setting}", 
            #         fontsize=12, color='gray', ha='center', va='center', transform=plt.gca().transAxes)
            plt.legend(bars, error_full_names, loc='upper left', fontsize=10)

            plt.tight_layout()  # Optimize the layout to avoid clipping of labels
            
            filename = root_folder_path + f'{flag}_metrics_visualization.jpg'
            plt.savefig(filename, dpi=600)
            print('<<< Test Visualization FINISH <<<')
            
        else:
            print('Unknown Flag -> Try to fix it or ignore the visualzation')

    
    
    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
            pred, true = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)     # "train_steps" == "total_batch_num" >>> len(train_loader) = dataset_len / batch_size
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        # for loss track
        loss_track_iter_list = []
        loss_track_iter_count = 0
        loss_value_track = []
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
                '''
                batch_x -> input    batch_y -> label (GT)   (probably)
                mark -> encoder/decoder additional features/encodings?
                '''
                
                iter_count += 1
                loss_track_iter_count += 1
                
                model_optim.zero_grad()
                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(pred, true)
                train_loss.append(loss.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    
                    ## for tracking & visualization
                    loss_track_iter_list.append(loss_track_iter_count)
                    loss_value_track.append(loss.item())
                    ## 
                    
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)     # judge whether should early_stop && save checkpoint.pth
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        self.loss_visualization(loss_value_track, loss_track_iter_list, setting, path, flag='train')
        loss_track_iter_list = []
        loss_track_iter_count = 0
        loss_value_track = []
        
        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        # print('test shape:', preds.shape, trues.shape)
        print(f'test -> preds shape: {preds.shape}, trues shape: {trues.shape}')
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # print('test shape:', preds.shape, trues.shape)
        print(f'test -> preds shape: {preds.shape}, trues shape: {trues.shape}')
        

        # result save
        # folder_path = './results/' + setting +'/'
        folder_path = './checkpoints/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        metric_value_list = np.array([mae, mse, rmse, mape, mspe])
        error_types = np.array(['MAE', 'MSE', 'RMSE', 'MAPE', 'MSPE'])  # mainly for visualization
        
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path+'metrics.npy', metric_value_list)
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)

        self.loss_visualization(metric_value_list, error_types, setting, folder_path, flag='test')
        self.test_true_pred_curve_compare(preds, trues, folder_path)  # plot curves for comparison
        
        return

    def test_true_pred_curve_compare(self, preds, trues, folder_path):
        print('>>> Start Sample Test Pred&True Curves >>>')
        
        pred_samples, true_samples = preds[0], trues[0] # pick the first sets of preds & trues as samples
        x = np.arange(pred_samples.shape[1])

        # the default predict len is 24, we assume the pred_len%4==0
        if self.args.pred_len % 4 == 0:
            subplot_rows = self.args.pred_len//4
            fig, axs = plt.subplots(subplot_rows, 4, figsize=(20, 5*subplot_rows))
            fig.suptitle('Comparison of After-Test Prediction & Ground Truth', fontsize=18, fontweight='bold')

            axs = axs.flatten() # Flatten the 2D axes array so we can iterate through it easily
            
            # Loop over each row of the arrays and create a subplot for each
            for i in range(subplot_rows * 4):
                axs[i].plot(x, pred_samples[i], label='pred', color='blue', marker='o')  
                axs[i].plot(x, true_samples[i], label='true', color='red', linestyle='--', marker='x') 
                
                # Set title and labels
                axs[i].set_title(f'Pred & True {i+1}', fontsize=10)
                axs[i].set_xlabel('Time', fontsize=8)
                axs[i].set_ylabel('Target Value', fontsize=8)
                
                # Add grid for readability
                axs[i].grid(True)
                
                # Add legend
                axs[i].legend(fontsize=8)
            
            img_name = folder_path + 'pred_true_comparision_curves.jpg'
            plt.savefig(img_name)
        else:   # not 4-times pred_len
            print('Not a 4-times pred_len -> disable the after-test pred & true comparision curves')
        
        ### Plot the Good prediction rate
        if len(np.shape(trues)) == 3 and len(np.shape(preds)) == 3: # Trues & Preds must have three dimensions 
            ZERO_EPSILON = 1e-6
            LOW_ERROR_BOUND = 0.05
            UP_ERROR_BOUND = 0.35
            error_margin_list = np.round(np.linspace(start=LOW_ERROR_BOUND, stop=UP_ERROR_BOUND, num=10), decimals=3)
            good_pred_rate_list = []
            
            trues_no_zero = trues + (trues < ZERO_EPSILON) * ZERO_EPSILON
            preds_no_zero = preds + (preds < ZERO_EPSILON) * ZERO_EPSILON
            
            relative_error = np.abs(trues_no_zero - preds_no_zero) / trues_no_zero

            batch_num, pred_len, output_dim = np.shape(trues_no_zero)
            total_element_num = batch_num * pred_len * output_dim
            
            for error_margin in error_margin_list:
                good_pred_rate = np.sum(relative_error < error_margin) / total_element_num
                good_pred_rate_list.append(good_pred_rate)
                print(f'error bound {error_margin}: good prediction rate: {good_pred_rate}')

                
            good_pred_rate_list = np.array(good_pred_rate_list)
            
            plot_rows = self.args.pred_len//4
            plt.figure(figsize=(20, 10))
            plt.bar(error_margin_list, good_pred_rate_list, width=0.01, color='skyblue')

            # Adding labels and title
            plt.xlabel('Error Margin')
            plt.ylabel('Good Prediction Rate')
            plt.title('Error Margin & Good Prediction Rate')
            plt.xticks(error_margin_list)
            plt.ylim(0, 1)

            # Adding numerical values on top of bars
            for i in range(len(error_margin_list)):
                plt.text(error_margin_list[i], good_pred_rate_list[i] + 0.01, 
                        f'{good_pred_rate_list[i]:.3f}', ha='center')
            
            img_name = folder_path + 'pred_good_rate.png'
            plt.savefig(img_name)
        else:
            print('Shape of Trues or Preds is not 3-dim >>> skip prediction good rate calculation')
        
        print('<<< Finish Sample Test Pred&True Curves <<<')

    
    
    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        
        preds = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
            pred, true = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path+'real_prediction.npy', preds)
        
        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        if self.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)

        return outputs, batch_y
