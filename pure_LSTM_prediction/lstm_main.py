import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import os
from lstm_model import LSTMModel
from utils.data_preprocess import Data_preprocessor
from utils.visualization_tools import Visualization
from utils.earlystop_save_checkpoint import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

###################### Args ################################
parser = argparse.ArgumentParser(description='Pure LSTM Time-Series Forecsating')
parser.add_argument('--model', type=str, default='PureLSTM', help='model of experiment')

parser.add_argument('--data', type=str, required=True, default='ETTh1', help='data (e.g., ETTh1)')
# parser.add_argument('--data_path', type=str, default='/home/ypengbb/Respiratory-Anomaly-Detection/Common_datasets/ETT/ETTh1.csv', help='data path')
parser.add_argument('--time_steps', type=int, default=24, help='time steps for sequence length')
parser.add_argument('--hidden_size', type=int, default=64, help='Number of LSTM units')
parser.add_argument('--num_layers', type=int, default=4, help='Number of LSTM layers')
parser.add_argument('--input_size', type=int, default=1, help='Number of input size (features)')  # optional -> will be specified manually during training
parser.add_argument('--output_size', type=int, default=1, help='Number of LSTM prediction length')
parser.add_argument('--lr', type=float, default=0.0005, help='optimizer learning rate')
parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
parser.add_argument('--train_epochs', type=int, default=500, help='train epochs')
parser.add_argument('--pred_sample_num', type=int, default=50, help='prediction sample number for visualization')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
# parser.add_argument('--exp_itr', type=int, default=1, help='experiment times')    # temporarily banned (from: 2024 Nov 13)
parser.add_argument('--device', type=str, default='cuda', help='experiment times')
parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
parser.add_argument('--features', type=str, default='MS', help='forecasting task, options:[M, S, MS]; LSTM default set MS')
parser.add_argument('--random_state', type=int, default=21, help='random state seed')
parser.add_argument('--do_predict', type=bool, default=True, help='whether to predict unseen future data')

parser.add_argument('--lradj', type=bool, default=False, help='whether automatically adjust lr during training')
parser.add_argument('--pretrained_base', type=bool, default=False, help='whether continue to train a pre-trained and saved checkpoint model')
parser.add_argument('--pretrained_ckpt_path', type=str, default='/home/ypengbb/Respiratory-Anomaly-Detection/LSTM_track/pure_LSTM/checkpoints/PureLSTM_WTH_ts24_hs64_ln4_in11_out1_lr0.0005_bs64_ep1000/checkpoint.pth', help='whether continue to train a pre-trained and saved checkpoint model')


args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device

#### Feel Free to edit the Dataset and relevant info here!
# By default, the Target col should be the last col of the dataset
# By default, the TimeStamp col should be the first col of the dataset
data_parser = {
    'ETTh1':{'data_path':'/home/ypengbb/Respiratory-Anomaly-Detection/Common_datasets/ETT/ETTh1.csv', 'Target':'OT', 'MS':[7, 1]},
    'ETTh2':{'data_path':'/home/ypengbb/Respiratory-Anomaly-Detection/Common_datasets/ETT/ETTh2.csv', 'Target':'OT', 'MS':[7, 1]},
    'ETTm1':{'data_path':'/home/ypengbb/Respiratory-Anomaly-Detection/Common_datasets/ETT/ETTm1.csv', 'Target':'OT', 'MS':[7, 1]},
    'ETTm2':{'data_path':'/home/ypengbb/Respiratory-Anomaly-Detection/Common_datasets/ETT/ETTm2.csv', 'Target':'OT', 'MS':[7, 1]},
    'ECL':{'data_path':'/home/ypengbb/Respiratory-Anomaly-Detection/Common_datasets/ECL.csv', 'Target':'MT_320', 'MS':[321, 1]},
    'WTH':{'data_path':'/home/ypengbb/Respiratory-Anomaly-Detection/Common_datasets/WTH.csv', 'Target':'WetBulbCelsius', 'MS':[12,1]}
}   

if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data_path']
    args.target = data_info['Target']
    args.dataset_structure = data_info[args.features]  # [sensor_data_dim, target_dim]
else:   # not been added into the data_parser
    dataset_status = input('Selected dataset has not been added into the [data_parser]. Are you sure want to proceed (y)?: ')
    if dataset_status != 'y':   # Abort the program, no need to proceed with non-existing dataset
        assert False, "Program stopped because of the not found dataset!"

        
print('Args in experiment: ')
print(args)

########################### Data Prrprocess ######################################
data_preprocessor = Data_preprocessor(args=args)

X_train, X_val, X_test, y_train, y_val, y_test = data_preprocessor.generate_dataset()
train_loader, val_loader, test_loader = data_preprocessor.create_dataloader(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)
print(">>>>> Finish Data Preprocessing >>>>>")

########################### Define Model ######################################
args.input_size = X_train.shape[2]
model = LSTMModel(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.num_layers, output_size=args.output_size).to(device=args.device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

if args.pretrained_base == True:    # continue to train a saved ckpt model
    model.load_state_dict(torch.load(args.pretrained_ckpt_path, weights_only=True))
    print(">>>>> Successfully load pre-trained checkpoint model >>>>>")
else:
    print(">>>>> No pre-trained ckpt model loaded => start from a new model >>>>>")
    
print(">>>>> Finish Model Defining >>>>>")

########################### Model Training & validation ######################################
setting = f'{args.model}_{args.data}_ts{args.time_steps}_hs{args.hidden_size}_ln{args.num_layers}_in{args.input_size}_out{args.output_size}_lr{args.lr}_bs{args.batch_size}_ep{args.train_epochs}_pa{args.patience}_rand{args.random_state}'
args.setting = setting

early_stopping = EarlyStopping(args=args, patience=args.patience, verbose=True)

train_losses = []
val_losses = []

print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
with tqdm(total=args.train_epochs, desc=f'LSTM Training', leave=True) as pbar:
    for epoch in range(args.train_epochs):
        model.train()
        
        epoch_train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
    
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation loss calculation
        model.eval()
        
        epoch_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                output = model(X_batch)
                val_loss = criterion(output.squeeze(), y_batch)
                epoch_val_loss += val_loss.item()
                
        # Average validation loss
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # early stop decision & save checkpoint
        early_stopping(val_loss=avg_val_loss, model=model)
        if early_stopping.early_stop == True:
            print("Early stopping")
            break
        
        adjust_learning_rate(optimizer, epoch+1, args)
        
        # update pbar
        pbar.set_postfix({'avg_train_loss':f'{avg_train_loss:.4f}', 
                        'avg_val_loss':f'{avg_val_loss:.4f}'})
        pbar.update(1)
        
## Re-assign the best model OR load best model for inference
best_model_path = os.path.join(args.checkpoints, args.setting) + '/' + 'checkpoint.pth'
model.load_state_dict(torch.load(best_model_path, weights_only=True))


### visualization
visualizer = Visualization(args=args)
# plot the training loss curve
visualizer.loss_curve(train_losses=train_losses, val_losses=val_losses)

print('>>>>> Finish Training & Validation >>>>>')


########################### Model Test  ######################################
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    test_loss = criterion(y_pred.squeeze(), y_test)
    print(f'Test Loss: {test_loss.item():.4f}')

# Move predictions back to CPU for plotting
y_pred = y_pred.squeeze().cpu().numpy()
y_test = y_test.cpu().numpy()

### visualization
# visualize error metrics
metric_value_list = metric(y_pred, y_test)
error_types = np.array(['MAE', 'MSE', 'RMSE'])
visualizer.loss_metrics(metric_value_list=metric_value_list, error_types=error_types, preds=y_pred, trues=y_test)

# plot the good prediction rate bar plot
visualizer.good_prediction_bar_plot(y_test=y_test, y_pred=y_pred)

print(">>>>> Finish Testing >>>>>")

########################### Model Predict  ######################################
# plot the predicted samples
if args.do_predict == True: # do prediction
    visualizer.plot_pred_samples(y_test=y_test, y_pred=y_pred)
    print('>>>>> Finish Prediction >>>>>')

#################################################################
print(">>>>> All LSTM_main finish >>>>>")
