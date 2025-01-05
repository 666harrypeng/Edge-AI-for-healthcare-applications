import matplotlib.pyplot as plt
import numpy as np
import os

class Visualization:
    def __init__(self, args):
        self.args = args
    
    def loss_curve(self, train_losses, val_losses): # Plot and save the loss curve
        assert len(train_losses) == len(val_losses)
        num_epochs = len(train_losses)
        
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(num_epochs), train_losses, label='Training Loss', color='#8E8BFE')
        plt.plot(np.arange(num_epochs), val_losses, label='Validation Loss', color='#F27970')
        plt.title('Training and Validation Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')  # Add legend in the upper right corner
        plt.tight_layout()
        path = os.path.join(self.args.checkpoints, self.args.setting)
        if not os.path.exists(path):
            os.makedirs(path)
        train_curve_path = path + '/' + 'training_loss_curve.png'
        plt.savefig(train_curve_path)    # save the loss curve as a PNG image
        print(f'Training curve has been saved at: {train_curve_path}')
        # plt.show()
    
    def loss_metrics(self, metric_value_list, error_types, preds, trues):
        mae, mse, rmse = metric_value_list
        print(f'MSE:{mse}, MAE:{mae}, RMSE:{rmse}')
        
        path = os.path.join(self.args.checkpoints, self.args.setting)
        if not os.path.exists(path):
            os.makedirs(path)
        metrics_path = path + '/' + 'error_metrics.txt'
        
        with open(metrics_path, 'w') as f:
            f.write(f'Mean Squared Error (MSE): {mse:.5f}\n')
            f.write(f'Mean Absolute Error (MAE): {mae:.5f}\n')
            f.write(f'Root Mean Squared Error (RMSE): {rmse:.5f}\n')
        print(f"\nError Metrics have been saved at {metrics_path}")
        
        np.save(path+'/'+'preds.npy', preds)
        np.save(path+'/'+'trues.npy', trues)
        print(f"\nPreds & Trues have been saved at {path}")
        
        
        
    def good_prediction_bar_plot(self, y_test, y_pred):
        ZERO_EPSILON = 1e-6
        LOW_ERROR_BOUND = 0.05
        UP_ERROR_BOUND = 0.35
        
        error_margin_list = np.round(np.linspace(start=LOW_ERROR_BOUND, stop=UP_ERROR_BOUND, num=10), decimals=3)
        good_pred_rate_list = []
        y_test = y_test + (y_test < ZERO_EPSILON) * ZERO_EPSILON
        y_pred = y_pred + (y_pred < ZERO_EPSILON) * ZERO_EPSILON
        relative_error = np.abs(y_test - y_pred) / y_test

        for error_margin in error_margin_list:
            good_prediction_rate = np.sum(relative_error < error_margin) / len(y_test)
            good_pred_rate_list.append(good_prediction_rate)
            print(f'error bound {error_margin}: good prediction rate: {good_prediction_rate}')
        
    
        good_pred_rate_list = np.array(good_pred_rate_list)

        plt.figure(figsize=(10, 8))
        plt.bar(error_margin_list, good_pred_rate_list, width=0.01, color='skyblue')

        # Adding labels and title
        plt.xlabel('Error Margin')
        plt.ylabel('Good Prediction Rate')
        plt.title(f'Error Margin & Good Prediction Rate\n{self.args.setting}')
        plt.xticks(error_margin_list)
        plt.ylim(0, 1)

        # Adding numerical values on top of bars
        for i in range(len(error_margin_list)):
            plt.text(error_margin_list[i], good_pred_rate_list[i] + 0.01, 
                    f'{good_pred_rate_list[i]:.3f}', ha='center')

        path = os.path.join(self.args.checkpoints, self.args.setting)
        if not os.path.exists(path):
            os.makedirs(path)
        good_pred_rate_path = path + '/' + 'lstm_good_pred_rate.png'
        plt.savefig(good_pred_rate_path)
        print(f'Good prediction rate bar plot has been saved at {good_pred_rate_path}')
        # plt.show()

    def plot_pred_samples(self, y_test, y_pred):
        PRED_SAMPLES = self.args.pred_sample_num
        plt.figure(figsize=(12, 6))
        plt.plot(y_test[:PRED_SAMPLES], label='Actual', color='red',linestyle='--', marker='o')
        plt.plot(y_pred[:PRED_SAMPLES], label='Prediction', color='blue', linestyle='--', marker='x')
        plt.title(f'Actual vs Predicted (First {PRED_SAMPLES} Samples)')
        plt.xlabel('Sample')
        plt.ylabel('Predicted Values')
        plt.legend()
        path = os.path.join(self.args.checkpoints, self.args.setting)
        if not os.path.exists(path):
            os.makedirs(path)
        pred_sample_path = path + '/' + 'lstm_prediction.png'
        plt.savefig(pred_sample_path)
        print(f'Predicted samples have been saved at {pred_sample_path}')
        # plt.show()
        


