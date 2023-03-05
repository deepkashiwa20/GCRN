import sys
import os
import shutil
import math
import numpy as np
import pandas as pd
import scipy.sparse as ss
from datetime import datetime
import time
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchsummary import summary
import argparse
import logging
from metrics import evaluate
from utils import masked_mae, masked_mae_loss, StandardScaler, getDayTimestamp
from GCRN import GCRN

def get_xs_ys(data, mode):
    train_num = int(data.shape[0] * args.trainval_ratio)
    xs, ys = [], []
    if mode == 'train':    
        for i in range(train_num - args.horizon - args.seq_len + 1):
            x = data[i:i+args.seq_len, ...]
            y = data[i+args.seq_len:i+args.seq_len+args.horizon, ...]
            xs.append(x), ys.append(y)
    elif mode == 'test':
        for i in range(train_num - args.seq_len,  data.shape[0] - args.horizon - args.seq_len + 1):
            x = data[i:i+args.seq_len, ...]
            y = data[i+args.seq_len:i+args.seq_len+args.horizon, ...]
            xs.append(x), xs.append(y)
    xs, ys = np.array(xs), np.array(ys)
    xs, ys = xs[:, :, :, np.newaxis], ys[:, :, :, np.newaxis]
    return xs, ys

def get_xs_ys_time(data, data_time, mode):
    train_num = int(data.shape[0] * args.trainval_ratio)
    xs, ys, ys_time = [], [], []
    if mode == 'train':    
        for i in range(train_num - args.horizon - args.seq_len + 1):
            x = data[i:i+args.seq_len, ...]
            y = data[i+args.seq_len:i+args.seq_len+args.horizon, ...]
            t = data_time[i+args.seq_len:i+args.seq_len+args.horizon, ...]
            xs.append(x), ys.append(y), ys_time.append(t)
    elif mode == 'test':
        for i in range(train_num - args.seq_len,  data.shape[0] - args.horizon - args.seq_len + 1):
            x = data[i:i+args.seq_len, ...]
            y = data[i+args.seq_len:i+args.seq_len+args.horizon, ...]
            t = data_time[i+args.seq_len:i+args.seq_len+args.horizon, ...]
            xs.append(x), ys.append(y), ys_time.append(t)
    xs, ys, ys_time = np.array(xs), np.array(ys), np.array(ys_time)
    xs, ys, ys_time = xs[:, :, :, np.newaxis], ys[:, :, :, np.newaxis], ys_time[:, :, :, np.newaxis]
    return xs, ys, ys_time

def print_params(model):
    # print trainable params
    param_count = 0
    logger.info('Trainable parameter list:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape, param.numel())
            param_count += param.numel()
    logger.info(f'In total: {param_count} trainable parameters. \n')
    return

def get_model(mode):
    model = GCRN(num_nodes=args.num_nodes, input_dim=args.input_dim, output_dim=args.output_dim, horizon=args.horizon, 
                    rnn_units=args.rnn_units, num_layers=args.num_rnn_layers, cheb_k = args.max_diffusion_step,
                    cl_decay_steps=args.cl_decay_steps, use_curriculum_learning=args.use_curriculum_learning).to(device)
    if mode == 'train':
        summary(model, [(args.seq_len, args.num_nodes, args.input_dim), (args.horizon, args.num_nodes, args.output_dim)], device=device)   
        print_params(model)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
    return model

def evaluate_model(model, data_iter):
    criterion = masked_mae # or # masked_mae_loss(y_pred, y_true)
    model.eval()
    loss_sum, n, ys_pred = 0.0, 0, []
    with torch.no_grad():
        for x, y, ycov in data_iter:
            y_pred = model(x, ycov)
            y_pred = scaler.inverse_transform(y_pred)
            loss = criterion(y_pred, y)
            loss_sum += loss.item() * y.shape[0]
            n += y.shape[0]
            ys_pred.append(y_pred.cpu().numpy())     
    loss = loss_sum / n
    ys_pred = np.vstack(ys_pred)
    return loss, ys_pred

def train_model(name, mode, xs, ys, ycov):
    model = get_model(mode)
    
    xs = scaler.transform(xs)
    xs_torch, ys_torch, ycov_torch = torch.Tensor(xs).to(device), torch.Tensor(ys).to(device), torch.Tensor(ycov).to(device)
    trainval_data = torch.utils.data.TensorDataset(xs_torch, ys_torch, ycov_torch)
    trainval_size = len(trainval_data)
    train_size = int(trainval_size * (1 - args.val_ratio))
    train_data = torch.utils.data.Subset(trainval_data, list(range(0, train_size)))
    val_data = torch.utils.data.Subset(trainval_data, list(range(train_size, trainval_size)))
    train_iter = torch.utils.data.DataLoader(train_data, args.batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_data, args.batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=args.epsilon)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.steps, gamma=args.lr_decay_ratio)
    
    criterion = masked_mae # or # masked_mae_loss(y_pred, y_true)
        
    min_val_loss = np.inf
    wait = 0   
    batches_seen = 0
    for epoch in range(args.epochs):
        starttime = datetime.now()     
        loss_sum, n = 0.0, 0
        loss_sum1, loss_sum2, loss_sum3 = 0.0, 0.0, 0.0
        model.train()
        for x, y, ycov in train_iter:
            optimizer.zero_grad()
            y_pred = model(x, ycov, y, batches_seen)
            y_pred = scaler.inverse_transform(y_pred)
            loss = criterion(y_pred, y)
            loss_sum += loss.item() * y.shape[0]
            n += y.shape[0]
            batches_seen += 1
            loss.backward()
            optimizer.step()
        lr_scheduler.step()    
        train_loss = loss_sum / n
        val_loss, _ = evaluate_model(model, val_iter)
        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            torch.save(model.state_dict(), modelpt_path)
        else:
            wait += 1
            if wait == args.patience:
                logger.info('Early stopping at epoch: %d' % epoch)
                break
        endtime = datetime.now()
        epoch_time = (endtime - starttime).seconds
        logger.info("epoch", epoch, "time used:", epoch_time, "seconds", 
                    "train loss:", '%.6f' % train_loss, 
                    "validation loss:", '%.6f' % val_loss, 
                    "lr:", '%.6f' % optimizer.param_groups[0]['lr'])
        with open(epochlog_path, 'a') as f:
            f.write("%s, %d, %s, %d, %s, %s, %.6f, %s, %.6f\n" % ("epoch", epoch, "time used", epoch_time, "seconds", "train loss", train_loss, "validation loss:", val_loss))
    
def test_model(name, mode, xs, ys, ycov):
    model = get_model(mode)
    model.load_state_dict(torch.load(modelpt_path))
    
    xs = scaler.transform(xs)
    xs_torch, ys_torch, ycov_torch = torch.Tensor(xs).to(device), torch.Tensor(ys).to(device), torch.Tensor(ycov).to(device)
    test_data = torch.utils.data.TensorDataset(xs_torch, ys_torch, ycov_torch)
    test_iter = torch.utils.data.DataLoader(test_data, args.batch_size, shuffle=False)
    loss, ys_pred = evaluate_model(model, test_iter)
    logger.info('ys.shape, ys_pred.shape,', ys.shape, ys_pred.shape)
    ys, ys_pred = np.squeeze(ys), np.squeeze(ys_pred)
    # np.save(path + f'/{name}_prediction.npy', ys_pred)
    # np.save(path + f'/{name}_groundtruth.npy', ys)
    MSE, RMSE, MAE, MAPE = evaluate(ys, ys_pred)
    logger.info("%s, %s, test loss, %.6f" % (name, mode, loss))
    logger.info("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.6f, %.6f, %.6f, %.6f" % (name, mode, MSE, RMSE, MAE, MAPE))
    with open(score_path, 'a') as f:
        f.write("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.6f, %.6f, %.6f, %.6f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
        for i in range(args.horizon):
            MSE, RMSE, MAE, MAPE = evaluate(ys[:, i, :], ys_pred[:, i, :])
            logger.info("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.6f, %.6f, %.6f, %.6f" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
            f.write("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.6f, %.6f, %.6f, %.6f\n" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
        
#########################################################################################    
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['METRLA', 'PEMSBAY'], default='METRLA', help='which dataset to run')
parser.add_argument('--trainval_ratio', type=float, default=0.8, help='the ratio of training and validation data among the total')
parser.add_argument('--val_ratio', type=float, default=0.125, help='the ratio of validation data among the trainval ratio')
parser.add_argument('--num_nodes', type=int, default=207, help='num_nodes')
parser.add_argument('--seq_len', type=int, default=12, help='input sequence length')
parser.add_argument('--horizon', type=int, default=12, help='output sequence length')
parser.add_argument('--input_dim', type=int, default=1, help='number of input channel')
parser.add_argument('--output_dim', type=int, default=1, help='number of output channel')
parser.add_argument('--max_diffusion_step', type=int, default=3, help='max diffusion step or Cheb K')
parser.add_argument('--num_rnn_layers', type=int, default=1, help='number of rnn layers')
parser.add_argument('--rnn_units', type=int, default=64, help='number of rnn units')
parser.add_argument("--loss", type=str, default='mask_mae_loss', help="mask_mae_loss")
parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--patience", type=int, default=20, help="patience used for early stop")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.01, help="base learning rate") # can be further fine-tuned
parser.add_argument("--steps", type=eval, default=[50, 100], help="steps") # can be further fine-tuned
parser.add_argument("--lr_decay_ratio", type=float, default=0.1, help="lr_decay_ratio")
parser.add_argument("--epsilon", type=float, default=1e-3, help="optimizer epsilon")
parser.add_argument("--use_curriculum_learning", type=eval, choices=[True, False], default='True', help="use_curriculum_learning")
parser.add_argument("--cl_decay_steps", type=int, default=2000, help="cl_decay_steps")
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
# parser.add_argument('--seed', type=int, default=100, help='random seed.')
args = parser.parse_args()

if args.dataset == 'METRLA':
    data_path = f'../{args.dataset}/metr-la.h5'
    args.num_nodes = 207
elif args.dataset == 'PEMSBAY':
    data_path = f'../{args.dataset}/pems-bay.h5'
    args.num_nodes = 325
else:
    pass # including more datasets in the future    

model_name = 'GCRN'
timestring = time.strftime('%Y%m%d%H%M%S', time.localtime())
path = f'../save/{args.dataset}_{model_name}_{timestring}'
logging_path = f'{path}/{model_name}_{timestring}_logging.txt'
score_path = f'{path}/{model_name}_{timestring}_scores.txt'
epochlog_path = f'{path}/{model_name}_{timestring}_epochlog.txt'
modelpt_path = f'{path}/{model_name}_{timestring}.pt'
if not os.path.exists(path): os.makedirs(path)
shutil.copy2(sys.argv[0], path)
shutil.copy2(f'{model_name}.py', path)
shutil.copy2('utils.py', path)
shutil.copy2('metrics.py', path)
    
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
class MyFormatter(logging.Formatter):
    def format(self, record):
        spliter = ' '
        record.msg = str(record.msg) + spliter + spliter.join(map(str, record.args))
        record.args = tuple() # set empty to args
        return super().format(record)
formatter = MyFormatter()
handler = logging.FileHandler(logging_path, mode='a')
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(console)

logger.info('model', model_name)
logger.info('dataset', args.dataset)
logger.info('trainval_ratio', args.trainval_ratio)
logger.info('val_ratio', args.val_ratio)
logger.info('num_nodes', args.num_nodes)
logger.info('seq_len', args.seq_len)
logger.info('horizon', args.horizon)
logger.info('input_dim', args.input_dim)
logger.info('output_dim', args.output_dim)
logger.info('num_rnn_layers', args.num_rnn_layers)
logger.info('rnn_units', args.rnn_units)
logger.info('max_diffusion_step', args.max_diffusion_step)
logger.info('loss', args.loss)
logger.info('batch_size', args.batch_size)
logger.info('epochs', args.epochs)
logger.info('patience', args.patience)
logger.info('lr', args.lr)
logger.info('epsilon', args.epsilon)
logger.info('steps', args.steps)
logger.info('lr_decay_ratio', args.lr_decay_ratio)
logger.info('use_curriculum_learning', args.use_curriculum_learning)

cpu_num = 1
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)
device = torch.device("cuda:{}".format(args.gpu)) if torch.cuda.is_available() else torch.device("cpu")
# Please comment the following three lines for running experiments multiple times.
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
#####################################################################################################

data = pd.read_hdf(data_path).values
data_time = getDayTimestamp(pd.read_hdf(data_path))
mean = np.mean(data[:int(data.shape[0]*args.trainval_ratio)])
std = np.std(data[:int(data.shape[0]*args.trainval_ratio)])
scaler = StandardScaler(mean, std)

def main():
    logger.info(args.dataset, 'training started', time.ctime())
    train_xs, train_ys, train_ycov = get_xs_ys_time(data, data_time, 'train')
    logger.info('Train xs.shape ys.shape, ycov.shape', train_xs.shape, train_ys.shape, train_ycov.shape)
    train_model(model_name, 'train', train_xs, train_ys, train_ycov)
    logger.info(args.dataset, 'training ended', time.ctime())
    
    test_xs, test_ys, test_ycov = get_xs_ys_time(data, data_time, 'test')
    logger.info('Test xs.shape, ys.shape, ycov.shape', test_xs.shape, test_ys.shape, test_ycov.shape)
    test_model(model_name, 'test', test_xs, test_ys, test_ycov)
    logger.info(args.dataset, 'testing ended', time.ctime())
    logger.info('=' * 90)

if __name__ == '__main__':
    main()
