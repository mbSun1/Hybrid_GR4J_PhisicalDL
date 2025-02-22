import pandas as pd
import torch
import torch.optim as optim
from pathlib import Path
from utils.data import get_loader
import os
import argparse
from create_model import create_model
from torchinfo import summary
from utils.loss import nse, nse_loss, mse_loss
from utils.early_stopping import EarlyStopping
from utils.weight_constraint import weightConstraint
import time
import random
import torch.backends.cudnn as cudnn
import numpy as np
from train_model import train_model

if __name__ == '__main__':

    def hybrid_main(basin_id,choose_model_type):
        parser = argparse.ArgumentParser()
        parser.add_argument('--task', type=str, default='PRNN')
        parser.add_argument('--gpu', action='store_true', default=True, help='Use gpu to accelerate')
        parser.add_argument('--data_dir', type=str, default=f'data/{basin_id}.csv', help='path of input data')
        parser.add_argument('--basin_id', type=str, default=basin_id,
                            help='Choose which basin to simulate from sheetName')
        parser.add_argument('--model_type', type=str, default=choose_model_type, help='hybrid-J')
        parser.add_argument('--loss_fn', type=str, default='NSE', help='choose a loss function to train model: NSE or MSE')
        parser.add_argument('--sequence_length', type=int, default=1825,
                            help='2190,It can be other values, but recommend not be less than 5 years (1825 days)')
        parser.add_argument('--window_step', type=int, default=365, help='the step of moving window to generate sequences')
        parser.add_argument('--warm_up_dayl', type=int, default=365, help='the length to warm up model')
        parser.add_argument('--annual_loss_weight', type=float, default=0, help='weight of annual streamflow loss')
        # parser.add_argument('--baseflow_loss_weight', type=float, default=0, help='weight of baseflow loss')
        parser.add_argument('--epochs', type=int, default=50)
        parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
        parser.add_argument('--clip', type=float, default=1, help='gradient clip to avoid explosion')
        parser.add_argument('--bs', type=int, default=5, help='batch size')
        parser.add_argument('--hs_rnn', type=int, default=64, help='hidden size for lstm')
        parser.add_argument('--es', type=int, default=4, help='embedding size for timestamp')
        parser.add_argument('--hs_cnn', type=int, default=25, help='hidden size for the first Conv1D layer')
        parser.add_argument('--patience', type=int, default=30, help='patience of early stopping')
        parser.add_argument('--ks', type=int, default=30, help='kernel size for the first Conv1D layer')
        parser.add_argument('--seed', type=int, default=2024)
        args = parser.parse_args()

        # fix random seed
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

        device = torch.device('cuda:0' if args.gpu and torch.cuda.is_available() else 'cpu')

        # load dataset from the given path
        hydrodata = pd.read_csv(args.data_dir, index_col='date', parse_dates=True)

        # Split data set into training, and test sets
        training_start, training_end = '1980-01-01', '2003-12-31'
        testing_start, testing_end = '2004-01-01', '2009-12-31'




        train_set = hydrodata[hydrodata.index.isin(pd.date_range(training_start, training_end))]
        test_set = hydrodata[hydrodata.index.isin(pd.date_range(testing_start, testing_end))]
        dataset = get_loader(train_set, test_set, sequence_length=args.sequence_length, batch_size=args.bs,
                             window_step=args.window_step, spin_up_dayl=args.warm_up_dayl)
        train_loader, test_loader = dataset.train_loader, dataset.test_loader
        # configure model
        model = create_model(input_dim=dataset.train_set.x.shape[2], hs_cnn=args.hs_cnn, hs_rnn=args.hs_rnn, es=args.es,
                             model_type=args.model_type, ks=args.ks, device=device)
        model = model.double().to(device)
        summary(model)

        loss_dict = {'NSE': nse_loss, 'MSE': mse_loss}
        criterion = loss_dict[args.loss_fn]()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)
        # apply to constraint hydrological parameters into range 0~1
        constraints = weightConstraint()

        # save path for best model and training loss curve
        now = time.strftime('%Y%m%d-%H%M', time.localtime())

        save_dir = f'./checkpoints/{args.basin_id}'
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        # dump_fname = f'{args.model_type}_lr={args.lr}_bs={args.bs}_w1={args.annual_loss_weight}_w2=' \
        #              f'{args.baseflow_loss_weight}_hs_rnn={args.hs_rnn}_es={args.es}_hs_cnn={args.hs_cnn}_ks={args.ks}_' \
        #              f'seed={args.seed}_time={now}.pt'
        dump_fname = f'{args.model_type}_lr={args.lr}_bs={args.bs}_w1={args.annual_loss_weight}' \
                     f'_hs_rnn={args.hs_rnn}_es={args.es}_hs_cnn={args.hs_cnn}_ks={args.ks}_' \
                     f'seed={args.seed}_time={now}.pt'
        dump_path = os.path.join(save_dir, dump_fname)

        # early stopping for training
        early_stopping = EarlyStopping(dump_path, patience=args.patience)
        train_model(model, train_loader, test_loader, criterion, args.epochs, optimizer, scheduler, device,
                    args.warm_up_dayl, early_stopping, constraints, args.model_type, clip=args.clip, basin_id=args.basin_id,
                    batch_first=True)



    # Loop through the calculation method of a single site
    working_path = './camels'  # The folder path where the camels data is indexed
    # List of basin IDs
    basin_list = pd.read_csv(os.path.join(working_path, 'basin_list.txt'),
                             sep='\t', header=0, dtype={'HUC': str, 'BASIN_ID': str})
    basin_ids = basin_list['BASIN_ID']
    choose_model_type = 'hybrid-J'
    cal_counter = 1
    for basin_id in basin_ids:
        hybrid_main(basin_id,choose_model_type)
        cal_counter += 1
        print('basin #{} has been successfully cal.'.format(basin_id))
        print('\n' * 3)
    print("All sites have been simulated!")

