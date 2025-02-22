import torch
import pandas as pd
import os
from utils.loss import nse, nse_loss, mse_loss
import numpy as np
from utils.evaluationMetrics import calculate_r,\
    calculate_nse, calculate_mse, calculate_mae, calculate_nrmse,\
    calculate_mape, calculate_rmse, calculate_pbias, calculate_kge, calculate_tpe

def train_model(model, train_loader, test_loader, criterion, epochs, optimizer, scheduler, device,
                warm_up_dayl, early_stopping, constraints, model_type, clip, basin_id, batch_first=True):
    def train(model, train_loader, criterion, optimizer, scheduler, constraints, model_type,clip):
        model.train()
        total_loss = 0.0
        total_daily_nse = 0.0
        hidden = None
        for input, target in train_loader:
            if model_type in ['hybrid-J', 'hybrid-Z', 'physical']:
                model.PRNNLayer.apply(constraints)  # constraint hydrological parameters into range 0~1
            input, target = input.to(device), target.to(device)
            hidden = tuple(v.clone().detach() for v in hidden) if hidden is not None else None
            output, hidden = model(input, hidden)
            loss = criterion(y_pred=output, y_true=target, skip_index=warm_up_dayl, batch_first=batch_first)
            nse_daily_value = nse(y_true=target, y_pred=output, skip_index=warm_up_dayl)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            total_loss += loss.item()
            total_daily_nse += nse_daily_value.item()

        epoch_loss = total_loss / len(train_loader)
        epoch_daily_nse = total_daily_nse / len(train_loader)
        if scheduler is not None:
            scheduler.step(epoch_loss)
        return epoch_loss, epoch_daily_nse

    def test(model, test_loader, criterion):
        model.eval()
        total_loss = 0.0
        total_daily_nse = 0.0
        hidden = None
        for input, target in test_loader:
            input, target = input.to(device), target.to(device)
            hidden = tuple(v.clone().detach() for v in hidden) if hidden is not None else None
            output, hidden = model(input, hidden)
            loss = criterion(y_pred=output, y_true=target, skip_index=warm_up_dayl, batch_first=batch_first)
            nse_daily_value = nse(y_true=target, y_pred=output, skip_index=warm_up_dayl)
            total_loss += loss.item()
            total_daily_nse += nse_daily_value.item()


        epoch_loss = total_loss / len(test_loader)
        epoch_daily_nse = total_daily_nse / len(test_loader)

        return epoch_loss, epoch_daily_nse

    best_epoch = 0
    best_nse = float('-inf')

    # sim_flow_train = []
    # obs_flow_train = []
    sim_flow_test = []
    obs_flow_test = []
    for epoch in range(epochs):
        print('epoch:{:d}/{:d}'.format(epoch, epochs))
        print('*' * 100)
        train_loss, train_daily_nse = train(model, train_loader, criterion, optimizer, scheduler, constraints, model_type, clip)
        print('training: loss {:.4f}, daily nse {:.4f}'.format(train_loss, train_daily_nse))
        test_loss, test_daily_nse = test(model, test_loader, criterion)
        print('testing: loss {:.4f}, daily nse {:.4f}'.format(test_loss, test_daily_nse))
        early_stopping(test_daily_nse, model, mode='max')
        if early_stopping.early_stop:
            print(f'Early stopping with best loss: {early_stopping.test_loss_best: .4f}')
            break

        if test_daily_nse > best_nse:
            best_nse = test_daily_nse
            best_epoch = epoch
            # Save simulated and observed flows
            sim_flow_test = []
            obs_flow_test = []
            with torch.no_grad():
                model.eval()
                for input, target in test_loader:
                    input, target = input.to(device), target.to(device)
                    output, _ = model(input, None)
                    sim_flow_test.extend(output.squeeze().cpu().numpy())
                    obs_flow_test.extend(target.squeeze().cpu().numpy())
            # 保存训练数据的模拟和观测流量
            sim_flow_train = []
            obs_flow_train = []
            # with torch.no_grad():
            #     model.eval()
            #     for input1, target1 in train_loader:
            #         input1, target1 = input1.to(device), target1.to(device)
            #         output1, _ = model(input1, None)
            #         sim_flow_train.extend(output1.view(-1).cpu().numpy())
            #         obs_flow_train.extend(target1.view(-1).cpu().numpy())


    #训练期指标
    # obs_array_train= np.array(obs_flow_train)
    # sim_array_train = np.array(sim_flow_train)
    # R_train = calculate_r(obs_array_train, sim_array_train)
    # NSE_train = calculate_nse(obs_array_train, sim_array_train)
    # MSE_train = calculate_mse(obs_array_train, sim_array_train)
    # RMSE_train= calculate_rmse(obs_array_train, sim_array_train)
    # MAE_train= calculate_mae(obs_array_train, sim_array_train)
    # NRMSE_train = calculate_nrmse(obs_array_train, sim_array_train)
    # MAPE_train = calculate_mape(obs_array_train, sim_array_train)
    # PBIAS_train = calculate_pbias(obs_array_train, sim_array_train)
    # KGE_train = calculate_kge(obs_array_train, sim_array_train)


    #测试期计算指标
    obs_array_test = np.array(obs_flow_test)
    sim_array_test = np.array(sim_flow_test)
    R_test = calculate_r(obs_array_test, sim_array_test)
    NSE_test = calculate_nse(obs_array_test, sim_array_test)
    MSE_test = calculate_mse(obs_array_test, sim_array_test)
    RMSE_test = calculate_rmse(obs_array_test, sim_array_test)
    MAE_test = calculate_mae(obs_array_test, sim_array_test)
    NRMSE_test = calculate_nrmse(obs_array_test, sim_array_test)
    MAPE_test = calculate_mape(obs_array_test, sim_array_test)
    PBIAS_test = calculate_pbias(obs_array_test, sim_array_test)
    KGE_test = calculate_kge(obs_array_test, sim_array_test)


    # 保存数据到 CSV 文件中，每个站点的数据保存到以站点命名的 CSV 文件中
    sim_obs_filename = f"{basin_id}.csv"
    output_filepath = os.path.join(f'result/{model_type}', sim_obs_filename)
    sim_obs_test = pd.DataFrame({'Obs_test(mm)': obs_flow_test, 'Sim_test(mm)': sim_flow_test})
    # 创建测试期包含评价指标的数据框
    metrics_data_test = pd.DataFrame({'R_test': [R_test], 'NSE_test': [NSE_test],  'MSE_test': [MSE_test], 'RMS_testE': [RMSE_test],
                                'MAE_test': [MAE_test], 'NRMSE_test': [NRMSE_test], 'MAPE_test': [MAPE_test],
                                'PBIAS_test': [PBIAS_test], 'KGE_test': [KGE_test]})

    # sim_obs_train = pd.DataFrame({'Obs_train(mm)': obs_flow_train, 'Sim_train(mm)': sim_flow_train})
    # # 创建测试期包含评价指标的数据框
    # metrics_data_train = pd.DataFrame({'R_train': [R_train], 'NSE_train': [NSE_train],  'MSE_train': [MSE_train], 'RMSE_test': [RMSE_train],
    #                             'MAE_train': [MAE_train], 'NRMSE_train': [NRMSE_train], 'MAPE_train': [MAPE_train],
    #                             'PBIAS_train': [PBIAS_train], 'KGE_train': [KGE_train]})
    # 合并两个数据框
    # outputdata = pd.concat([sim_obs_test, metrics_data_test,sim_obs_train, metrics_data_train], axis=1)
    outputdata = pd.concat([sim_obs_test, metrics_data_test], axis=1)
    outputdata.to_csv(output_filepath, index=False)



