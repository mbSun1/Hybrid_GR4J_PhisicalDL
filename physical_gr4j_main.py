import os
import spotpy
import torch
from sceua_gr4j.SPOT import gr4j_spot_setup
import pandas as pd
import numpy as np
from sceua_gr4j.gr4j import GR4J
from utils.evaluationMetrics import calculate_r,\
    calculate_nse, calculate_mse, calculate_mae, calculate_nrmse,\
    calculate_mape, calculate_rmse, calculate_pbias, calculate_kge, calculate_tpe

if __name__ == "__main__":
    def physical_main(basin_id):

        ##——---------------------------------------------
        # # 1.Prepare data
        fileplace = f'data/{basin_id}.csv'
        # Read data from the folder
        hydrodata = pd.read_csv(fileplace, index_col='date', parse_dates=True)
        # Select data for specific date range
        training_start, training_end = '1980-01-01', '2003-12-31'
        testing_start, testing_end = '2004-01-01', '2009-12-31'
        training_data = hydrodata.loc[training_start:training_end]
        testing_data = hydrodata.loc[testing_start:testing_end]

        # Extract the required columns and convert them to tensors
        prec_train = torch.tensor(training_data['prcp(mm/day)'].values)
        tmean_train = torch.tensor(training_data['tmean(C)'].values)
        dayl_train = torch.tensor(training_data['dayl(day)'].values)
        qobs_train = torch.tensor(training_data['flow(mm)'].values)

        prec_test = torch.tensor(testing_data['prcp(mm/day)'].values)
        tmean_test = torch.tensor(testing_data['tmean(C)'].values)
        dayl_test = torch.tensor(testing_data['dayl(day)'].values)
        qobs_test = torch.tensor(testing_data['flow(mm)'].values)

        # Potential evapotranspiration
        pet_train = 29.8 * (dayl_train * 24) * 0.611 * torch.exp(17.3 * tmean_train / (tmean_train + 237.3)) / (
                    tmean_train + 273.2)
        pet_test = 29.8 * (dayl_test * 24) * 0.611 * torch.exp(17.3 * tmean_test / (tmean_test + 237.3)) / (
                    tmean_test + 273.2)

        # --------------------------------------------------
        # 2.Calibrate parameters
        # Initialize GR4J SPOT setup
        spot_setup = gr4j_spot_setup(prec_train, pet_train, qobs_train)
        # Select the maximum number of repetitions allowed for SCEUA calibration
        rep = 50
        sampler = spotpy.algorithms.sceua(spot_setup, dbname="SCEUA_gr4j", dbformat="csv")
        # Start the sampler
        sampler.sample(rep, ngs=7, kstop=3, peps=0.1, pcento=0.1)
        # load_csv_results
        results = spotpy.analyser.load_csv_results("SCEUA_gr4j")
        # Find the run ID with the minimum objective function value
        bestindex, rmse = spotpy.analyser.get_minlikeindex(results)
        # Select the best model run
        best_model_run = results[bestindex]

        # Print the parameter settings of the best model run
        print("rmse", rmse)
        print("best hydrological parameter value")

        best_params = []  # Create a list to store parameter values

        for field in best_model_run.dtype.names:
            if field.startswith("par"):
                param_value = best_model_run[field]
                best_params.append(param_value)

        # Print the best parameter value
        for param_value in best_params:
            print(param_value)

        # --------------------------------------------------------
        #  3. Enter the parameters for simulation
        model = GR4J()
        # Initialize parameters
        # x1, x2, x3, x4 = model.init_params()
        best_x1, best_x2, best_x3, best_x4 = best_params[0], best_params[1], best_params[2], best_params[3]
        # update_params
        model.update_params(best_x1, best_x2, best_x3, best_x4)
        # The predicted value needs to be of type list (it is a standard)
        # The unit of qsim is mm
        qsim_train, _, _ = model.run(prec_train, pet_train)
        qsim_test, _, _ = model.run(prec_test, pet_test)

        # calculate statistical indicators
        obs_flow_train = np.array(qobs_train)
        sim_flow_train = np.array(qsim_train)
        obs_flow_test = np.array(qobs_test)
        sim_flow_test = np.array(qsim_test)

        # training period
        obs_array_train= np.array(obs_flow_train)
        sim_array_train = np.array(sim_flow_train)
        R_train = calculate_r(obs_array_train, sim_array_train)
        NSE_train = calculate_nse(obs_array_train, sim_array_train)
        MSE_train = calculate_mse(obs_array_train, sim_array_train)
        RMSE_train= calculate_rmse(obs_array_train, sim_array_train)
        MAE_train= calculate_mae(obs_array_train, sim_array_train)
        NRMSE_train = calculate_nrmse(obs_array_train, sim_array_train)
        MAPE_train = calculate_mape(obs_array_train, sim_array_train)
        PBIAS_train = calculate_pbias(obs_array_train, sim_array_train)
        KGE_train = calculate_kge(obs_array_train, sim_array_train)
        TPE_train = calculate_tpe(obs_array_train, sim_array_train)
        # testing period
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
        TPE_test = calculate_tpe(obs_array_test, sim_array_test)



        # Save data to a CSV file, with data for each site saved to a CSV file named after the site
        sim_obs_filename = f"{basin_id}.csv"
        output_filepath = os.path.join(f'result/sceua-gr4j', sim_obs_filename)
        sim_obs_test = pd.DataFrame({'Obs_test(mm)': obs_flow_test.tolist(), 'Sim_test(mm)': sim_flow_test.tolist()})
        sim_obs_train = pd.DataFrame({'Obs_train(mm)': obs_flow_train.tolist(), 'Sim_train(mm)': sim_flow_train.tolist()})
        # Create a data frame containing evaluation indicators for the test period
        metrics_data_train = pd.DataFrame({'R_train': [R_train], 'NSE_train': [NSE_train],  'MSE_train': [MSE_train], 'RMSE_test': [RMSE_train],
                                    'MAE_train': [MAE_train], 'NRMSE_train': [NRMSE_train], 'MAPE_train': [MAPE_train],
                                    'PBIAS_train': [PBIAS_train], 'KGE_train': [KGE_train], 'TPE_train': [TPE_train]})

        # Create a data frame containing evaluation indicators for the test period
        metrics_data_test = pd.DataFrame(
            {'R_test': [R_test], 'NSE_test': [NSE_test], 'MSE_test': [MSE_test], 'RMS_testE': [RMSE_test],
             'MAE_test': [MAE_test], 'NRMSE_test': [NRMSE_test], 'MAPE_test': [MAPE_test],
             'PBIAS_test': [PBIAS_test], 'KGE_test': [KGE_test], 'TPE_test': [TPE_test]})
        # concatenate two data frames
        outputdata = pd.concat([sim_obs_test, metrics_data_test,sim_obs_train, metrics_data_train], axis=1)
        outputdata.to_csv(output_filepath, index=False)
        print('The data in basin #{} has been successfully processed.'.format(basin_id))



    #Loop through the calculation method of a single site
    working_path = './camels'  # The folder path where the camels data is indexed
    # List of basin IDs
    basin_list = pd.read_csv(os.path.join(working_path, 'basin_list.txt'),
                             sep='\t', header=0, dtype={'HUC': str, 'BASIN_ID': str})
    basin_ids = basin_list['BASIN_ID']
    cal_counter = 1
    for basin_id in basin_ids:
        physical_main(basin_id)
        cal_counter += 1
        print('basin #{} has been successfully cal.'.format(basin_id))
        print('\n' * 3)
    print("All sites have been simulated!")

