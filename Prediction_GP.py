import numpy as np
import pandas as pd
from datetime import datetime
import GPy
from pep.PEP_reg import PEP
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
#get Data
dates = pd.read_csv('data_raw_dates.csv', parse_dates=["data_raw_dates"])
dates['data_raw_dates'] = pd.to_datetime(dates.data_raw_dates).dt.tz_localize(None)
full_data = pd.read_csv("Data_clean.csv")
#preprocessing for modelling
y = full_data["RecRate"]
X = full_data.iloc[:,1:192]
np.random.seed(22)
#########################
#Fixed Window 2001-2012##
#########################
def fixed_pred(M,alpha,lengthscale,variance,lik_noise_var):
    # train on dates before 2012
    ind_training = dates.data_raw_dates < datetime.fromisoformat("2012-01-01")
    ind_test = dates.data_raw_dates >= datetime.fromisoformat("2012-01-01")
    X_fixed_train = np.c_[X[ind_training].values]
    X_fixed_test = np.c_[X[ind_test].values]
    y_fixed_train = np.c_[y[ind_training].values]
    y_fixed_test = y[ind_test].values
    # fit the Model
    np.random.seed(22)
    a = np.random.randint(0, np.shape(X_fixed_train)[0], size=M)
    Z = np.c_[X[ind_training].iloc[a].values]
    k = GPy.kern.RBF(input_dim=2, lengthscale=lengthscale, variance=variance)
    model = GPy.models.SparseGPRegression(X_fixed_train, y_fixed_train, kernel=k, Z=Z)  # was soll Z??
    model.name = 'POWER-EP'
    model.inference_method = PEP(alpha=alpha)
    model.Gaussian_noise.variance = lik_noise_var
    model.unfix()
    model.checkgrad()
    model.optimize('bfgs', messages=True, max_iters=2e3)
    #predict and evaluate the prediction
    (m, V) = model.predict(X_fixed_test, full_cov=False)
    mse_fixed = mean_squared_error(m,y_fixed_test)
    mae_fixed = mean_absolute_error(m,y_fixed_test)
    return [m,y_fixed_test, mse_fixed,mae_fixed]

##########################
# rolling expanding window with two year prediction
##########################
def rolling_expanding_2y(M,alpha,lengthscale,variance,lik_noise_var):
     date_stop = [datetime.fromisoformat("2012-01-01"),datetime.fromisoformat("2013-01-01"),datetime.fromisoformat("2014-01-01"),datetime.fromisoformat("2015-01-01")]
     date_2year = [datetime.fromisoformat("2014-01-01"),datetime.fromisoformat("2015-01-01"),datetime.fromisoformat("2016-01-01"),datetime.fromisoformat("2017-01-01")]
     tupels_exp = list(zip(date_stop,date_2year))
     predictions_exp_mean = []
     y_test_exp = []
     for d in tupels_exp:
         #train on an expanding window predict 2 years per window
         ind_training = dates.data_raw_dates < d[0]
         ind_test = dates.data_raw_dates.between(d[0],d[1])
         X_exp_train = np.c_[X[ind_training].values]
         X_exp_test = np.c_[X[ind_test].values]
         y_exp_train = np.c_[y[ind_training].values]
         y_exp_test = np.c_[y[ind_test].values]
         #fitting the model
         np.random.seed(22)
         a = np.random.randint(0, np.shape(X_exp_train)[0], size=M)
         Z = np.c_[X[ind_training].iloc[a].values]
         k = GPy.kern.RBF(input_dim=2, lengthscale=lengthscale, variance=variance)
         model_exp = GPy.models.SparseGPRegression(X_exp_train, y_exp_train, kernel=k, Z = Z)  # was soll Z??
         model_exp.name = 'POWER-EP'
         model_exp.inference_method = PEP(alpha=alpha)
         model_exp.Gaussian_noise.variance = lik_noise_var
         model_exp.unfix()
         model_exp.checkgrad()
         model_exp.optimize('bfgs', messages=True, max_iters=2e3)
         #predict and evaluate
         (m, V) = model_exp.predict(X_exp_test, full_cov=False)
         predictions_exp_mean.append(m)
         y_test_exp.append(y_exp_test)
     predictions_exp_mean = np.concatenate(predictions_exp_mean).ravel().tolist()
     y_exp_test = np.concatenate(y_test_exp).ravel().tolist()
     mse_exp = mean_squared_error(predictions_exp_mean,y_exp_test)
     mae_exp = mean_absolute_error(predictions_exp_mean, y_exp_test)
     return([predictions_exp_mean,y_exp_test,mse_exp,mae_exp])
##############################
# rolling 2 years ahead
############################
def rolling_2y(M,alpha,lengthscale,variance,lik_noise_var):
    date_stop = [datetime.fromisoformat("2012-01-01"), datetime.fromisoformat("2013-01-01"),datetime.fromisoformat("2014-01-01"), datetime.fromisoformat("2015-01-01")]
    date_2year = [datetime.fromisoformat("2014-01-01"), datetime.fromisoformat("2015-01-01"),datetime.fromisoformat("2016-01-01"), datetime.fromisoformat("2017-01-01")]
    dates_start = [datetime.fromisoformat("2001-01-01"), datetime.fromisoformat("2002-01-01"),datetime.fromisoformat("2003-01-01"), datetime.fromisoformat("2004-01-01")]
    tupels_rolling = list(zip(date_stop, date_2year, dates_start))
    predictions_roll_mean = []
    y_test_roll = []
    for d in tupels_rolling:
        # train on an rolling window predict on an 2 year rolling window
        ind_training = dates.data_raw_dates.between(d[2],d[1])
        ind_test = dates.data_raw_dates.between(d[0],d[1])
        X_roll_train = np.c_[X[ind_training].values]
        X_roll_test = np.c_[X[ind_test].values]
        y_roll_train = np.c_[y[ind_training].values]
        y_roll_test = np.c_[y[ind_test].values]
        #fitting the model
        np.random.seed(22)
        a = np.random.randint(0, np.shape(X_roll_train)[0], size=M)
        Z = np.c_[X[ind_training].iloc[a].values]
        # model just like in test_PEP
        k = GPy.kern.RBF(input_dim=2, lengthscale=lengthscale, variance=variance)
        model_roll = GPy.models.SparseGPRegression(X_roll_train, y_roll_train, kernel=k, Z = Z)  # was soll Z??
        model_roll.name = 'POWER-EP'
        model_roll.inference_method = PEP(alpha=alpha)
        model_roll.Gaussian_noise.variance = lik_noise_var
        model_roll.unfix()
        model_roll.checkgrad()
        model_roll.optimize('bfgs', messages=True, max_iters=20e3)
        # predict and evaluate
        (m, V) = model_roll.predict(X_roll_test, full_cov=False)
        predictions_roll_mean.append(m)
        y_test_roll.append(y_roll_test)
    predictions_roll_mean = np.concatenate(predictions_roll_mean).ravel().tolist()
    y_test_roll = np.concatenate(y_test_roll).ravel().tolist()
    mse_roll = mean_squared_error(predictions_roll_mean,y_test_roll)
    mae_roll = mean_absolute_error(predictions_roll_mean, y_test_roll)
    return([predictions_roll_mean,y_test_roll,mse_roll,mae_roll])
#########################
#Daily Rolling window
########################
def pred_daily(M,alpha,lengthscale,variance,lik_noise_var):
    df = pd.DataFrame()
    df["days_to_predict"] = np.unique(dates[dates['data_raw_dates'] >= datetime.fromisoformat("2012-01-01")].values)
    df["days_to_start"] = df["days_to_predict"] - pd.DateOffset(years=11)
    dtp = df["days_to_predict"].values
    dts = df["days_to_start"].values
    tup = list(zip(dts,dtp))
    daily_mean = []
    daily_y_test = []
    for t in tup:
        #predict every day after 2012 training the model with every date up to that point
         ind_training_daily = dates.data_raw_dates.between(t[0],t[1],inclusive = "left")
         print(ind_training_daily)
         ind_test_daily = (dates.data_raw_dates == t[1])
         X_d_train = np.c_[X[ind_training_daily].values]
         X_d_test = np.c_[X[ind_test_daily].values]
         y_d_train = np.c_[y[ind_training_daily].values]
         y_d_test = np.c_[y[ind_test_daily].values]
         #fitting the model
         np.random.seed(22)
         a = np.random.randint(0, np.shape(X_d_train)[0], size=M)
         Z = np.c_[X[ind_training_daily].iloc[a].values]
         k = GPy.kern.RBF(input_dim=2, lengthscale=lengthscale, variance=variance)
         model_d = GPy.models.SparseGPRegression(X_d_train, y_d_train, kernel=k,Z =Z)  # was soll Z??
         model_d.name = 'POWER-EP'
         model_d.inference_method = PEP(alpha=alpha)
         model_d.Gaussian_noise.variance = lik_noise_var
         model_d.unfix()
         model_d.checkgrad()
         model_d.optimize('bfgs', messages=True, max_iters=2e3)
         #predict an evaluate
         (m, V) = model_d.predict(X_d_test, full_cov=False)
         daily_mean.append(m)
         daily_y_test.append(y_d_test)
    daily_mean = np.concatenate(daily_mean).ravel().tolist()
    daily_y_test = np.concatenate(daily_y_test).ravel().tolist()
    mse_daily = mean_squared_error(daily_mean,daily_y_test)
    mae_daily = mean_absolute_error(daily_mean, daily_y_test)
    return ([daily_mean,daily_y_test,mse_daily,mae_daily])

#-----------------------
df_fixed = pd.DataFrame()
df_fixed["predictions"] = fixed_pred(1,0.25,1,1,0.2)[0].ravel()
df_fixed["real_values"] = fixed_pred(1,0.25,1,1,0.2)[1].ravel()
df_fixed.to_csv("Fixed_Predictions.csv",index = False)

df_exp = pd.DataFrame()
df_exp["predictions"] = rolling_expanding_2y(10,0.25,1,1,0.2)[0]
df_exp["real_values"] = rolling_expanding_2y(10,0.25,1,1,0.2)[1]
df_exp.to_csv("exp_Predictions.csv",index = False)

df_roll = pd.DataFrame()
df_roll["predictions"] = rolling_2y(50,0.5,1,1,0.2)[0]
df_roll["real_values"] = rolling_2y(50,0.5,1,1,0.2)[1]
df_exp.to_csv("rolling_Predictions.csv",index = False)

df_days = pd.DataFrame()
df_days["predictions"] = rolling_2y(10,0.5,1,1,0.2)[0]
df_days["real_values"] = rolling_2y(10,0.5,1,1,0.2)[1]
df_days.to_csv("daily_Predictions.csv",index = False)


#good results:
# reasonable parameters for fixed prediction : 1,0.25,1,1,0.2 -> MSE:751.6776508215119 RMSE: 27.41674033909779 MAE: 21.735596099783013
# reasonable parameters for expanding 2 year : 10,0.25,1,1,0.2 -> MSE:1079.6699444973995 RMSE: 32.85833143203409 MAE: 23.587870196951197
# reasonable parameters for rolling 2 year : 50,0.5,1,1,0.2 -> MSE:676.5983433610182 RMSE: 26.01150405803206 MAE: 21.01157624787808
# reasonable parameters for daily : 10,0.5,1,1,0.2 -> MSE: 765.55377 RMSE:27.668642370625935 MAE:21.06305633925444
