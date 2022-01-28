import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from scipy import stats
import seaborn as sns
from statsmodels.formula.api import ols
import warnings
#from sklearn.model_selection import test_test_split
warnings.filterwarnings('ignore')
import pmdarima as pm
from collections import Counter
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import pacf
from sklearn.linear_model import LogisticRegression
import sklearn
from math import sin, cos, sqrt, atan2, radians
from sklearn import metrics
from matplotlib import pyplot
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
from math import sqrt

from numpy import asarray
from pandas import read_csv
from sklearn.feature_selection import RFE

from sklearn.ensemble import RandomForestRegressor
from pandas import DataFrame
import numpy as np
from datetime import timedelta
import investpy

#DATA
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from itertools import product
# Везде n - длина ряда, который должен получиться
def trans(n):   # Дрейф\переходный процесс типа сигмоиды
    x=np.linspace(-n*0.009, n*0.009, n)
    y=1-1/(1+np.exp(x))
    return y
def period(n):  # Сезонность \ периодичность
    x=np.linspace(0,16*np.pi, n)
    y=np.sin(x)/2+.5
    return y
def noise(n):   # Шум с нормальным распределением
    y=np.random.randn(n)
    y=(y-min(y))/(max(y)-min(y)) # В масштаб [0..1]
    return y
def rndwalk(n): # Случайное блуждание
    y=[0.]
    for i in range(n-1):
        k=np.random.rand()
        sign = 1. if np.random.randint(2) else -1. #x(t+1)=x(t)+-1
        y.append(y[-1]+sign*k)
    y=np.array(y)
    y=(y-min(y))/(max(y)-min(y)) # В масштаб [0..1]
    return y
def compose(n,kt,kp,kn,kr): # Собрать всё вместе с весами
    y=kt*trans(n) + kp * period(n) + kn * noise(n) + kr * rndwalk(n)
    y=(y-min(y))/(max(y)-min(y))
    return y







#create random data:
import random 

i=0
n=750
noisy=pd.DataFrame()
noisy['t']=np.arange(n)
for alpha_init in range(0,70): # по всем комбинациям
    alpha = alpha_init*(10/7)/100
    i+=1 
    y=compose(n,1-alpha,0,0,alpha)
    y=(y-min(y))/(max(y)-min(y)) # В масштаб [0..1]
    ser=pd.DataFrame({'t':np.arange(n), 'noisy':y}) # вместо t можно естественные даты/время
    '''Можно собрать в одну таблицу'''
    noisy=pd.merge(noisy, pd.DataFrame({'t':np.arange(n),str(i-1).zfill(2):y}), on='t', how='inner')
    '''Можно каждый ряд в отдельный файл'''
    #ser.to_csv('../_DataSets/Artificial/art'+str(i-1).zfill(4)+'.csv', index=False)
print('%i artificial series created'%i)
print(noisy.head())

plt.plot(noisy['05'])

noisy = noisy.reset_index(drop = True)
all_df = noisy

all_columns = (noisy.columns[1:].tolist())
all_columns.sort(key=int)



train_columns = random.sample(all_columns, 50)
train_columns.sort(key=int)

df = pd.DataFrame()
df["t"] = np.arange(n)
test = pd.DataFrame()

for i in all_columns:
    if i not in train_columns:
        test[str(i)] = noisy[str(i)]    
        print(i)
    else:
        df[str(i)] = noisy[str(i)] 

df = df.reset_index(drop = True)






#MODELS

def LSTM_tuning(X, lag = 30, layers = 2, train_ratio = 0.7):

    scaler = MinMaxScaler(feature_range=(0,1))
    x = scaler.fit_transform(np.array(X).reshape(-1,1))
    training_size = int(len(x) * train_ratio)
    train_data, test_data = x[0:training_size,:], x[training_size:len(x),:1]
    
    def create_dataset(dataset, time_step=1):
      dataX, dataY = [], []
      for i in range(len(dataset)-time_step):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
      return np.array(dataX), np.array(dataY)
    
    time_step = lag
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
    
    
    len(train_data)
    len(test_data)
    
    

    model = Sequential()
    model.add(LSTM(64*layers, input_shape=(30, 1)))
    for i in range(0,layers):
        i += 1
        model.add(Dense(64*layers/(i-0.5), activation='relu'))

    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')
    #model.summary()
    
    model.fit(X_train, y_train, epochs=100, verbose=0)
    
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    train_predict_norm = scaler.inverse_transform(train_predict)
    test_predict_norm = scaler.inverse_transform(test_predict)
    
    len(test_predict)
    len(test_data)
    
    
    def MSE(real,pred):
        return np.sum((real - pred)**2)/(len(real))


    def mean_absolute_percentage_error(y_true, y_pred): 
        y_true = y_true + 1
        y_pred = y_pred + 1

        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    def smape(act,forc):
        return 100/len(act) * np.sum(2 * np.abs(forc - act) / (np.abs(act) + np.abs(forc)))

    
    #plt.plot(test_predict,color="red")
    #plt.plot(test_data[lag:])
    #plt.show()

    mse_obs = MSE(test_data[lag:], test_predict) 
    
    mape_obs = mean_absolute_percentage_error(test_data[lag:], test_predict)

    smape_obs = smape(test_data[lag:], test_predict)

    return smape_obs, test_predict_norm






#Random forest


def random_forest(X,lag = 30, trees = 2, train_ratio = 0.7):
    def create_lags(X, lags):
        dataframe = DataFrame()
        for i in range(lags, 0, -1):
           dataframe['t-' + str(i)] = X.shift(i)
        final_data = pd.concat([X, dataframe], axis=1)
        final_data.dropna(inplace=True)
        return final_data

    final_data = create_lags(X, lag)


    finaldf = final_data.reset_index(drop=True)
    end_point = len(finaldf)*0.7
    test_length=int(len(finaldf)*(1-train_ratio))
    x = end_point - test_length

    finaldf_train = finaldf.loc[:x - 1, :]
    finaldf_test = finaldf.loc[x:, :]
    finaldf_test_x = finaldf_test.iloc[:,1:]

    finaldf_test_y = finaldf_test.iloc[:,0]
    finaldf_train_x = finaldf_train.iloc[:,1:]
    finaldf_train_y = finaldf_train.iloc[:,0]


    rfe = RandomForestRegressor(n_estimators=trees )
    fit = rfe.fit(finaldf_train_x, finaldf_train_y)
    y_pred = fit.predict(finaldf_test_x)

    #plt.plot(y_pred)
    #plt.plot(finaldf_test_y.tolist())

    def MSE(real,pred):
        return np.sum((real - pred)**2)/(len(real))


    def mean_absolute_percentage_error(y_true, y_pred): 
        y_true = y_true + 1
        y_pred = y_pred + 1

        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    def smape(act,forc):
        return 100/len(act) * np.sum(2 * np.abs(forc - act) / (np.abs(act) + np.abs(forc)))



    mse_obs = MSE(finaldf_test_y.tolist(), y_pred) 
    
    #mape_obs = mean_absolute_percentage_error(finaldf_test_y.tolist(), y_pred)

    smape_obs = smape(finaldf_test_y.tolist(), y_pred)


    return smape_obs, y_pred





#MLP

from numpy import array

def MLP_tuning(X, lag = 30, layers = 2, train_ratio = 0.7):
    
    # split a univariate sequence into samples
    def split_sequence(sequence, n_steps):
        X, y = list(), list()
        for i in range(len(sequence)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the sequence
            if end_ix > len(sequence)-1:
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return array(X), array(y)

    # define input sequence
    
    scaler = MinMaxScaler(feature_range=(0,1))
    x = scaler.fit_transform(np.array(X).reshape(-1,1))
    x = [item for sublist in x for item in sublist]

    training_size = int(len(x) * train_ratio)
    train_data, test_data = x[0:training_size], x[training_size:len(x)]

    # choose a number of time steps
    # split into samples
    X_train, y_train = split_sequence(train_data, lag)
    X_test, y_test = split_sequence(test_data, lag)
    # define model
    model = Sequential()
    model.add(Dense(64*layers, activation='relu', input_dim=lag))
    for i in range(0,layers):
        i += 1
        model.add(Dense(64*layers/(i-0.5), activation='relu'))

    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(X_train, y_train, epochs=2000, verbose=0)
    #predict
    test_predict = model.predict(X_test, verbose=0)

    plt.plot(test_predict,color="red")
    plt.plot(test_data[lag:])

    def MSE(real,pred):
        return np.sum((real - pred)**2)/(len(real))

    def mean_absolute_percentage_error(y_true, y_pred): 
        y_true = y_true + 1
        y_pred = y_pred + 1
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def smape(act,forc):
        return 100/len(act) * np.sum(2 * np.abs(forc - act) / (np.abs(act) + np.abs(forc)))

    mse_obs = MSE(test_data[lag:], test_predict) 

    #mape_obs = mean_absolute_percentage_error(y_test.tolist(), test_predict)
    smape_obs = smape(test_data[lag:], test_predict)
    return mse_obs, test_predict



#THE MODEL

def data_meta_learning(df,prediction_model="RF", levels = 5):
    #output table
    list_of_columns = df.columns.tolist()
    list_of_columns[0] = "Level"
    result = pd.DataFrame(df,columns = list_of_columns)
    #Create empty table
    for i in result.columns:
        result[str(i)] = [0.0] * len(result)

    result["Level"] = range(0,len(result))

    list_of_vars = df.columns[1:].tolist()
    for level in range(0,levels):
        print(level)

        mse_on_level = []
        if prediction_model == "RF":
            print("pobefa")
            for colname in list_of_vars:
                data = df[str(colname)]
                mse = random_forest(data, trees = (1+level)*15)[0]
                result[str(colname)][level] = mse
                mse_on_level.append(mse)
        elif prediction_model == "LSTM":
            for colname in list_of_vars:
                data = df[str(colname)]
                mse = LSTM_tuning(data, layers = 2 + level)[0]
                result[str(colname)][level] = mse
                mse_on_level.append(mse)
        elif prediction_model == "MLP":
            for colname in list_of_vars:
                data = df[str(colname)]
                mse = MLP_tuning(data, layers = 2 + level)[0]
                result[str(colname)][level] = mse
                mse_on_level.append(mse)

        np.median(mse_on_level)

        bad = [i for i,v in enumerate(mse_on_level) if v > np.median(mse_on_level)]
        list_of_vars = [list_of_vars[i] for i in bad]
        print(list_of_vars)

    quality = []
    for colname in df.columns[1:].tolist():
        data = result[str(colname)].tolist()
        #get the level
        level = len(list(filter(lambda a: a != 0.0, data)))
        quality.append(level)
    return result, quality


def data_meta_learning_quantile(df,prediction_model="RF", levels = 5):
    #output table
    list_of_columns = df.columns.tolist()
    list_of_columns[0] = "Level"
    result = pd.DataFrame(df,columns = list_of_columns)
    #Create empty table
    for i in result.columns:
        result[str(i)] = [0.0] * len(result)

    result["Level"] = range(0,len(result))

    list_of_vars = df.columns[1:].tolist()
    for level in range(0,levels):
        print(level)

        mse_on_level = []
        if prediction_model == "RF":
            print("pobefa")
            for colname in list_of_vars:
                data = df[str(colname)]
                mse = random_forest(data, trees = (1+level)*15)[0]
                result[str(colname)][level] = mse
                mse_on_level.append(mse)
        elif prediction_model == "LSTM":
            for colname in list_of_vars:
                data = df[str(colname)]
                mse = LSTM_tuning(data, layers = 2 + level)[0]
                result[str(colname)][level] = mse
                mse_on_level.append(mse)
        elif prediction_model == "MLP":
            for colname in list_of_vars:
                data = df[str(colname)]
                mse = MLP_tuning(data, layers = 2 + level)[0]
                result[str(colname)][level] = mse
                mse_on_level.append(mse)

        np.median(mse_on_level)

        bad = [i for i,v in enumerate(mse_on_level) if v > np.quantile(mse_on_level,0.3)]
        list_of_vars = [list_of_vars[i] for i in bad]
        print(list_of_vars)

    quality = []
    for colname in df.columns[1:].tolist():
        data = result[str(colname)].tolist()
        #get the level
        level = len(list(filter(lambda a: a != 0.0, data)))
        quality.append(level)
    return result, quality

np.quantile([0,1,2,3,4,5,6,7,8,9,10], 0.3)


def data_meta_learning_equal(df,prediction_model="RF", levels = 5):
    #output table
    list_of_columns = df.columns.tolist()
    list_of_columns[0] = "Level"
    result = pd.DataFrame(df,columns = list_of_columns)
    #Create empty table
    for i in result.columns:
        result[str(i)] = [0.0] * len(result)

    result["Level"] = range(0,len(result))

    list_of_vars = df.columns[1:].tolist()
    variables_num = len(list_of_vars)
    levels_num = levels
    for level in range(0,levels):
        print(level)

        mse_on_level = []
        if prediction_model == "RF":
            
            for colname in list_of_vars:
                data = df[str(colname)]
                mse = random_forest(data, trees = (1+level)*15, train_ratio= 0.65 + iter*2/100)[0]
                result[str(colname)][level] = mse
                mse_on_level.append(mse)
        elif prediction_model == "LSTM":
            for colname in list_of_vars:
                data = df[str(colname)]
                mse = LSTM_tuning(data, layers = 2 + level)[0]
                result[str(colname)][level] = mse
                mse_on_level.append(mse)
        elif prediction_model == "MLP":
            for colname in list_of_vars:
                data = df[str(colname)]
                mse = MLP_tuning(data, layers = 2 + level)[0]
                result[str(colname)][level] = mse
                mse_on_level.append(mse)

        np.median(mse_on_level)
        sorted = pd.DataFrame()
        sorted["mse"] = mse_on_level
        sorted["colnames"] = list_of_vars
        sorted = sorted.sort_values('mse')
        slicing = int((len(list_of_columns)-1)/levels_num)
        sorted['colnames'][slicing:].tolist()
        print("Slicing on : ", slicing)
        
        #bad = [i for i,v in enumerate(mse_on_level) if v in sorted[10:]]
        list_of_vars = sorted['colnames'][slicing:].tolist()
        print(len(list_of_vars))
        print(list_of_vars)

    quality = []
    for colname in df.columns[1:].tolist():
        data = result[str(colname)].tolist()
        #get the level
        level = len(list(filter(lambda a: a != 0.0, data)))
        quality.append(level)
    return result, quality

b =  np.sort([1,4,2,3])
b[3]


np.sort([1,4,2,3])[2:]



def data_meta_learning_equal_meaniter(df,prediction_model="RF", levels = 5):
    #output table
    list_of_columns = df.columns.tolist()
    list_of_columns[0] = "Level"
    result = pd.DataFrame(df,columns = list_of_columns)
    #Create empty table
    for i in result.columns:
        result[str(i)] = [0.0] * len(result)

    result["Level"] = range(0,len(result))

    list_of_vars = df.columns[1:].tolist()
    variables_num = len(list_of_vars)
    levels_num = levels
    for level in range(0,levels):
        print(level)

        mse_on_level = []
        if prediction_model == "RF":
            
            for colname in list_of_vars:
                data = df[str(colname)]
                temp_mse = []
                for iter in range (0,5):
                    mse_t = random_forest(data, trees = (1+level)*15, train_ratio= 0.65 + iter*2/100)[0]
                    temp_mse.append(mse_t)
                mse = np.mean(temp_mse)
                result[str(colname)][level] = mse
                mse_on_level.append(mse)
        elif prediction_model == "LSTM":
            for colname in list_of_vars:
                data = df[str(colname)]
                temp_mse = []
                for iter in range (0,5):
                    mse_t = LSTM_tuning(data, layers = 2 + level, train_ratio= 0.65 + iter*2/100)[0]
                    temp_mse.append(mse_t)
                mse = np.mean(temp_mse)
                result[str(colname)][level] = mse
                mse_on_level.append(mse)
        elif prediction_model == "MLP":
            for colname in list_of_vars:
                data = df[str(colname)]
                temp_mse = []
                for iter in range (0,5):
                    mse_t = MLP_tuning(data, layers = 2 + level, train_ratio= 0.65 + iter*2/100)[0]
                    temp_mse.append(mse_t)
                mse = np.mean(temp_mse)
                result[str(colname)][level] = mse
                mse_on_level.append(mse)

        np.median(mse_on_level)
        sorted = pd.DataFrame()
        sorted["mse"] = mse_on_level
        sorted["colnames"] = list_of_vars
        sorted = sorted.sort_values('mse')
        slicing = int((len(list_of_columns)-1)/levels_num)
        sorted['colnames'][slicing:].tolist()
        print("Slicing on : ", slicing)
        
        #bad = [i for i,v in enumerate(mse_on_level) if v in sorted[10:]]
        list_of_vars = sorted['colnames'][slicing:].tolist()
        print(len(list_of_vars))
        print(list_of_vars)

    quality = []
    for colname in df.columns[1:].tolist():
        data = result[str(colname)].tolist()
        #get the level
        level = len(list(filter(lambda a: a != 0.0, data)))
        quality.append(level)
    return result, quality





#RESULT

#on part of df
#result_LSTM_equal2, quality_LSTM_equal2 = data_meta_learning_equal(df,prediction_model = "LSTM",levels = 5)



#all df
result_LSTM_equal_meaniter, quality_LSTM_equal_meaniter = data_meta_learning_equal_meaniter(all_df,prediction_model = "LSTM",levels = 5)

result_LSTM_equal_meaniter.to_csv("result_LSTM_equal_meaniter.csv")
quality_LSTM_equal_meaniter_df = pd.DataFrame()
quality_LSTM_equal_meaniter_df['quality'] = quality_LSTM_equal_meaniter
quality_LSTM_equal_meaniter_df['data'] = all_df
quality_LSTM_equal_meaniter.to_csv('quality_LSTM_equal_meaniter')



from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint


#classification 


#


X_class = list()
list_of_vars = df.columns[1:].tolist()
for colname in list_of_vars:
    X_class.append(df[str(colname)].tolist())

X_class = np.array(X_class)
X_class = X_class.reshape((X_class.shape[0], X_class.shape[1], 1))
X_class.shape

def create_target(quality_LSTM):
    Y_class = np.array(quality_LSTM)
    le = LabelEncoder()
    le = le.fit(Y_class)
    labels = np.array(le.transform(Y_class))
    Y_class = labels
    Y_class.shape
    return Y_class

#choose train results
quality_LSTM_equal_meaniter_train = []
for i in range(0,len(quality_LSTM_equal_meaniter)):
    if i in list(map(int, df.columns[1:].tolist())):
        quality_LSTM_equal_meaniter_train.append(quality_LSTM_equal_meaniter[i])



Y_class = create_target(quality_LSTM_equal_meaniter_train)

num_classes = len(np.unique(Y_class))

def make_model(input_shape):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)

model = make_model(input_shape=X_class.shape[1:])



epochs = 500
batch_size = 5

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "best_model.h5", save_best_only=True, monitor="loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.5, patience=20, min_lr=0.0001
    ),
    keras.callbacks.EarlyStopping(monitor="loss", patience=50, verbose=1),
]
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)
history = model.fit(
    X_class,
    Y_class,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1,
)

from sklearn.metrics import accuracy_score



#make test 
Test_class = list()
list_of_vars = test.columns.tolist()
for colname in list_of_vars:
    Test_class.append(test[str(colname)].tolist())

Test_class = np.array(Test_class)
Test_class = Test_class.reshape((Test_class.shape[0], Test_class.shape[1], 1))
Test_class.shape

#choose test results
quality_LSTM_equal_meaniter_test = []
for i in range(0,len(quality_LSTM_equal_meaniter)):
    if i in list(map(int, test.columns.tolist())):
        quality_LSTM_equal_meaniter_test.append(quality_LSTM_equal_meaniter[i])


pred = model.predict(Test_class)
quality_pred = []
for p in pred:
    quality_pred.append(p.argmax())


quality_pred = [x+1 for x in quality_pred]

#plot

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



#all data
to_plot = pd.DataFrame()
to_plot['real_quality'] = all_df.columns[1:].tolist()
to_plot['predicted_class'] = quality_LSTM_equal_meaniter


to_plot.to_csv("all_df.csv")


to_plot_test = pd.DataFrame()
to_plot_test['real_quality'] = test.columns.tolist()
to_plot_test['predicted_class'] = quality_LSTM_equal_meaniter_test
to_plot_test['classificator_class'] = quality_pred
to_plot_test.to_csv("for_plot_test.csv")

#for plots with smape 


result_LSTM_equal_meaniter

smape = []
quality = []
level = []

for column in result_LSTM_equal_meaniter.columns[1:]:
    print(column)
    data = result_LSTM_equal_meaniter[str(column)].tolist()
    for i in range(0,10):
        smape.append(data[i])
        level.append(i)
        if data[(i+1)] == 0.0:
            print("true")
            quality.append("good")
            break
        else:
            print("false")
            quality.append("bad")


to_plot_smape = pd.DataFrame()

to_plot_smape['smape'] = smape
to_plot_smape['quality'] = quality
to_plot_smape['level'] = level

to_plot_smape.to_csv("for_plot_smape.csv")




#Multiclass classification score
from sklearn import metrics


y_test =  to_plot_test['predicted_class'] 
y_pred = to_plot_test['classificator_class'] 

#accuracy 
print(metrics.accuracy_score(y_test, y_pred))


#precision

metrics.precision_score(y_test, y_pred, average = 'micro')


#Recall
metrics.recall_score(y_test, y_pred, average = 'micro')

#F1

metrics.f1_score(y_test, y_pred, average = 'macro')


#Roc

from sklearn.metrics import roc_auc_score

def roc_auc_score_multiclass(actual_class, pred_class, average = "macro"):
    
    #creating a set of all the unique classes using the actual class list
    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:
        
        #creating a list of all the classes except the current class 
        other_class = [x for x in unique_class if x != per_class]

        #marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]

        #using the sklearn metrics method to calculate the roc_auc_score
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
        roc_auc_dict[per_class] = roc_auc

    return roc_auc_dict


roc_auc_dict = roc_auc_score_multiclass(y_test, y_pred)
roc_auc_dict


#confusion

plt.figure(figsize = (18,8))
sns.heatmap(metrics.confusion_matrix(y_test, y_pred), annot = True, xticklabels = y_test.unique(), yticklabels = y_test.unique(), cmap = 'summer')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()




















