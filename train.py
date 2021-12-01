import numpy as np
import tensorflow
import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
from tensorflow.keras import layers
from tensorflow import keras



train_file = open('train-io.txt', 'r')
train_data_in = []
train_data_out = []

for ind, line in enumerate(train_file):
    tokenised_line = line.split(' ')
    data_point = []
    for i in range(len(tokenised_line)-1):
        if not i in []:                                     #Features to be eliminated
            data_point.append(float(tokenised_line[i]))
    train_data_in.append(data_point)
    curr_label =float(tokenised_line[10])
    train_data_out.append(curr_label)




train_data_in=np.asarray(train_data_in).astype('float32').reshape((100000,10))
train_data_out=np.asarray(train_data_out).astype('float32')

def plot_correlation_matrix(data):
    data = pd.DataFrame(train_data_in, columns=range(1,11))
    data['target'] = train_data_out
    data.head()
    corr=data.corr()
    mask = np.triu(np.ones_like(corr,dtype=np.bool))
    sns.set_style(style='white')
    f, ax = plt.subplots(figsize=(11,11))
    cmap=sns.diverging_palette(11,250,as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, square=True, ax=ax)
    plt.show()

def standardiseData(matrix):
    scaler=StandardScaler()
    return scaler.fit_transform(matrix)

def minmaxData(matrix):
    scaler=MinMaxScaler()
    return scaler.fit_transform(matrix)

def train(train_data,labels):
    model = get_model()
    model.fit(train_data,labels,epochs=175)
    return model

def kfoldCrossValidation(X,y,k=8):
    num_validation_samples=len(X) // k
    validation_scores=[]
    for fold in range(k):
        valid_x=X[num_validation_samples*fold: num_validation_samples*(fold+1)]
        valid_y=y[num_validation_samples*fold: num_validation_samples*(fold+1)]
        train_x=np.concatenate((X[:num_validation_samples*fold] , X[num_validation_samples*(fold+1):]))
        train_y=np.concatenate((y[:num_validation_samples*fold] , y[num_validation_samples*(fold+1):]))
        model=get_model()
        model.fit(train_x,train_y, epochs=175)
        validation_scores.append(model.evaluate(valid_x,valid_y))
    return validation_scores

def plot_roc(labels,data, model):
    
    predictions=model.predict(data)
    fp,tp,_ =roc_curve(labels,predictions)
    plt.plot(100*fp, 100*tp)
    plt.xlabel("False Positive Rate [%]")
    plt.ylabel("True Positive Rate [%]")
    plt.show()


def get_model():
    model = keras.Sequential()
    model.add(layers.Dense(64, input_dim=10, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    opt=keras.optimizers.Adam(learning_rate=0.0008)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model



if __name__ == '__main__':
    #Data preprocessing
    train_data_scaled=standardiseData(train_data_in)
    #plot_correlation_matrix(train_data_scaled)
    #TODO: Model Training
    model = train(train_data_scaled[:-1000], train_data_out[:-1000])
    plot_roc(train_data_out[-1000:],train_data_scaled[-1000:],model)
    #Model validation
    #results= kfoldCrossValidation(train_data_scaled,train_data_out)



    avg_acc=0
    avg_loss=0
    k=len(results)
    for i in results:
        avg_loss += i[0]/k
        avg_acc+=i[1]/k
    print('AVG LOSS: ' + str(avg_loss))
    print('AVG ACCURACY: ' + str(avg_acc))
