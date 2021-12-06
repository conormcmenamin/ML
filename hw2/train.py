import numpy as np
import tensorflow
import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from typing import Tuple
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std,array
from tensorflow.keras import layers
from tensorflow import keras



def import_data(filename, test=False):
    file = open(filename, 'r')
    features = []
    labels = []

    for ind, line in enumerate(file):
        tokenised_line = line.split(' ')
        data_point = []
        for i in range(len(tokenised_line)):
            data_point.append(float(tokenised_line[i]))
        features.append(data_point)

    if test:
        print('len features:' +str(len(features)))
        features=np.asarray(features).astype('float32').reshape((10000,10))
        labels=None
    else:
        features=array(features)
        features,labels=features[:,:-1],features[:,-1]
        features=np.asarray(features).astype('float32').reshape((100000,10))
        labels=np.asarray(labels).astype('float32')

    return (features,labels)

def plot_correlation_matrix(data):
    data = pd.DataFrame(train_features, columns=range(1,11))
    data['target'] = train_labels
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
    train_data=np.asarray(train_data)
    labels=np.asarray(labels)
    model.fit(train_data,labels,epochs=175)
    model.save('NNModel')
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
        valid_x=np.asarray(valid_x)
        valid_y=np.asarray(valid_y)
        eval=model.evaluate(valid_x,valid_y)
        validation_scores.append(eval)

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
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    opt=keras.optimizers.Adam(learning_rate=0.001)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model



if __name__ == '__main__':
    #=====Data Acuisition=====
    train_features,train_labels=import_data('train-io.txt',test=False)


    #=====Data inspection and preprocessing=====
    #plot_correlation_matrix(train_features)
    train_data_scaled=standardiseData(train_features)

    #=====Model Training=====
    #model = train(train_data_scaled, train_labels)
    #plot_roc(train_labels[-1000:],train_data_scaled[-1000:],model)


    #=====Model validation=====
    #results= kfoldCrossValidation(train_data_scaled,train_labels,k=8)
    # avg_acc=0
    # avg_loss=0
    # k=len(results)
    # for i in results:
    #     avg_loss += i[0]/k
    #     avg_acc+=i[1]/k
    # print('AVG LOSS: ' + str(avg_loss))
    # print('AVG ACCURACY: ' + str(avg_acc))
    # print(avg_loss,avg_acc)

    #=====Test=====

    model = keras.models.load_model('NNModel')
    test_features,test_labels=import_data('test-io.txt',test=True)

    test_o= model.predict(test_features)
    test_file=open('test-o.txt','a')
    for i in range(len(test_o)):
        pred=round(float(test_o[i]))
        test_file.write(str(pred)+'\n')

    print('Done.\n')