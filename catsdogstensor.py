import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn
import sklearn.ensemble
import sklearn.linear_model
import os
import sklearn.datasets
import sklearn.model_selection
import pylab as pl
from PIL import Image
import PIL
import matplotlib.pyplot as plt
ori="C:\\Users\\Ritvik\\Desktop\\train"
x=[]
y=[]
k=[]
feature_columns=[]

os.chdir(ori)
dir=os.listdir()
for i in range(0,200,50):
    file=dir[i]
    f=os.fsdecode(file)
    img=Image.open(file)
    img=img.convert('L')
    img = img.resize((8, 8), PIL.Image.ANTIALIAS)

    img=np.array(img)
    for j in range(len(img)):
        for i in range(len(img)):
            img[i][j]=int(img[i][j])
    x.append(img)
    y.append(f.split('.')[0])
x = np.array(x).reshape(-1,64)
for i in range(64):
    k.append(str(i))
x=pd.DataFrame(x,columns=k)
x=x.dropna()
for i in range(len(y)):
    if(y[i]=='cat'):
        y[i]==0
    else:
        y[i]==1
for key in x.keys():
    feature_columns.append(tf.contrib.layers.feature_column.real_valued_column(key))
print(x)
estimator=tf.estimator.DNNClassifier(feature_columns=feature_columns,hidden_units=[10],n_classes=2)
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.2)
def train():
    return dict(x_train),y_train
def test():
    return dict(x_test),y_test
estimator.train(input_fn=train,steps=200)
r=estimator.evaluate(input_fn=test,steps=1)
print(r)
