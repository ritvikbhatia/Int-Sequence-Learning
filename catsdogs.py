#import tensorflow as tf
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
os.chdir(ori)
dir=os.listdir()
for i in range(0,20000,5):
    file=dir[i]
    f=os.fsdecode(file)
    img=Image.open(file)
    img=img.convert('L')
    img = img.resize((8, 8), PIL.Image.ANTIALIAS)
    img=np.array(img)
    x.append(img)
    y.append(f.split('.')[0])
x = np.array(x).reshape(-1,64)
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.2)
clf=sklearn.ensemble.RandomForestClassifier(random_state=0)
clf.fit(x_train,y_train)
print(clf.score(x_test,y_test))
