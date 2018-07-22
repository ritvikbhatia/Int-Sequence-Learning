import sklearn.ensemble
import sklearn.datasets
import sklearn.linear_model
import sklearn.tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
d=pd.read_csv("C:\\Users\\Ritvik\\Desktop\\project\\train.csv",low_memory=False) #here we open our csv file in read mode
feature_columns=[]
d=d.dropna() # To drop all the null boxes
x=[]
d=d.drop(['Id'],axis=1) # Id column is removed
for j in range(100):
    a=d.loc[j]  # This help us to iterate in a row
    b=str(a).split(',')
    del b[-1]
    b[0]=b[0].split(' ')   # These two are used to remove word "Sequence" which is printing with our series
    b[0]=b[0][4]
    b[len(b)-1]=b[len(b)-1].split('\n')  # These two lines will remove the extra part printing with the series
    b[len(b)-1]=b[len(b)-1][0]
    for i in range(0,len(b)-3):  # Here we are grouping 4-4 elements for testing and appending it in x 1 by 1
        if( len(b[i+3])<4 and '...'not in b[i+3]):
            x.append(int(b[i]))
            x.append(int(b[i+1]))
            x.append((int(b[i+2])))
            x.append(int(b[i+3]))
x=np.array(x).reshape(-1,4) #This is to divide it into 4 columns
df=pd.DataFrame(x,columns=['x1','x2','x3','y1']) #This will make a data frame with given headings
df=df.dropna()
for key in df.keys():
    if key=='y1':
        continue
    feature_columns.append(tf.contrib.layers.feature_column.real_valued_column(key))
estimator=tf.estimator.DNNRegressor(feature_columns=feature_columns,hidden_units=[10,10])
df=df.dropna()
y=df['y1']
y=np.array(y,dtype=np.int32)
df=df.drop(['y1'],axis=1)
x=df
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.2)
def train():
    return dict(x_train),y_train
def test():
    return dict(x_test),y_test

estimator.train(input_fn=train,steps=200)
ev = estimator.evaluate(input_fn=test,steps=1)
print(ev)
