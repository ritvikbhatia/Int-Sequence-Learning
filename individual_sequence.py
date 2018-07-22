import sklearn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.linear_model
import sklearn.ensemble
import pylab as pl
import os
x=[]
p=[]
data=pd.read_csv("C:\\Users\\varsha\\Desktop\\project\\train.csv")
data=data.fillna(0)
id=int(input('Enter Id :'))
pick=data[['Sequence']][data.Id==id]
a=str(pick).split(',')
a[0]=a[0].split(' ') #these two is use to remove word "Sequence" which is printing with our series
a[0]=a[0][-1]
print(a)
for j in range(len(a)-3):
	if('...' not in a[j+3]):
		x.append((a[j]))
		x.append((a[j+1]))
		x.append((a[j+2]))
		x.append((a[j+3]))
x=np.array(x).reshape(-1,4) #this is to divide it into 4 columns
df=pd.DataFrame(x,columns=['x1','x2','x3','y1']) #this will make a data frame with given headings
df=df.dropna()
y=df['y1']
df=df.drop(['y1'],axis=1)
x=df
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.2)
reg=sklearn.linear_model.LinearRegression()
reg.fit(x_train,y_train)
print(reg.score(x_train,y_train))
