import sklearn
import sklearn.ensemble
import sklearn.datasets
import sklearn.linear_model
import sklearn.tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#training
d=pd.read_csv("C:\\Users\\Ritvik\\Desktop\\project\\train.csv")
d=d.dropna()
x=[]
d=d.drop(['Id'],axis=1)
for j in range(len(d)):
    a=d.loc[j]
    b=str(a).split(',')
    del b[-1]
    b[0]=b[0].split(' ')
    b[0]=b[0][4]
    b[len(b)-1]=b[len(b)-1].split('\n')
    b[len(b)-1]=b[len(b)-1][0]
    for i in range(0,len(b)-3):
        if( len(b[i+3])<5 and '...'not in b[i+3]):
            x.append(int(b[i]))
            x.append(int(b[i+1]))
            x.append((int(b[i+2])))
            x.append(int(b[i+3]))
x=np.array(x).reshape(-1,4)
df=pd.DataFrame(x,columns=['x1','x2','x3','y1'])
df=df.dropna()
y=df['y1']
df=df.drop(['y1'],axis=1)
x=df
#testing
d=pd.read_csv("C:\\Users\\Ritvik\\Desktop\\project\\test.csv")
d=d.dropna()
x1=[]
d=d.drop(['Id'],axis=1)
for j in range(len(d)):
    a=d.loc[j]
    b=str(a).split(',')
    del b[-1]
    b[0]=b[0].split(' ')
    b[0]=b[0][4]
    b[len(b)-1]=b[len(b)-1].split('\n')
    b[len(b)-1]=b[len(b)-1][0]
    for i in range(0,len(b)-3):
        if( len(b[i+3])<5 and '...'not in b[i+3]):
            x1.append(int(b[i]))
            x1.append(int(b[i+1]))
            x1.append((int(b[i+2])))
            x1.append(int(b[i+3]))
x1=np.array(x1).reshape(-1,4)
df=pd.DataFrame(x1,columns=['x1','x2','x3','y1'])
df=df.dropna()
print(df)
y1=df['y1']
df=df.drop(['y1'],axis=1)
x1=df
clf=sklearn.ensemble.RandomForestRegressor(random_state=0)
clf.fit(x,y)
print(clf.score(x1,y1))
