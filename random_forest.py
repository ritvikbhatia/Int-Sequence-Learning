import sklearn
import sklearn.ensemble
import sklearn.datasets
import sklearn.linear_model
import sklearn.tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
d=pd.read_csv("C:\\Users\\Ritvik\\Desktop\\project\\train.csv")#here we open our csv file in read mode
d=d.dropna()# To drop all the null boxes
x=[]
d=d.drop(['Id'],axis=1)# Id column is removed
for j in range(len(d)):
    a=d.loc[j]
    b=str(a).split(',')
    del b[-1]
    b[0]=b[0].split(' ')# These two are used to remove word "Sequence" which is printing with our series
    b[0]=b[0][4]
    b[len(b)-1]=b[len(b)-1].split('\n')# These two lines will remove the extra part printing with the series
    b[len(b)-1]=b[len(b)-1][0]
    for i in range(0,len(b)-3):
        if( len(b[i+3])<5 and '...'not in b[i+3]):
            x.append(int(b[i]))
            x.append(int(b[i+1]))
            x.append((int(b[i+2])))
            x.append(int(b[i+3]))
x=np.array(x).reshape(-1,4)#This is to divide it into 4 columns
df=pd.DataFrame(x,columns=['x1','x2','x3','y1'])#This will make a data frame with given headings 
df=df.dropna()
print(df)
y=df['y1']
df=df.drop(['y1'],axis=1)
x=df
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.2)
clf=sklearn.ensemble.RandomForestRegressor(random_state=0)
clf.fit(x_train,y_train)
print(clf.score(x_test,y_test))
