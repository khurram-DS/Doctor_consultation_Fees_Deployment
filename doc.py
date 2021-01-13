
import pandas as pd
import numpy as np

train= pd.read_excel('Final_train.xlsx')
train.drop(['Qualification','Place','Miscellaneous_Info'],inplace=True,axis=1)

from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='most_frequent')
train['Rating']=imp.fit_transform(train['Rating'].values.reshape(-1,1))

train['Experience'].replace(regex=True, inplace=True, to_replace=r'[^0-9.\-]', value=r'')
train['Experience'] = train['Experience'].astype('int')
train['Rating'].replace(regex=True, inplace=True, to_replace=r'[^0-9.\-]', value=r'')
train['Rating'] = train['Rating'].astype('int')

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
train['Profile']=le.fit_transform(train['Profile'])
train

from scipy.stats import zscore
z=abs(zscore(train))
print(train.shape)
new=train.loc[(z<3).all(axis=1)]
print(new.shape)

x=new.drop(columns=['Fees']) # Input variable.
y=pd.DataFrame(new['Fees']) #Target Variable.



from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=77,test_size=0.20)
# lets check the best parameter using grid search cv.
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')
gbr=GradientBoostingRegressor()
parameters={'learning_rate':[0.001,0.01,0.1,1],'n_estimators':[10,100,500,100]} 
# use n_estimator with step of 50
clf=GridSearchCV(gbr,parameters,cv=5)
clf.fit(x,y)
clf.best_params_
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=77,test_size=0.20)
gbr=GradientBoostingRegressor(learning_rate=0.01,n_estimators=500)
gbr.fit(x,y)
pred=gbr.predict(x_test)

import pickle
pickle.dump(gbr,open('model.pkl','wb'))


















