# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 06:14:42 2019

@author: Bharathraj C L
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder 
import json
from sklearn.metrics import r2_score
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

lb = LabelEncoder()

col_name = ["carat", "cut", "color", "clarity", "depth", "table", "price", "x", "y","z"]
cut_data = ["Fair", "Good", "Ideal", "Premium", "Very Good"]
color_data = ["J", "I", "H", "D", "F", "G", "E","K","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","E"]
clarity_data = ["VS2", "IF", "SI2", "VVS2", "VVS1", "VS1", "SI1", "I1","FL","I2","I3"]


config_data = 'C:/Users/Bharathraj C L/Projects/diamonds/config.json'
with open(config_data) as json_data:
    mo_data = json.load(json_data)



data = pd.read_csv(mo_data["train_path"])
act_col_name = list(data.columns)

if('Unnamed: 0' in [i for i in act_col_name]):
    data = data.set_index('Unnamed: 0')


data = data.drop_duplicates()


for x in col_name:
    if(x not in act_col_name):
        print('feature ',x,' missing in the training data document')

cut_lb = lb.fit(cut_data)
print(type(data['cut']))
data['cut'] = cut_lb.transform(data['cut'])
color_lb = lb.fit(color_data)
data['color'] = color_lb.transform(data['color'])
clarity_lb = lb.fit(clarity_data)
data['clarity'] = clarity_lb.transform(data['clarity'])

price = data['price']

del data['price']



x_train,x_test,y_train,y_test = train_test_split(data,price, test_size = 0.2)


ex = ExtraTreesRegressor(n_estimators = 100)
model = ex.fit(x_train,y_train)


print(r2_score(model.predict(x_train),y_train))
print(r2_score(model.predict(x_test),y_test))



filename = 'finalized_model.sav'
joblib.dump(model, filename)


