# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 08:46:43 2019

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
cut_lb = lb.fit(cut_data)
color_lb = lb.fit(color_data)
clarity_lb = lb.fit(clarity_data)




config_data = 'C:/Users/Bharathraj C L/Projects/diamonds/config.json'
with open(config_data) as json_data:
    mo_data = json.load(json_data)
    
    
test_data = mo_data["test_data"]

for i in test_data:
    if(test_data[i] == ' '):
        print("Please provide the data for feature : ", i)

cut_lb = lb.fit(cut_data)
cut_act_data = cut_lb.transform([test_data['cut']])[0]
color_lb = lb.fit(color_data)
color_act_data = color_lb.transform([test_data['color']])[0]
clarity_lb = lb.fit(clarity_data)
clarity_act_data = clarity_lb.transform([test_data['clarity']])[0]

test_act = [test_data['carat'],cut_act_data,color_act_data,clarity_act_data,test_data['depth'],test_data['table'],test_data['x'],test_data['y'],test_data['z']]


filename = 'finalized_model.sav'
loaded_model = joblib.load(filename)

result = loaded_model.predict([test_act])[0]
