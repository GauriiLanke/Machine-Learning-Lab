#Assignment 0

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date

#Reading Dataset
data=pd.read_csv("used_cars_data.csv")

#Analyzing Data
print(data.head(10))
print(data.tail())
print(data.info())
print(data.nunique())
print(data.isnull().sum())
print((data.isnull().sum()/(len(data)))*100)

#Data Reduction
data=data.drop(['S.No.'],axis=1)
print(data.info())

#Splitting
data['Brand'] = data.Name.str.split().str.get(0)
data['Model'] = data.Name.str.split().str.get(1) + data.Name.str.split().str.get(2)
print(data[['Name','Brand','Model']])

#Creating Feature
data['Car_age']=date.today().year-data['Year']
print(data.head(10))
print(data.info())

data['Parsing Renewal']=data['Car_age'].apply(lambda age:'Yes' if age >= 15 else 'No')
print(data.head())
print(data.info())

brandCounts = data['Brand'].value_counts()
data['Brand_Popularity'] = data['Brand'].map(brandCounts)
print(data.head())

#Data Cleaning
print(data.Brand.unique())
print(data.Brand.nunique())

searchfor = ['Isuzu' ,'ISUZU','Mini','Land']
data[data.Brand.str.contains('|'.join(searchfor))].head(5)
data["Brand"].replace({"ISUZU": "Isuzu", "Mini": "Mini Cooper","Land":"Land Rover"}, inplace=True)
print(data.head(20))

#Statistics Summary
print(data.describe().T)
print()
print(data.describe(include='all').T)

print()
cat_cols=data.select_dtypes(include=['object']).columns
num_cols = data.select_dtypes(include=np.number).columns.tolist()
print("Categorical Variables:")
print(cat_cols)
print("Numerical Variables:")
print(num_cols)
