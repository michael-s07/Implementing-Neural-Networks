import pandas as pd

dataset = pd.read_csv("life_expectancy.csv")
print(dataset.head())
print(dataset.describe())

#Filtering the  data 
dataset = dataset.drop(['Country'], axis='columns')
labels = dataset.iloc[:, -1]
features = dataset.iloc[:, 0:-1]

