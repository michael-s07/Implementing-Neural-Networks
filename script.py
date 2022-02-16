import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential

dataset = pd.read_csv("life_expectancy.csv")
print(dataset.head())
print(dataset.describe())

#Filtering the  data 
dataset = dataset.drop(['Country'], axis='columns')
labels = dataset.iloc[:, -1]
features = dataset.iloc[:, 0:-1]

#Data Preprocessing
samp = pd.get_dummies(features.Status)
features = pd.concat([features, samp], axis = 'columns')
features.drop(['Status'], axis = 'columns')


features_train, labels_train, features_test, labels_test = train_test_split(features, labels, test_size = 0.2)

numerical_features = features.select_dtypes(include=['float64', 'int64'])
numerical_columns = numerical_features.columns

ct = ColumnTransformer([("only numeric", StandardScaler(), numerical_columns)], remainder='passthrough')
features_train_scaled = ct.fit_transform(features_train)
features_test_scaled = ct.transform(features_train)

