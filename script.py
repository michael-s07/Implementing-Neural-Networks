import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.optimizers import Adam
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

my_model = Sequential()
input = InputLayer(input_shape=(features.shape[1], ))
my_model.add(input)
my_model.add(Dense(77, activation = 'relu'))
my_model.add(Dense(1))
print(my_model.summary())

#Optimizing model 
opt  = Adam(learning_rate=0.01)
my_model.compile(loss='mse', metrics=['mae'], optimizer=opt)

# Training Model 
my_model.fit(features_train_scaled, labels_train, epochs=40, batch_size=1, verbose=1)
res_mse, res_mae =my_model.evaluate(features_test_scaled, labels_test, verbose=0)
print(res_mae)
print(res_mse)