import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.multioutput import MultiOutputRegressor
from scipy import stats
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/Matan/Desktop/MSA-MRM/Classes/MSA 8150 - Machine Learning for Analytics/FinalProject/data_music/Train.csv")
df = pd.DataFrame(df) 
df = df[(np.abs(stats.zscore(df[[col for col in df.columns if col not in ['latitude','longitude']]])) < 4).all(axis=1)]

X = df[[col for col in df.columns if col not in ['latitude','longitude']]]
y = pd.concat([df['latitude'],df['longitude']], axis = 1) #original coordinates array

scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X) 

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)

def train_model(X,y,p1,p2):
        model = MultiOutputRegressor(RandomForestRegressor(n_estimators = p1, max_features= p2, min_samples_split = 5, max_depth = 30, n_jobs =2, random_state=2))
        kf = KFold(n_splits=5)
        kf.get_n_splits(X)
        scores = []
        for train_index, test_index in kf.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                model.fit(X_train,y_train)
                scores.append(metrics.mean_squared_error(model.predict(X_test),y_test))
        return np.mean(scores)

results = pd.DataFrame()
for p1 in np.arange(300,450,50):
    for p2 in np.arange(30,60,20):
        mse = train_model(X,y,p1,p2)
        results = results.append({'mse':mse,'p1':p1, 'p2':p2},ignore_index=True)
        print("P1:", p1, "P2:", p2, "MSE:", mse)

print(results.sort_values('mse'))

#results.to_csv("C:/Users/Matan/Desktop/RF_music.csv")

plt.scatter(results['p2'],results['mse'], c=results['p1'])
#plt.scatter(results['p1'],results['mse'], c='blue')
plt.show()



def scale_mean_var(data,scalar_vars):
        scaler = preprocessing.StandardScaler()
        for feat in scalar_vars:
                data[feat] = scaler.fit_transform(data[[feat]])
        return data
test = pd.read_csv('Test.csv')
test = scale_mean_var(test,[col for col in test.columns if 'Var' in col])
p = model.predict(test)
pred = pd.DataFrame(p,columns=['latitude','longitude'])
#pred.to_csv('music_prediction.csv')


plt.scatter(test['latitude'],test['longitude'], color = "blue")
plt.scatter(pred['latitude'],pred['longitude'], color = "red")
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.show()