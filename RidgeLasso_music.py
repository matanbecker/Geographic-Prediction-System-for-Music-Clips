import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn import metrics

##using lasso CV:
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_predict

df = pd.read_csv("C:/Users/Matan/Desktop/MSA-MRM/Classes/MSA 8150 - Machine Learning for Analytics/FinalProject/data_music/Train.csv")
df = pd.DataFrame(df) 

response = pd.concat([df['latitude'],df['longitude']], axis = 1) #original coordinates array
features = df.loc[:,'Var1':'Var116']

X = features
y = response
#lasso = linear_model.Lasso(alpha=2)
ridge = linear_model.Ridge(alpha=150)
y_pred = cross_val_predict(ridge, X, y, cv = 5) 
y_pred = pd.DataFrame(y_pred)

true_lat = response['latitude']
true_lon = response['longitude']

pred_lat = y_pred[0]
pred_lon = y_pred[1]

# blue = true coordinates, red = predicted coordinates
scat1 = plt.scatter(true_lat,true_lon, color = "blue")
scat2 = plt.scatter(pred_lat,pred_lon, color = "red")
plt.xlabel('Latitude')
plt.ylabel('Longitude')

print("R-squared: %.2f" % metrics.r2_score(response,y_pred)) 
print("MSE: %.2f" % metrics.mean_squared_error(response,y_pred))
print("Mean absolute error: %.2f" % metrics.mean_absolute_error(response,y_pred))
print("Root mean squared error: %.2f" % np.sqrt(metrics.mean_squared_error(response,y_pred)))
print("R-squared: %.2f" % metrics.r2_score(response,y_pred))