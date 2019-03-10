import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn import metrics
import statsmodels.api as sm

train = pd.read_csv("C:/Users/Matan/Desktop/MSA-MRM/Classes/MSA 8150 - Machine Learning for Analytics/FinalProject/data_music/Train.csv")
df = pd.DataFrame(train) 
response = pd.concat([df['latitude'],df['longitude']], axis = 1) #original coordinates array
features = df.loc[:,'Var1':'Var116']

sns.regplot(x=df['latitude'], y=df['longitude'])

f, ax = plt.subplots(figsize=(15, 12))
corr = features.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)


# Split up the data and perform simple linear regression:

x_train = features.head(767)
x_test = features.tail(192)
y_train = response.head(767)
y_test = response.tail(192)

LinearRegr = LinearRegression()
LinearRegr.fit(x_train, y_train)

y_pred = LinearRegr.predict(x_test)
y_pred = pd.DataFrame(y_pred)

true_lat = y_test['latitude']
true_lon = y_test['longitude']


pred_lat = y_pred[0]
pred_lon = y_pred[1]

# blue = true coordinates, red = predicted coordinates: we don't see much of a relationship
scat1 = plt.scatter(true_lat,true_lon, color = "blue")
scat2 = plt.scatter(pred_lat,pred_lon, color = "red")
plt.xlabel('Latitude')
plt.ylabel('Longitude')


print('Score:', LinearRegr.score(x_test, y_test))
print("Mean squared error: %.2f" % metrics.mean_squared_error(y_test,y_pred))
print("Mean absolute error: %.2f" % metrics.mean_absolute_error(y_test,y_pred))
print("Root mean squared error: %.2f" % np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
print("R-squared:", metrics.r2_score(y_test,y_pred))

