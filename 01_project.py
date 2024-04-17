import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
            ## lets load the bost house pricing dataset
from sklearn.datasets import fetch_california_housing
            ##this is a dataset not a data frame
calafonia=fetch_california_housing()
print(calafonia.keys())
            ##check the description
print(calafonia.DESCR)
print(calafonia.data)
print(calafonia.target)
print(calafonia.feature_names)

            ##prepare the dataset
            ##convert the calofonia datas set to dataframe
dataset=pd.DataFrame(calafonia.data,columns=calafonia.feature_names)
            # print(dataset) #columns feature gives the headers to the dataframe
print(dataset.head(5))
            #the output price is in target so create a column in dataset to map the price
dataset['Price']=calafonia.target
print(dataset.head(5))
dataset.info()
            ##summarizing the stats of the data
print(dataset.describe())
            ## check the missing values
print(dataset.isnull().sum())
            ##Exploratory data Analysis
            ##Correlation
print(dataset.corr())
sns.pairplot(dataset)
plt.show()
plt.scatter(dataset['MedInc'],dataset['Price'])
plt.xlabel("Medical Insurance")
plt.ylabel("Price")
plt.show()

plt.scatter(dataset['Population'],dataset['Price'])
plt.xlabel("Population")
plt.ylabel("Price")
plt.show()



sns.regplot(x="Population",y="Price",data=dataset)
plt.show()
sns.regplot(x="AveRooms",y="Price",data=dataset)
plt.show()
sns.regplot(x="AveBedrms",y="Price",data=dataset)
plt.show()
sns.regplot(x="HouseAge",y="Price",data=dataset)
plt.show()

#plt.show()

            #Indipendent and Dependent features
X=dataset.iloc[:,:-1]   #take all columns expept price
y=dataset.iloc[:,-1]    #take price column

print(X.head(4))

            ##Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)



            ##Standartizind daraset
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


print(X_test)
print(X_train)

import pickle
pickle.dump(scaler,open('scaling.pkl','wb'))

from sklearn.linear_model import LinearRegression

regression=LinearRegression()

regression.fit(X_train,y_train)

            ##pRINT THE COEFFIIENTS AND THE INTERCEPT

print(regression.coef_)
print(regression.intercept_)

            ##on which parameters the modles has been trained

print(regression.get_params())

            ##prediction with test data

reg_pred=regression.predict(X_test)

print(reg_pred)

            ##plot a scatter plot
plt.scatter(y_test,reg_pred)


#plt.show()


## Residuals(errors)
residual=y_test-reg_pred

print(residual)

##plot the residuals
sns.displot(residual,kind="kde")
#plt.show()

##Scatter plot with respect to prediction and residuals
plt.scatter(reg_pred,residual)

#plt.show()

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test,reg_pred))
print(mean_squared_error(y_test,reg_pred))
print(np.sqrt(mean_squared_error(y_test,reg_pred)))

            ## R square and adjusted R square
            ##R^2=1-ssr/sst
from sklearn.metrics import r2_score
score=r2_score(y_test,reg_pred)
print(score)

            ##display adjusted R-square
score_1=1-(1-score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
print(score_1)

                ##New data Prediction
print(calafonia.data[0])
print(calafonia.data[0].shape)
print(calafonia.data[0].reshape(1,-1))
firstdata=calafonia.data[0].reshape(1,-1)
            ##transfo rmation of new data
scaler.transform(firstdata)

print(regression.predict(scaler.transform(firstdata)))


##import pickle
import pickle
pickle.dump(regression,open('regmodel.pkl','wb'))
pickled_model=pickle.load(open('regmodel.pkl','rb'))
print(pickled_model.predict(scaler.transform(firstdata)))