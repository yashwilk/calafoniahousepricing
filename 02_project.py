import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
            ## lets load the bost house pricing dataset
from sklearn.datasets import load_iris
            ##this is a dataset not a data frame
iris=load_iris()
print(iris.keys())
            ##check the description
print(iris.DESCR)
print(iris.data)
print(iris.target)
print(iris.feature_names)

            ##prepare the dataset
            ##convert the calofonia datas set to dataframe
dataset=pd.DataFrame(iris.data,columns=iris.feature_names)
            # print(dataset) #columns feature gives the headers to the dataframe
print(dataset.head(5))
            #the output price is in target so create a column in dataset to map the price
dataset['Price']=iris.target
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
plt.scatter(dataset['sepal length (cm)'],dataset['Price'])
plt.xlabel("sepal length (cm)")
plt.ylabel("Price")
plt.show()

plt.scatter(dataset['sepal width (cm)'],dataset['Price'])
plt.xlabel("sepal width (cm)")
plt.ylabel("Price")
plt.show()


sns.regplot(x="sepal length (cm)",y="Price",data=dataset)
plt.show()
sns.regplot(x="sepal width (cm)",y="Price",data=dataset)
plt.show()
sns.regplot(x="petal length (cm)",y="Price",data=dataset)
plt.show()
sns.regplot(x="petal width (cm)",y="Price",data=dataset)


plt.show()

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


# print(X_test)
# print(X_train)

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


plt.show()


## Residuals(errors)
residual=y_test-reg_pred

print(residual)

##plot the residuals
sns.displot(residual,kind="kde")
plt.show()

##Scatter plot with respect to prediction and residuals
plt.scatter(reg_pred,residual)

plt.show()

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test,reg_pred))
print(mean_squared_error(y_test,reg_pred))
print(np.sqrt(mean_squared_error(y_test,reg_pred)))

## R square and adjusted R square
