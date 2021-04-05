import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score 
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import datetime as dt
#-----------------------------------------------------------------------------------------------------------------------
train_data=pd.read_csv("train.csv")
test_data=pd.read_csv("test.csv")

#轉換屋齡 (銷售年分-重建日期)*12
houseage=(train_data['YrSold']-train_data['YearRemodAdd'])*12
train_data.insert(75, 'houseage', houseage)

#經過分析 取出相關係數>0.5之變量
連續型變量=['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'YearBuilt',
       'GarageArea', 'FullBath', 'TotalBsmtSF', 'GarageYrBlt', '1stFlrSF',
       'YearRemodAdd', 'TotRmsAbvGrd', 'Fireplaces', 'houseage']
類別型變量=['Foundation','BsmtQual','KitchenQual','ExterQual']

類別型變量_df=pd.get_dummies(train_data[類別型變量])
連續型變量_df=train_data[連續型變量]
train_df=pd.concat([連續型變量_df,類別型變量_df],axis=1)

#"GarageYrBlt" columns 有 81個 null值
print(" null值:",train_df['GarageYrBlt'].isnull().sum())
# "GarageYrBlt" columns 之 null值 用  "YearBuilt" columns的值代入
nanindex=train_df[train_df['GarageYrBlt'].isnull()]['GarageYrBlt'].index
train_df.loc[nanindex,'GarageYrBlt']=train_df.loc[nanindex,'YearBuilt'].values


print(train_df.isnull().any())
display(train_df)

#start machine learning------------------------------------------------------------------------------------------------------------------------

#LinearRegression

X = train_df.drop(["SalePrice"], axis = 1) # 自變數
y = train_df["SalePrice"]                  # 應變數

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

reg = LinearRegression()
reg.fit(X_train, y_train) 
print("R Square:    ", reg.score(X_test, y_test)) 

predicted_y = reg.predict(X_test) # 帶入驗證資料及的自變數 求得預測值
rmse = np.sqrt(mean_squared_error(y_test,predicted_y))
print("Root Mean Squared Error(均方根誤差): {}".format(rmse))  

#cross_val_score
for x in range(3,11):
    cvscores = cross_val_score(reg, X, y, cv = x)
    print(f"Average {x}-Fold CV Score: {np.mean(cvscores)}")
#-----------------------------------------------------------------------------------------------------------------------
#Regularization  1 (正規化回歸): Lasso
#正規化回歸模型會對回歸係數大小做出約束，並逐漸的將回歸係數壓縮到零。而對回歸係數的限制將有助於降低係數的幅度和波動，並降低模型的變異。

#透過在線性迴歸中加入L1懲罰函數(如下式)，目的在於讓模型中不要存在過多的參數，當模型參數越多時懲罰函數的值會越大
#  minimize {SSE+ λ*∑(j~p)=|βj|} 


# Instantiate a lasso regressor: lasso
lasso = Lasso(alpha=0.6,normalize=True)

# Fit the regressor to the data
lasso.fit(X,y)

# Compute and print the coefficients(係數)
lasso_coef = lasso.fit(X,y).coef_
print(lasso_coef)



plt.figure(figsize=(20,10))
plt.plot(range(len(X.columns)), lasso_coef)
plt.xticks(range(len(X.columns)), X.columns.values, rotation=60)
plt.margins(0.02)
plt.show()
#-----------------------------------------------------------------------------------------------------------
def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='red')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()

# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []

# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)

# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha
    
    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge, X, y, cv=10)
    
    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))
    
    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))

# Display the plot
display_plot(ridge_scores, ridge_scores_std)
#print(ridge_scores,alpha_space)
