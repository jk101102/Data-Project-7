import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
train_data=pd.read_csv("train.csv")
test_data=pd.read_csv("test.csv")
#df_all = pd.concat([df_train,df_test],ignore_index=True)
train_data.drop(['Id'], axis = 1, inplace = True)
train_data.drop(['Alley', 'PoolQC', 'MiscFeature', 'Fence'], axis = 1, inplace = True)
test_data.drop(['Id', 'Alley', 'PoolQC', 'MiscFeature', 'Fence'], axis = 1, inplace = True)
## Continuous Variables
con_var = train_data.dtypes[train_data.dtypes.values != 'object'].index
# Categorical Variables
cat_var = train_data.dtypes[train_data.dtypes.values == 'object'].index

print(con_var,cat_var)
con_data = train_data.copy()
con_data.drop(cat_var, axis = 1, inplace = True)
mask = np.zeros_like(con_data.corr(method = 'spearman')) #pearson kendall spearman
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize = (20,20)) 
sns.heatmap(con_data.corr(method = "spearman"), cmap = "YlGnBu", linewidths = 1, mask = mask)
corrs = con_data.corr('spearman')['SalePrice'].sort_values(ascending = False)

corrs_abs = corrs.abs()
corrs_abs[corrs_abs>0.5]
