#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import numpy as np
plot_scatter_matrix = pd.plotting.scatter_matrix


# In[2]:


data = pd.read_csv("housing.csv")
data["income_cat"] = pd.cut(data["median_income"], bins=[0, 1.5, 3, 4.5, 6, np.inf], labels=[1,2,3,4,5])
#creating new attributes
households = data["households"]
data["rooms_per_household"] = data["total_rooms"]/households
data["bedrooms_per_room"] = data["total_bedrooms"]/data["total_rooms"]
data["population_per_household"] = data["population"] / data["households"]


# In[3]:


# creating test and train data
split=StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data,data["income_cat"]):
    strat_train_set = data.loc[train_inde x]
    strat_test_set = data.loc[test_index]


# In[4]:


# correlation matrix
corr_matrix = data.corr()


# In[5]:


#seperating predictors and labels 
housing = strat_train_set.drop("median_house_value",axis=1)
housing_labels = strat_train_set["median_house_value"].copy()


# In[6]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity",axis=1)
imputer.fit(housing_num)
imputer.statistics_
X=imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)


# In[7]:


# Encoding categorical(textual) data into numerical 
from sklearn.preprocessing import OneHotEncoder
category_encoder = OneHotEncoder();
housing_cat = housing[["ocean_proximity"]]
hot_encoded_cat = category_encoder.fit_transform(housing_cat)


# In[8]:


rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
from sklearn.base import BaseEstimator, TransformerMixin
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
    


# In[9]:


# We can create a transformation pipeline for our data
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

text_num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")), 
    ("attribs_adder",CombinedAttributesAdder()), 
    ("std_scaler",StandardScaler()),
])


# In[10]:


housing_num_train = text_num_pipeline.fit_transform(housing_num)
housing_num_train.shape


# In[11]:


# Apply all the transformation to the columns appropriately
from sklearn.compose import ColumnTransformer
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ("num",text_num_pipeline, num_attribs), 
    ("cat",OneHotEncoder(),cat_attribs),
])
housing_final = full_pipeline.fit_transform(housing)
print(housing.shape, housing_final.shape)


# In[12]:


# Implementing a linear regression model 
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_final, housing_labels)


# In[19]:


# Testing few instances of data with out linear regression model 
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
housing_predictions = lin_reg.predict(some_data_prepared)


# In[21]:


from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(some_labels, housing_predictions))


# In[24]:


# using a decision tree regressor
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_final, housing_labels)

tree_predictions = tree_reg.predict(some_data_prepared)
tree_predictions


# In[31]:


tree_errrs = mean_squared_error(some_labels, tree_predictions)
tree_errrs


# In[47]:


# We have to do model validation now to process how well our model reacts to new data
from sklearn.model_selection import cross_val_score
# scores = cross_val_score(tree_reg,housing_final, housing_labels,scoring="neg_mean_squared_error",cv=10)
# tree_rmse_scores = np.sqrt(-scores)
lin_scores = cross_val_score(lin_reg,housing_final,housing_labels,scoring="neg_mean_squared_error",cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)


# In[49]:


s,m,std = lin_rmse_scores,lin_rmse_scores.mean(),lin_rmse_scores.std() 
print(s,m, std)


# In[ ]:




