#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# All the libraries necessary for the execution of the processes are imported

# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,accuracy_score


# ## Data Extraction

# The data is extracted from a csv file

# In[5]:


data=pd.read_csv("Data.csv")


# Shows the first five rows of our dataset

# In[6]:


data.head()


# Shows the last five rows of our dataset

# In[7]:


data.tail()


# ## Data Validation & Cleansing

# All the columns of our dataset

# In[8]:


data.columns = data.columns.str.strip()
data.columns


# No. of rows and columns in our dataset

# In[9]:


data.shape


# Generation of descriptive statistics to summarize central tendency, standard distribution, minimum and maximum values and percentiles

# In[10]:


data.describe()


# Finding the unique values for the features

# In[11]:


data.nunique()


# Find the missing values in our dataset

# In[12]:


data.isnull().sum()


# Drops the rows that contain exactly 2 null values

# In[13]:


data.drop(data[data.isnull().sum(axis=1) == 2].index, inplace=True)
data.isnull().sum()


# Drop data from columns that show less than 15 missing data

# In[14]:


data.dropna(subset=['Life expectancy', 'Adult Mortality'],inplace=True)
data.isnull().sum()


# For the other features, we observe a large amount of missing data, and conclude that we will lose a lot if we remove all the missing data from the dataset. So, we proceed to handle this by using the imputation process.

# ### Imputation

# Select the independent columns that have missing data

# In[15]:


mean_col_list = ['Alcohol','Hepatitis B', 'BMI', 'Polio', 'Total expenditure', 'Diphtheria',
                 'thinness  1-19 years', 'thinness 5-9 years', 'Income composition of resources', 'Schooling']
mean_col_list


# replace_mean is defined, this function replaces the null values of the series with the mean value of the series

# In[16]:


def replace_mean(series):
    series = series.fillna(series.mean(), inplace=True)
    return series


# The function is used by cycling through each column to convert the missing values to the mean

# In[18]:


for series in mean_col_list:
    replace_mean(data[series])

data.isnull().sum()


# Fill null values in 'GDP' and 'Population' columns with the median values of their respective columns

# In[19]:


data['GDP'].fillna(data['GDP'].median(),inplace=True)
data['Population'].fillna(data['Population'].median(),inplace=True)

data.isnull().sum()


# ## Data Analysis

# Information about the column data types, memory usage, column labels and null values

# In[20]:


data.info()


# The data is taken between 2000 and 2015

# In[21]:


print(data['Year'].min())
print(data['Year'].max())


# Plot the maximum number of people from each group

# In[22]:


plt.figure(figsize=(15, 6))
sns.histplot(data=data, x="Year")
plt.title("Number of people from each year")
plt.show()


# Generation of descriptive statistics to summarize central tendency, standard distribution, minimum and maximum values and percentiles of the Life Expectancy

# In[23]:


data['Life expectancy'].describe()


# Boxplot of Life Expectancy

# In[24]:


plt.figure(figsize=(6,2))
sns.boxplot(x= data['Life expectancy'])
plt.title('Distribution of Life Expectancy')
plt.show()


# Histogram of Life Expectancy

# In[25]:


sns.histplot(data['Life expectancy'], binrange=(35,90), bins=12)
plt.title('Distribution of Life Expectancy')
plt.xticks(list(range(35,95,5)))
plt.show()


# Boxplot of Status vs Life Expectancy

# In[26]:


sns.boxplot(data,x='Status',y='Life expectancy',hue='Status')
plt.title('Box Plot for the Life expectancy with status of country')


# Transforms 'Status' column from objects into booleans to be able to do mathematical analysis

# In[27]:


data['Status'] = data['Status'].replace({'Developing': True , 'Developed': False})
data.head()


# Calculate correlation for numerical features

# In[28]:


data.corr(numeric_only=True)


#  Create correlation heatmap

# In[29]:


plt.figure(figsize=(14,7))
sns.heatmap(data.corr(numeric_only=True), annot=True)
plt.title('Correlation')
plt.show()


# In[31]:


variable_list = ['Schooling', 'Income composition of resources', 'GDP', 'Diphtheria', 
                'Polio', 'BMI', 'percentage expenditure', 'Alcohol', 'Life expectancy','Status','Adult Mortality']
data_req = data[variable_list]
data_req.head()


# In[32]:


data_req.corr()


# In[34]:


sns.heatmap(data_req.corr(), annot=True)
plt.title('Correlation')
plt.show()


# ## Data Visualization

# Scatter plot showing the relationship between Life Expectancy and BMI

# In[35]:


sns.scatterplot(x=data['BMI'], y=data['Life expectancy'])
plt.title('Scatter plot between BMI and Life expectancy')
plt.show()


# Scatter plot showing the relationship between Life Expectancy and Schooling

# In[36]:


sns.lmplot(x='Schooling', y='Life expectancy', data=data, ci=None,line_kws={'color': 'red'})
plt.title('Scatter plot between Schooling and Life expectancy')
plt.show()


# Scatter plot showing the relationship between Life Expectancy and Adult Mortality

# In[37]:


sns.lmplot(x='Adult Mortality', y='Life expectancy', data=data, ci=None,line_kws={'color': 'red'})
plt.title('Scatter plot between Adult Mortality and Life expectancy with trend')
plt.show()


# Scatter plot showing the relationship between Life Expectancy and Income Composition

# In[38]:


sns.lmplot(x='Income composition of resources', y='Life expectancy', data=data, ci=None,line_kws={'color': 'red'})
plt.title('Scatter plot between Income composition of resources and Life expectancy with trend')
plt.show()


# ### Model Training and Testing

# Useful features are selected for the models

# In[39]:


variable_list = ['Schooling', 'Income composition of resources', 'GDP', 'Diphtheria', 
                'Polio', 'BMI', 'percentage expenditure', 'Alcohol', 'Life expectancy','Status','Adult Mortality']
data_last = data[variable_list]
data_last.head()


# Splitting the dataset into training and testing data

# In[40]:


X = data_last.drop('Life expectancy', axis=1)
y = data_last['Life expectancy']

print(X.shape)
print(y.shape)


# Scaling down the values using Standard Scaler

# In[41]:


scaler=StandardScaler()
X_standardized_data = scaler.fit_transform(X)
print(X_standardized_data)


# Splitting the training dataset into X_train, X_test, y_train and y_test

# In[42]:


X_train, X_test, y_train, y_test = train_test_split(X_standardized_data, y, test_size=0.2, random_state=42)


# ## 1. Multiple Linear Regression Model (MLRM)

# Initiate the multiple linear regression model and fit into the training data

# In[43]:


LR = LinearRegression()
LR.fit(X_train, y_train)


# Save predicted data

# In[44]:


y_pred = LR.predict(X_test)


# Evaluate the model performance on the training data

# In[45]:


print('Coefficient of determination: ', LR.score(X_train, y_train))


# Calculate R² score

# In[46]:


r2_lr = r2_score(y_test, y_pred)
print(f'R² Score: {r2_lr}')


# Calculate Mean Absolute Error (MAE)

# In[47]:


mae_lr = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error (MAE): {mae_lr}')


# Calculate Mean Squared Error (MSE)

# In[48]:


mse_lr = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error (MSE): {mse_lr}')


# Calculate Root Mean Squared Error (RMSE)

# In[49]:


rmse_lr=np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE: ', rmse_lr)


# ## 2. Random Forest Regression Model (RFRM)

# Initiate the random forest regression model and fit into the testing data

# In[50]:


RF = RandomForestRegressor(n_estimators=100,random_state=42)
RF.fit(X_train, y_train)


# Save predicted data

# In[51]:


y_pred1 = RF.predict(X_test)


# Evaluate the model performance on the training data

# In[52]:


print('Coefficient of determination: ', RF.score(X_train, y_train))


# Calculate R² score

# In[53]:


r2_rf = r2_score(y_test, y_pred1)
print(f'R² Score: {r2_rf}')


# Calculate Mean Absolute Error (MAE)

# In[54]:


mae_rf = mean_absolute_error(y_test, y_pred1)
print(f'Mean Absolute Error (MAE): {mae_rf}')


# Calculate Mean Squared Error (MSE)

# In[55]:


mse_rf = mean_squared_error(y_test, y_pred1)
print(f'Mean Squared Error (MSE): {mse_rf}')


# Calculate Root Mean Squared Error (RMSE)

# In[56]:


rmse_rf=np.sqrt(mean_squared_error(y_test, y_pred1))
print('RMSE: ', rmse_rf)


# ## Utilization of Analysis Results

# 1. Multiple Linear Regression Model (MLRM)
# 
# Create results dataframe

# In[57]:


results_MLRM = pd.DataFrame({'actual': y_test, 'predicted': y_pred.ravel(), 'residual': y_test - y_pred})
results_MLRM.head()


# In[58]:


sns.scatterplot(x='actual', y='predicted', data=results_MLRM)
sns.regplot(x='actual',y='predicted', data=results_MLRM, color='Green',scatter=False)
plt.title('Actual vs Predicted values for MLRM')
plt.show()


# 2. Random Forest Regression Model (RFRM)
# 
# Create results dataframe

# In[59]:


results_RFRM = pd.DataFrame({'actual': y_test, 'predicted': y_pred1.ravel(), 'residual': y_test - y_pred1})
results_RFRM.head()


# In[60]:


sns.scatterplot(x='actual', y='predicted', data=results_RFRM)
sns.regplot(x='actual',y='predicted', data=results_RFRM, color='Green',scatter=False)
plt.title('Actual vs Predicted values for RFRM')
plt.show()

