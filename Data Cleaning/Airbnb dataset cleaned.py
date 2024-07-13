#!/usr/bin/env python
# coding: utf-8

# In[53]:


import pandas as pd
import numpy as np
df = pd.read_csv(r"C:\Users\afiroz\Downloads\AB_NYC_2019.csv")


# In[54]:


# Reading the 5 rows
df.head(5)


# In[55]:


# Checking the num of rows and columns
df.shape


# In[99]:


#checking the last 5 rows
df.tail(100)


# In[57]:


# Checking the columns
df.columns


# In[58]:


# checking to see if there are any duplicate values
len(df) - len(df.drop_duplicates())


# ### No duplicate values found

# In[59]:


# More info
df.info()


# In[60]:


#Checking the null values
df.isnull().sum()


# In[61]:


#  Checking the rationality of null values
round(df.isnull().sum()/len(df),10).sort_values(ascending=False)


# ### Dropping the null values and checking it

# In[69]:


# Dropping the null values from name and host_name column
df = df.dropna(subset=['name'])
df = df.dropna(subset=['host_name'])


# In[70]:


df.isnull().sum()


# In[71]:


df.shape


# ### Working with the other two null columns
# 

# In[74]:


# Creating a sample
df[df['last_review'].isnull()].sample(6)


# In[82]:


# Correcting data mismatch
df['last_review'] = pd.to_datetime(df['last_review'], format='%Y-%m-%d')


# In[83]:


df.info()


# ### Now, going to fill the null values of last_review, reviews_per_month column because the number_of_reviews has positive co-relation with the two columns.

# In[90]:


df['last_review']=df['last_review'].fillna(0)
df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
df.sample(5)


# In[91]:


# Checking if any other null values are left
df.isnull().sum()/len(df)


# ## Outlier Detection

# In[112]:


def detect_outliers_iqr(data, column):
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = np.percentile(data[column], 25)
    Q3 = np.percentile(data[column], 75)
    
    # Calculate IQR (Interquartile Range)
    IQR = Q3 - Q1
    
    # Define the lower and upper bound for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Detect outliers
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    
    return outliers

# Detect outliers in the 'price' column
outliers_price = detect_outliers_iqr(df, 'price')

# Display the number of outliers and the first few outliers
outliers_price_info = {
    'Number of outliers': len(outliers_price),
    'Outliers': outliers_price.head()
}

outliers_price_info


# In[118]:


# Outlier Visualization


# In[121]:


# Box Plot

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# Box Plot for Price
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='price')
plt.title('Box Plot of Price')
plt.xlabel('Price')
plt.show()


# In[122]:


# Scatter Plot

# Convert minimum_nights to numeric, coerce errors to NaN
df['minimum_nights'] = pd.to_numeric(df['minimum_nights'], errors='coerce')

# Drop rows with NaN values in price and minimum_nights
df = df.dropna(subset=['price', 'minimum_nights'])

# Scatter Plot for Price vs Minimum Nights
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='minimum_nights', y='price', hue='neighbourhood_group', palette='Set1', alpha=0.6)
plt.title('Scatter Plot of Price vs Minimum Nights')
plt.xlabel('Minimum Nights')
plt.ylabel('Price')
plt.legend(title='Neighbourhood Group')
plt.show()


# # Downloading the dataset

# In[132]:


df.to_csv(r"C:\Users\afiroz\Downloads\cleaned_dataset_1.csv")


# # Accomplishments:
# ### Here, the duplicate values are found which is zero. 
# ### The null values are found and replaced.
# ### The unmatched datas are formatted and data consistency is maintained through this cleaning precedure.
# ### The outliers have also been detected which could cause harm during model building.

# In[ ]:




