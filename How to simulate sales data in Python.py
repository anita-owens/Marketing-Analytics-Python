#!/usr/bin/env python
# coding: utf-8

# # How to simulate sales data in Python

# Why simulate data?
# 
# Have you ever spent an inordinate amount of time looking for the right
# data set to try out an analytical technique, but you don't quite find
# what you are looking for.
# 
# Well, why not just create your own dataset for maximum flexibility which
# gives you a few advantages:
# 
# 1.  To test your analysis to make sure it's working.
# 2.  To make changes to your dataset to see what happens and how your
#     conclusions might change based on different parameters . (e.g. How
#     sales are impacted by different levels of the price of our product).

# ## Import Packages

# In[1]:


#Import pandas and numpy
import pandas as pd
import numpy as np
from numpy import random

# From matplotlib, import pyplot under the alias plt
from matplotlib import pyplot as plt

#Import Seaborn
import seaborn as sns

#Import datetime for working with dates
from datetime import datetime


# ## Data Simulation

# First, we will create a skeleton data frame to hold our variables and
# initialize with 0 values for all columns except the date column.
# 
# 1.  Sales date (week level granularity)
# 2.  Unit sales
# 3.  Unit price
# 4.  Paid social ads
# 5.  Promotion - whether there was a promotion this week.

# In[2]:


#Set seed so that when reproducing the dataset, we get the same results.
np.random.seed(90210)


# Visit https://pandas.pydata.org/docs/user_guide/timeseries.html for a list of all parameters we can use in the freq argument of the date_range function, but since we want weekly dates, we will use the freq = 'W-TUE' argument since Jan 1st was a Tuesday.

# In[3]:


#Create date sequence of 52 weeks using date_range function
sales_date = pd.date_range('2019-01-01', periods=52, freq = 'W-TUE')

#Check results
print(sales_date)


# We want to add weeks with marketing promotions. We can simulate promotions using the binomial distribution. The notation of the binomial distribution is B(n,p), where n is the number of experiments or trials, and p is the probability of success.

# In[4]:


#Create weeks where there were promotions running. 10% likelihood of a promotion
promotion = np.random.binomial(n=1, p=0.10, size=len(sales_date))

#Check results
print(promotion)


# Let's create data for the paid social ads. We want to add Paid Social values for the range of dates between July and September and again in December.

# In[5]:


#Create a repeating list of 0's
social = [0] * len(sales_date)

print(social)


# Generate the product's price and place in a vector and then we'll use
# the random.choices method to randomly create prices. We will only create two (2) price points for our product. We want to use random sampling with replacement. If you wanted to use random sampling without replacement, you could use the random.sample function from the random module.

# In[6]:


import random #So as not to be confused with np.random

prices_list = [4.50, 4.99]

#Create price values
price = random.choices(prices_list, k = len(sales_date))

#Check output
print(price)


# ### Generate unit sales

# Next step is to generate sales data based on unit sales and place into a temporary sales variable. Sales is randomly generated based on a poisson distribution.

# In[7]:


#Generate unit sales - poisson distribution
temp_sales = np.random.poisson(size=len(sales_date), lam = 8300)

#Check results
print(temp_sales)


# In[8]:


#Scale sales up according to price to follow a logarithmic function
#Scale sales by multiplying sales by the log of price
temp_sales = temp_sales * np.log(price)

#Check output
print(temp_sales)


# In our last step, we need to take our temporary sales and
# add an increase of 30% in unit sales for weeks we have the promotion
# running. We want to show the boost in sales in the weeks where there are
# promotions. We will add the floor function from numpy to remove fractions and return the largest integer.

# In[9]:


#Add impact of increased sales due to week where a promotion was running
unit_sales = np.floor(temp_sales * (1 + promotion * 0.30))

#Check results
print(unit_sales)


# ## Putting it all together

# Can use the zip function to convert lists of rows into lists of columns

# In[10]:


# Creating DataFrame
# Can use the zip function to convert lists of rows into lists of columns
df = pd.DataFrame(list(zip(sales_date, unit_sales, promotion, social, price)), columns = ['sales_date', 'unit_sales', 'promotion', 'social', 'price'])

# displaying resulting DataFrame
print(df)


# Replacing values in pandas DataFrame based on single or multiple conditions
# 
# 
# **On a specific date:**
# If you want to select and replace 0 values on a specific date, use the following code:
# 
# df.loc[df.sales_date == '2019-01-06', 'social'] = 200
# 
# 
# **Between 1 date range:**
# If you want to select and replace 0 values between one date range (in between 2 specific dates,) use the following code:
# 
# 
# df.loc[(df['sales_date'] >= '2019-07-02') & (df['sales_date'] <= '2019-09-10'), 'social'] = 200
# 
# 
# **Between 2 date ranges:**
# If you want to select and replace 0 values between two date ranges, use the following code:
# 
# df.loc[(df['sales_date'] >= '2019-07-02') & (df['sales_date'] <= '2019-09-10') | (df['sales_date'] >= '2019-12-03') & (df['sales_date'] <= '2019-12-24'), 'social'] = 200
# 

# In[11]:


#Select and replace values between dates (We have 2 date ranges were we spent different amounts)
df.loc[(df['sales_date'] >= '2019-07-02') & (df['sales_date'] <= '2019-09-10'), 'social'] = 350
df.loc[(df['sales_date'] >= '2019-12-03') & (df['sales_date'] <= '2019-12-24'), 'social'] = 200
print(df)


# ## Check Results

# In[12]:


#Descriptive statistics
print(df.describe())


# In[13]:


# Set the color palette
sns.set_palette(sns.color_palette("vlag"))

#Initialize subplots with number of rows and number of columns
figure, ax = plt.subplots(nrows = 2, ncols = 2)

#Get readable axis labels for plot 1 which has a date on the x-axis(time series plot)
plt.sca(ax[0, 0])
plt.xticks(rotation=45, fontsize = 8)
#plt.xticks([]) #This disables the x-ticks (uncomment and run if you prefer the x-axis to be blank)

#See the distribution of the data
sns.lineplot(data=df, x="sales_date",y="unit_sales", ax=ax[0,0])
sns.boxplot(data=df, x="promotion", y="unit_sales", ax=ax[0,1])
sns.scatterplot(data=df, x="social", y="unit_sales", ax=ax[1,1])
sns.boxplot(data=df, x="price", y="unit_sales", ax=ax[1,0])

#Gives a tidy layout
plt.tight_layout()

#Show plot
plt.show()


# In[14]:


#Correlation Plot Matrix
#Pearson is the default correlation method that is used for normally distributed data
corr = df.corr(method = "pearson")
print(corr)


# In[15]:


corr.style.background_gradient(cmap='RdBu')


# ## Export simulated dataset

# In[16]:


#Save simulated dataset as a csv file or excel file
df.to_csv("datasets/weekly_sales_data.csv")

#Using to_excel from pandas
#https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_excel.html#
df.to_excel("datasets/weekly_sales_data.xlsx", sheet_name='Sheet1')


# In[17]:


#Check if files exist in directory we designated above using os package
import os

#Check file one - csv file
os.path.isfile("datasets/weekly_sales_data.csv")


# In[18]:


#Check file two - Excel sheets
os.path.isfile("datasets/weekly_sales_data.xlsx")


# ***

# #### References

# [1]  Chapman, C. and McDonnell Feit, E., (2015). R for marketing research
#     and analytics. Cham: Springer, pp.47-59, 162-191.
