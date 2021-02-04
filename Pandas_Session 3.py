#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd


# In[2]:


data = pd.DataFrame([[1,4,2], [2,3,5], [7,4,5]])
data


# In[3]:


print(data.sum()) #sum by default is adding the rows


# In[4]:


print(data.sum(axis=1)) #axis=1 ,adding the column elements


# In[7]:


data.sum(axis=1) #axis=1 ,adding the column elements


# In[8]:


print(data.max(axis=0)) # print maximum of eaxh column is finding maximum across rows


# In[9]:


print(data.max(axis=1)) # print maximum of eaxh row is finding maximum across columns


# In[10]:


new_df=pd.DataFrame(data.values,columns= ["A", "B", "C"], index= ["a", "b", "c"])
print(new_df)


# In[11]:


print(new_df.sum()) # sumby default is adding the rows


# In[12]:


print(new_df.sum(axis=1)) # axis=1,adding the column elements


# In[13]:


print(new_df.max(axis=0)) # prints maximum of each column i.e. finding maximum across each rows
print(new_df.max(axis=1)) # prints maximum of each row i.e. finding maximum across each column


# In[8]:


data = pd.DataFrame([[1,4,2], [2,3,5], [7,4,5]])
data


# In[9]:


print(data)
new_df=pd.DataFrame(data.values,columns= ["A", "B", "C"], index= ["a", "b", "c"])
print(new_df)


# In[10]:


new_df=pd.DataFrame(data.values,columns= ["A", "B", "C"], index= ["a", "b", "c"])
print(new_df)


# In[15]:


print(new_df)
print("\n", new_df.sort_values(by = "B", axis=0)) #sort a column


# In[14]:


print(new_df.max(axis=0)) # prints maximum of each column i.e. finding maximum across each rows
print(new_df.max(axis=1)) # prints maximum of each row i.e. finding maximum across each column


# In[15]:


print(new_df)
print("\n", new_df.sort_values(by = "C", axis=1)) #sort a row


# In[11]:


print(new_df)
print("\n", new_df.sort_values(by = "C", axis=0)) #sort a row


# In[13]:


print(new_df)
#print("\n", new_df.sort_values(by = "B", axis=0)) #sort a row
print("\n", new_df.sort_values(by = "C", axis=1)) #sort a row


# In[17]:


# drop
print(new_df)
print(new_df.drop(['a', 'b'],axis=0)) #axis=0,row deletion using row label
print(new_df.drop(['B'],axis=1)) #axis=1,column deletion
# print(new_df)


# In[18]:


## Visualizing a DataFrames
#First Create a DataFrame
data={'Animals': ['cat','cat','turtle','dog','dog','cat','turtle','cat','dog','dog'],
          'Age': [2.5,3,0.5,np.nan,5,2,4.5,np.nan,7,3], 
      'Visits' : [1,3,2,3,2,3,1,1,2,1], 
    'Priority' : ['y','y','n','y','n','n','n','y','n','n']}
labels=['a','b','c','d','e','f','g','h','i','j']
animals_data=pd.DataFrame(data,index=labels)
animals_data
#observe the data type of each column


# In[19]:


print("Mean of the Dataframe is: \n",animals_data.mean()) #mean of values in columns containg numeric data


# In[24]:


print("\n Mean of 'Age' is: ",animals_data[['Age']].mean())


# In[25]:


print("\n Total visits :",animals_data[['Visits']].sum())


# In[22]:


print("\n Sum of visits :",animals_data[['Visits']].sum())


# In[23]:


print("\n Max visits: ",animals_data[['Visits']].max())


# In[26]:


print("\n Index of Max visits: ",animals_data[['Visits']].idxmax()) # for what index is maximum


# In[27]:


print("\n Index of Min visits: ",animals_data[['Visits']].idxmin())


# In[28]:


print("\nSum: \n",animals_data.sum()) #for strings sum is string concatenation


# In[29]:


print("\n Index of Min visits: ",animals_data[['Visits']].idxmin())


# In[30]:


# Handling Missing Values
print(animals_data.info())
print(animals_data)


# In[32]:


# Handling Missing Values
print(animals_data.info())
animals_data


# In[36]:


# Difference between None and np.nan
import numpy as np
arr1 = np.array([1,None,3,4]) 
print(arr1,arr1.dtype)
print(arr1.mean())


# In[37]:


#Trouble with missing data
#Why we need to drop missing values
import numpy as np
arr2 = np.array([1, np.nan, 3,4]) #np.nan is a float type
print(arr2, arr2.dtype)
print(arr2.sum()) #so np.nan is handled by numpy but not None
print(arr2.mean())


# In[38]:


#print(pd.Series([1, np.nan, 2, None]))
ser_null = pd.Series([1,np.nan,2,None])
print('\n',ser_null.sum())
print('\n', ser_null.mean())


# In[40]:


# Dataframe aggregation methods ignore nan values and find the sum
data = pd.DataFrame([[1, np.nan, 2],[2, 3, 5],[np.nan, 4, 6]])
print(data)
print(data.sum()) #sum by default is column sum axis =0
print(data.sum(axis=1)) #sum across columns
print(data.sum(axis=0)) #sum across columns


# * Pandas handles np.nun by ignoring these values in the statistical calculations

# In[42]:


#Detecting and dropping Null Values
#print(pd.isnull(animals_data)) #isnull() function in pandas library
#print(animals_data.isnull()) #isnull() in DataFrame object
#Observe that Age has two missing values
#print(pd.notnull(animals_data))
print(data)
data = pd.DataFrame([[1, np.nan, 2],[2, 3, 5], [np.nan, 4, 6]])
print('\n data.isnull(): \n',data.isnull())
print('\n data.notnull(): \n',data.notnull())
#data[data.notnull()]


# In[43]:


#Detecting and dropping Null Values
#print(pd.isnull(animals_data)) #isnull() function in pandas library
#print(animals_data.isnull()) #isnull() in DataFrame object
#Observe that Age has two missing values
#print(pd.notnull(animals_data))
print(data)
data = pd.DataFrame([[1, np.nan, 2],[2, 3, 5], [np.nan, 4, 6]])
print('\n data.isnull(): \n',data.isnull().sum())
#print('\n data.notnull(): \n',data.notnull())
#data[data.notnull()]


# In[44]:


# dropping null values
print(data)
print(data.dropna()) # drop rows and columns with null values


# In[18]:


data=pd.DataFrame([[1,np.nan,2], [2,np.nan,5], [np.nan,4,6]])
print(data)
#data.dropna(axis='rows', thresh=1) # axis=0 means drop rows which have missing values
data.dropna(axis='columns', thresh=1) # axis=0 means drop rows which have missing values
#thresh = 2 means 2 Nansand beyond is dropped


# In[19]:


data=pd.DataFrame([[1,np.nan,2], [2,np.nan,5], [np.nan,4,6]])
print(data)
data.dropna(axis='rows', thresh=1) # axis=0 means drop rows which have missing values
#data.dropna(axis='columns', thresh=1) # axis=0 means drop rows which have missing values
#thresh = 2 means 2 Nansand beyond is dropped


# In[48]:


data.dropna(axis='rows', thresh=2) # axis=0 means drop rows which have missing values
data.dropna(axis='columns') # axis=0 means drop rows which have missing values


# In[47]:


data.dropna(axis='rows', thresh=1)


# In[49]:


data = pd.DataFrame([[1, np.nan, 2],[2, 3, 5], [np.nan, 4, 6]])
print(data)
# data.dropna(axis='rows', thresh=1) # axis=0 means drop rows wich have more missing values
data.dropna(axis='columns', thresh=2) # axis=0 means drop rows wich have more missing values
# thresh = 2 means it needs 2 non null values to retain a column.
# thresh should be a large to have more non null values


# In[50]:


data = pd.DataFrame([[1, np.nan, 2],[2, 3, 5], [np.nan, 4, 6]])
print(data)
# data.dropna(axis='rows', thresh=1)  # axis=0 means drop rows wich have more missing values
data.dropna(axis='columns', thresh=3) # axis=0 means drop rows wich have more missing values
# thresh should be a large to have more non null values


# In[52]:


data = pd.DataFrame([[1, np.nan, 2],[2, 3, 5], [np.nan, 4, 6]])
print(data)
# data.dropna(axis='rows', thresh=1) # axis=0 means drop rows wich have more missing values
data.dropna(axis='columns', thresh=4,inplace=True) # axis=0 means drop rows wich have more missing values
# thresh = 2 means it needs 2 non null values to retain a column.
# thresh should be a large to have more non null values


# In[55]:


## Visualizing a DataFrames
#First Create a DataFrame
data={'Animals': ['cat','cat','turtle','dog','dog','cat','turtle','cat','dog','dog'],
          'Age': [2.5,3,0.5,np.nan,5,2,4.5,np.nan,7,3], 
      'Visits' : [1,3,2,3,2,3,1,1,2,1], 
    'Priority' : ['y','y','n','y','n','n','n','y','n','n']}
labels=['a','b','c','d','e','f','g','h','i','j']
animals_data=pd.DataFrame(data,index=labels)
animals_data
#observe the data type of each column


# In[56]:


print(animals_data.dropna(axis='columns', thresh=2)) # if 2 non null value is there column is not dropped


# In[57]:


print(animals_data.dropna(axis='columns', thresh=8)) # if 8 non null value is there column is not dropped


# In[59]:


print(animals_data.dropna(axis='columns', thresh=9)) # if 9 non null value is there column is dropped


# In[58]:


print(animals_data.dropna(axis='columns', thresh=9)) # if 9 non null value is there column is dropped


# In[20]:


data = pd.DataFrame([[1, np.nan, 2],
 [2, 3, 5],
 [np.nan, 4, 6]])
print(data)
#print(data.fillna(0)) #we can fill with column mean or mode for categorical data
#print(data.fillna(method='ffill'))
print(data.fillna(method='bfill'))
print(data) # original data will not change, to change we need to set inplace = True
#find mean of each column and fill each individually


# In[6]:


data = pd.DataFrame([[1, np.nan, 2],
 [2, 3, 5],
 [np.nan, 4, 6]])
print(data)
print(data.fillna(0)) #we can fill with column mean or mode for categorical data
print(data.fillna(method='ffill'))
#print(data.fillna(method='bfill'))
print(data) # original data will not change, to change we need to set inplace = True
#find mean of each column and fill each individually


# In[7]:


data = pd.DataFrame([[1, np.nan, 2],
 [2, 3, 5],
 [np.nan, 4, 6]])
print(data)
print(data.fillna(0)) #we can fill with column mean or mode for categorical data
#print(data.fillna(method='ffill'))
print(data.fillna(method='bfill'))
print(data) # original data will not change, to change we need to set inplace = True
#find mean of each column and fill each individually


# In[63]:


## Visualizing a DataFrames
#First Create a DataFrame
data={'Animals': ['cat','cat','turtle','dog','dog','cat','turtle','cat','dog','dog'],
          'Age': [2.5,3,0.5,np.nan,5,2,4.5,np.nan,7,3], 
      'Visits' : [1,3,2,3,2,3,1,1,2,1], 
    'Priority' : ['y','y','n','y','n','n','n','y','n','n']}
labels=['a','b','c','d','e','f','g','h','i','j']
animals_data=pd.DataFrame(data,index=labels)
print(animals_data)
print(animals_data.fillna(0))
#print(animals_notnull)
#print("\n\n",animals_data.fillna(animals_data['Age'.mean()])) 
#observe the data type of each column


# In[7]:


## Visualizing a DataFrames
#First Create a DataFrame
data={'Animals': ['cat','cat','turtle','dog','dog','cat','turtle','cat','dog','dog'],
          'Age': [2.5,3,0.5,np.nan,5,2,4.5,np.nan,7,3], 
      'Visits' : [1,3,2,3,2,3,1,1,2,1], 
    'Priority' : ['y','y','n','y','n','n','n','y','n','n']}
labels=['a','b','c','d','e','f','g','h','i','j']
animals_data=pd.DataFrame(data,index=labels)
print(animals_data)
#print(animals_data.fillna(0))
#print(animals_notnull)
print("\n\n",animals_data.fillna(animals_data[['Age']].mean()))  
#observe the data type of each column


# In[9]:


## Visualizing a DataFrames
#First Create a DataFrame
data={'Animals': ['cat','cat','turtle','dog','dog','cat','turtle','cat','dog','dog'],
          'Age': [2.5,3,0.5,np.nan,5,2,4.5,np.nan,7,3], 
      'Visits' : [1,3,2,3,2,3,1,1,2,1], 
    'Priority' : ['y','y','n','y','n','n','n','y','n','n']}
labels=['a','b','c','d','e','f','g','h','i','j']
animals_data=pd.DataFrame(data,index=labels)
print(animals_data)
print(animals_data.fillna(0))
#print(animals_notnull)
#print("\n\n",animals_data.fillna(animals_data[['Age']].mean()))  
#observe the data type of each column


# In[66]:


#First Create a DataFrame
data={'Animals': ['cat','cat','turtle','dog','dog','cat','turtle','cat','dog','dog'],
          'Age': [2.5,3,0.5,np.nan,5,2,4.5,np.nan,7,3], 
      'Visits' : [1,3,2,3,2,3,1,1,2,1], 
    'Priority' : ['y','y','n','y','n','n','n','y','n','n']}
labels=['a','b','c','d','e','f','g','h','i','j']
animals_data=pd.DataFrame(data,index=labels)
print(animals_data)
#print(animals_data.fillna(0))
print("\n\n",animals_data.fillna(animals_data['Age'].mean()))  
#observe the data type of each column


# In[ ]:


data = pd.DataFrame([[1, np.nan, 2],
 [2, 3, 5],
 [np.nan, 4, 6]])
print(data)
#print(data.fillna(0)) we can fill with column mean or mode for categorical
data
#print(data.fillna(method='ffill'))
print(data.fillna(method='bfill'))
print(data)
# original data will not change, to change we need to set inplace = True
#find mean of each column and fill each individually


# In[23]:


#First Create a DataFrame
data={'Animals': ['cat','cat','turtle','dog','dog','cat','turtle','cat','dog','dog'],
          'Age': [2.5,3,0.5,np.nan,5,2,4.5,np.nan,7,3], 
      'Visits' : [1,3,2,3,2,3,1,1,2,1], 
    'Priority' : ['y','y','n','y','n','n','n','y','n','n']}
labels=['a','b','c','d','e','f','g','h','i','j']
animals_data=pd.DataFrame(data,index=labels)
print(animals_data)
print(animals_data.fillna(0))
#prin(animals_notnull)
print("\n\n",animals_data.fillna(animals_data['Age'].mean()))
#print(data.fillna(method='ffill')) # carry the previous data forward
print(data)
#print(data.fillna(method='bfill'))  # carry the following data backward
#observe the data type of each column


# In[24]:


#First Create a DataFrame
data={'Animals': ['cat','cat','turtle','dog','dog','cat','turtle','cat','dog','dog'],
          'Age': [2.5,3,0.5,np.nan,5,2,4.5,np.nan,7,3], 
      'Visits' : [1,3,2,3,2,3,1,1,2,1], 
    'Priority' : ['y','y','n','y','n','n','n','y','n','n']}
labels=['a','b','c','d','e','f','g','h','i','j']
animals_data=pd.DataFrame(data,index=labels)
print(animals_data)
animals_notnull= animals_data.fillna(0)
print("\n\n", animals_data.fillna(0.8))
#print(animals_data.fillna(0))
#print(animals_notnull)
#print("\n\n",animals_data.fillna(animals_data['Age'].mean()))
#print(data.fillna(method='ffill')) # carry the previous data forward
#print(data.fillna(method='bfill'))  # carry the following data backward
#print(data)
#print(animals_notnull)
#observe the data type of each column


# In[26]:


#First Create a DataFrame
data={'Animals': ['cat','cat','turtle','dog','dog','cat','turtle','cat','dog','dog'],
          'Age': [2.5,3,0.5,np.nan,5,2,4.5,np.nan,7,3], 
      'Visits' : [1,3,2,3,2,3,1,1,2,1], 
    'Priority' : ['y','y','n','y','n','n','n','y','n','n']}
labels=['a','b','c','d','e','f','g','h','i','j']
animals_data=pd.DataFrame(data,index=labels)
print(animals_data)
animals_notnull= animals_data.fillna(0)
print("\n\n", animals_data.fillna(0.8))
print(animals_data.fillna(0))
#print(animals_notnull)
#print("\n\n",animals_data.fillna(animals_data['Age'].mean()))
#print(data.fillna(method='ffill')) # carry the previous data forward
#print(data.fillna(method='bfill'))  # carry the following data backward
#print(data)
#print(animals_notnull)
#observe the data type of each column


# In[48]:


#First Create a DataFrame
data={'Animals': ['cat','cat','turtle','dog','dog','cat','turtle','cat','dog','dog'],
          'Age': [2.5,3,0.5,np.nan,5,2,4.5,np.nan,7,3], 
      'Visits' : [1,3,2,3,2,3,1,1,2,1], 
    'Priority' : ['y','y','n','y','n','n','n','y','n','n']}
labels=['a','b','c','d','e','f','g','h','i','j']
animals_data=pd.DataFrame(data,index=labels)
print(animals_data)
#animals_notnull= animals_data.fillna(0)
#print("\n\n", animals_data.fillna(0.5))
#print(animals_data.fillna(0))
#print(animals_notnull)
#print("\n\n",animals_data.fillna(animals_data['Age'].mean()))
print(animals_data.fillna(method='ffill')) # carry the previous data forward
#print(data.fillna(method='bfill'))  # carry the following data backward
print(animals_data)
#print(animals_notnull)
#observe the data type of each column


# In[50]:


#First Create a DataFrame
data={'Animals': ['cat','cat','turtle','dog','dog','cat','turtle','cat','dog','dog'],
          'Age': [2.5,3,0.5,np.nan,5,2,4.5,np.nan,7,3], 
      'Visits' : [1,3,2,3,2,3,1,1,2,1], 
    'Priority' : ['y','y','n','y','n','n','n','y','n','n']}
labels=['a','b','c','d','e','f','g','h','i','j']
animals_data=pd.DataFrame(data,index=labels)
print(animals_data)
#animals_notnull= animals_data.fillna(0)
#print("\n\n", animals_data.fillna(0.5))
#print(animals_data.fillna(0))
#print(animals_notnull)
#print("\n\n",animals_data.fillna(animals_data['Age'].mean()))
#print(animals_data.fillna(method='ffill')) # carry the previous data forward
print(animals_data.fillna(method='bfill'))  # carry the following data backward
print(animals_data)
#print(animals_notnull)
#observe the data type of each column


# In[34]:


#First Create a DataFrame
data={'Animals': ['cat','cat','turtle','dog','dog','cat','turtle','cat','dog','dog'],
          'Age': [2.5,3,0.5,np.nan,5,2,4.5,np.nan,7,3], 
      'Visits' : [1,3,2,3,2,3,1,1,2,1], 
    'Priority' : ['y','y','n','y',' ','n','n','  ','n','n']}
labels=['a','b','c','d','e','f','g','h','i','j']
animals_data=pd.DataFrame(data,index=labels)
print(animals_data)
print(animals_data['Priority'])
#print(animals_data.fillna(0))
#print(animals_notnull)
#print("\n\n",animals_data.fillna(animals_data['Age'].mean()))
#print(data.fillna(method='ffill')) # carry the previous data forward
#print(data.fillna(method='bfill'))  # carry the following data backward
#print(data)
#print(animals_notnull)
#observe the data type of each column


# In[46]:


data = pd.DataFrame([[1,      np.nan, 2],    
                   [2,      3,      5],     
                   [np.nan, 4,      6]]) 
print(data)
#print(data.fillna(0))     we can fill with column mean or mode for categorical data
#print(data.fillna(method='ffill'))
print(data.fillna(method='bfill'))
print(data)          # original data will not change, to change we need to set inplace = True
#find mean of each column and fill each individually


# In[78]:


animals_data.to_csv('animal.csv')


# In[79]:


df_animal=pd.read_csv('animal.csv')
print(df_animal)
#df_animal.head(3)


# In[81]:


df_animal=pd.read_csv('animal.csv')
#print(df_animal)
df_animal.head(3)


# In[85]:


animals_data.to_excel('animals.xlsx',sheet_name='sheet1')
#animals_data.to_excel('animals.xlsx',sheet_name='sheet1')
df_animal2=pd.read_excel('animals.xlsx','sheet1', index_col=None)
df_animal2


# In[36]:


animals_data.to_excel('animals.xlsx',sheet_name='sheet1')
#animals_data.to_excel('animals.xlsx',sheet_name='sheet1')
df_animal2=pd.read_excel('animals.xlsx','sheet1', index_col=None, na_values=['NA'])
df_animal2


# In[91]:


# Series Concatenation
ser1 = pd.Series(['A', 'B', 'C'], index=[1, 2, 3])
ser2 = pd.Series( ['D', 'E', 'F'], index=[4, 5, 6] ) #test with the same indser1
print(ser1)
print(ser2)
print("\n\n Concatenated Series\n", pd.concat([ser1,ser2])) # append ser2 below ser1
ser3=ser1.append(ser2)
ser3


# In[95]:


#DataFrame Concatenation
df1 = pd.DataFrame({'A' : ['axe', 'art', 'ant'], 'B' : ['bat', 'bar', 'bin'], 'C': ['cap', 'cat', 'car']},index = [1,2,3])
df2 = pd.DataFrame({'D' : ['dam', 'den', 'dot'], 'E': [ 'ear', 'eat', 'egg'], 'F':['fan', 'fog', 'fat']}, index =[ 2, 3, 6])
print("Data frame 1 : \n", df1,'\n Data Frame 2: \n', df2)
print("Concatenating Data Frames: \n",pd.concat([df1,df2], axis=0)) # axis =0 is stacking one below the other
#print("Concatenating Data Frames along axis 1: \n",pd.concat([df1,df2], axis = 1))
#will consider common indices
df1
#df2


# In[37]:


#DataFrame Concatenation
df1 = pd.DataFrame({'A' : ['axe', 'art', 'ant'], 'B' : ['bat', 'bar', 'bin'], 'C': ['cap', 'cat', 'car']},index = [1,2,3])
df2 = pd.DataFrame({'D' : ['dam', 'den', 'dot'], 'E': [ 'ear', 'eat', 'egg'], 'F':['fan', 'fog', 'fat']}, index =[ 2, 3, 6])
print("Data frame 1 : \n", df1,'\n Data Frame 2: \n', df2)
#print("Concatenating Data Frames: \n",pd.concat([df1,df2], axis=0)) # axis =0 is stacking one below the other
print("Concatenating Data Frames along axis 1: \n",pd.concat([df1,df2], axis = 1))
#will consider common indices
df1
#df2


# In[38]:


#DataFrame Concatenation
df1 = pd.DataFrame({'A' : ['axe', 'art', 'ant'], 'B' : ['bat', 'bar', 'bin'], 'C': ['cap', 'cat', 'car']},index = [1,2,3])
df2 = pd.DataFrame({'D' : ['dam', 'den', 'dot'], 'E': [ 'ear', 'eat', 'egg'], 'F':['fan', 'fog', 'fat']}, index =[ 2, 3, 6])
print("Data frame 1 : \n", df1,'\n Data Frame 2: \n', df2)
#print("Concatenating Data Frames: \n",pd.concat([df1,df2], axis=0)) # axis =0 is stacking one below the other
print("Concatenating Data Frames along axis 1: \n",pd.concat([df1,df2], axis = 1))
#will consider common indices
#df1
df2


# In[96]:


#DataFrame Concatenation
df1 = pd.DataFrame({'A' : ['axe', 'art', 'ant'], 'B' : ['bat', 'bar', 'bin'], 'C': ['cap', 'cat', 'car']},index = [1,2,3])
df2 = pd.DataFrame({'D' : ['dam', 'den', 'dot'], 'E': [ 'ear', 'eat', 'egg'], 'F':['fan', 'fog', 'fat']}, index =[ 2, 3, 6])
print("Data frame 1 : \n", df1,'\n Data Frame 2: \n', df2)
print("Concatenating Data Frames: \n",pd.concat([df1,df2], axis=0)) # axis =0 is stacking one below the other
#print("Concatenating Data Frames along axis 1: \n",pd.concat([df1,df2], axis = 1))
#will consider common indices
#df1
df2


# In[97]:


#DataFrame Concatenation
df1 = pd.DataFrame({'A' : ['axe', 'art', 'ant'], 'B' : ['bat', 'bar', 'bin'], 'C': ['cap', 'cat', 'car']},index = [1,2,3])
df2 = pd.DataFrame({'D' : ['dam', 'den', 'dot'], 'E': [ 'ear', 'eat', 'egg'], 'F':['fan', 'fog', 'fat']}, index =[ 2, 3, 6])
print("Data frame 1 : \n", df1,'\n Data Frame 2: \n', df2)
print("Concatenating Data Frames: \n",pd.concat([df1,df2], axis=1)) # axis =1 is stacking one below the othe,common rows are takenr
#print("Concatenating Data Frames along axis 1: \n",pd.concat([df1,df2], axis = 1))
#will consider common indices
#df1
#df2


# In[99]:


df_concat = pd.concat([df1, df2])
df_concat = pd.concat([df1, df2], ignore_index = True)
print("Concatenation of dataframes while ignoring the index: \n", df_concat)


# In[100]:


#DataFrame Concatenation
df1 = pd.DataFrame({'A' : ['axe', 'art', 'ant'], 'B' : ['bat', 'bar', 'bin'], 'C': ['cap', 'cat', 'car']},index = [1,2,3])
df2 = pd.DataFrame({'D' : ['dam', 'den', 'dot'], 'E': [ 'ear', 'eat', 'egg'], 'F':['fan', 'fog', 'fat']}, index =[ 2, 3, 6])
print("Data frame 1 : \n", df1,'\n Data Frame 2: \n', df2)
print("Concatenating Data Frames: \n",pd.concat([df1,df2], axis=0)) # axis =0 is stacking one below the other
#print("Concatenating Data Frames along axis 1: \n",pd.concat([df1,df2], axis = 1))
#will consider common indices


# In[102]:


print(pd.concat([df1, df2]))
df_concat = pd.concat([df1, df2], ignore_index = True)
print("Concatenation of dataframes while ignoring the index: \n", df_concat)


# In[23]:


df3= pd.DataFrame({'B' : ['ball','box', 'band'],  'C' : ['cat', 'calendar', 'cone'], 'G' : ['grain', 'grape', 'goat']},
                 index=[1,4,2])
df3


# In[112]:


print(df1)
print(df3)
print("\n Inner join\n", pd.concat([df1,df3],join = 'inner')) # intersection of columns
#print"\n Outer Join\n", pd.concat([df1,df3]),join = 'outer')) # union of columns
# print("\n Default Join\n", pd.concat([df1,df3]))


# In[114]:


print(df1)
print(df3)
#print("\n Inner join\n", pd.concat([df1,df3],join = 'inner')) # intersection of columns
print("\n Outer Join\n", pd.concat([df1,df3],join = 'outer')) # union of columns
# print("\n Default Join\n", pd.concat([df1,df3]))


# In[42]:


print(df1)
print(df3)
print("\n Inner join\n", pd.concat([df1,df3],join = 'inner')) # intersection of columns
print("\n Outer Join\n", pd.concat([df1,df3],join = 'outer')) # union of columns
print("\n Default Join\n", pd.concat([df1,df3]))


# In[44]:


print(df1)
df3= pd.DataFrame({'B' : ['ball','box', 'band'],  'C' : ['cat', 'calendar', 'cone'], 'G' : ['grain', 'grape', 'goat']},
                 index=[1,4,2])
print(df3)
#print("\n Inner join\n", pd.concat([df1,df3],axis=1,join = 'inner')) # intersection of columns
print("\n Outer Join\n", pd.concat([df1,df3],join = 'outer')) # union of columns
print("\n Default Join\n", pd.concat([df1,df3]))


# In[43]:


print(df1)
df3= pd.DataFrame({'B' : ['ball','box', 'band'],  'C' : ['cat', 'calendar', 'cone'], 'G' : ['grain', 'grape', 'goat']},
                 index=[1,4,2])
print(df3)
print("\n Inner join\n", pd.concat([df1,df3],axis=1,join = 'inner')) # intersection of columns
#print("\n Outer Join\n", pd.concat([df1,df3],join = 'outer')) # union of columns
print("\n Default Join\n", pd.concat([df1,df3]))


# In[45]:


print(df1)
df3= pd.DataFrame({'B' : ['ball','box', 'band'],  'C' : ['cat', 'calendar', 'cone'], 'G' : ['grain', 'grape', 'goat']},
                 index=[1,4,2])
print(df3)
print("\n Inner join\n", pd.concat([df1,df3],axis=0,join = 'inner')) # intersection of columns
print("\n Outer Join\n", pd.concat([df1,df3],axis=0,join = 'outer')) # union of columns
print("\n Default Join\n", pd.concat([df1,df3]))


# In[116]:


# Append (similar to concat but a dataframe method)
print(df1)
print(df2)
print(df1.append(df2)) # append is same as concat stocks dataframes one below another


# In[51]:


# Merge Operations
df_stud = pd.DataFrame({'St_id': [101,102,103,104,105],'Branch': ['IT','CS','ECE','CS','Mech']})
df_fac = pd.DataFrame({'F_id' : [110,120,130,140,150 ],'F_name' : ['A', 'B', 'C', 'D', 'E'],'Branch': ['ECE','Mech', 'EEE', "IT", 'CS'] })
print("Student dataframe: \n", df_stud,'\nFaculty Dataframe :\n', df_fac)
df_merge = pd.merge(df_stud, df_fac)
print("Merged dataframe : \n ", df_merge)   #Merge on a common column
#works only if both dataframes have the specified column Default is inner
#print("Merged dataframe : \n ", pd.merge(df_stud, df_fac, on = 'Branch'))


# In[52]:


df1 =pd. DataFrame({'key' :['b', 'b', 'a', 'c', 'a', 'a'], 'data1': range(6)}) #has multiple rows labelled a and b
df2 = pd. DataFrame({'key' :['a', 'b', 'd'], 'data2': range(3)})
print("DataFrame1 : \n", df1, '\nDataFrame2 :\n', df2)

#Example of many to one merge situation
#No of rows will be 5X2 
#print("Inner Join:\n", pd.merge(df1, df2, on ='key', how = 'inner', sort=True))     # intersection of keys 
#print("Outer Join:\n", pd.merge(df1, df2, on ='key', how = 'outer', sort=True))     # union of keys
#print("Left Join:\n", pd.merge(df1, df2, on ='key', how = 'left', sort=True))     #  keys from left dataframe
#print("Right Join:\n", pd.merge(df1, df2, on ='key', how = 'right', sort=True))     # 
#Here left and outer is same and Right and Inner is same


# In[53]:


df1 =pd. DataFrame({'key' : ['b', 'b', 'a','c', 'a', 'a', 'b'], 'data1': range(7)})  #has multiple rows labelled a and b
df2 = pd.DataFrame({'key' : ['a', 'b', 'a', 'b', 'd'], 'data2': range(5)})
print("DataFrame1 : \n", df1, '\nDataFrame2 :\n', df2)

#Example of many to one merge situation
#No of rows in the dataframe = 7x4 for inner
#No of rows will be 5X2 
print("Inner Join:\n", pd.merge(df1, df2, on ='key', how = 'inner', sort=True))     # intersection of keys 
#print("Outer Join:\n", pd.merge(df1, df2, on ='key', how = 'outer', sort=True))     # union of keys
#print("Left Join:\n", pd.merge(df1, df2, on ='key', how = 'left', sort=True))     #  keys from left dataframe
#print("Right Join:\n", pd.merge(df1, df2, on ='key', how = 'right', sort=True))     # 
#Here left and outer is same and Right and Inner is same


# In[54]:


df1 =pd. DataFrame({'key' :['b', 'b', 'a', 'c', 'a', 'a'], 'data1': range(6)}) #has multiple rows labelled a and b
df2 = pd. DataFrame({'key' :['a', 'b', 'd'], 'data2': range(3)})
print("DataFrame1 : \n", df1, '\nDataFrame2 :\n', df2)

#Example of many to one merge situation
#No of rows will be 5X2 
#print("Inner Join:\n", pd.merge(df1, df2, on ='key', how = 'inner', sort=True))     # intersection of keys 
print("Outer Join:\n", pd.merge(df1, df2, on ='key', how = 'outer', sort=True))     # union of keys
#print("Left Join:\n", pd.merge(df1, df2, on ='key', how = 'left', sort=True))     #  keys from left dataframe
#print("Right Join:\n", pd.merge(df1, df2, on ='key', how = 'right', sort=True))     # 
#Here left and outer is same and Right and Inner is same


# In[55]:


f1 =pd. DataFrame({'key' :['b', 'b', 'a', 'c', 'a', 'a'], 'data1': range(6)}) #has multiple rows labelled a and b
df2 = pd. DataFrame({'key' :['a', 'b', 'd'], 'data2': range(3)})
print("DataFrame1 : \n", df1, '\nDataFrame2 :\n', df2)

#Example of many to one merge situation
#No of rows will be 5X2 
#print("Inner Join:\n", pd.merge(df1, df2, on ='key', how = 'inner', sort=True))     # intersection of keys 
#print("Outer Join:\n", pd.merge(df1, df2, on ='key', how = 'outer', sort=True))     # union of keys
print("Left Join:\n", pd.merge(df1, df2, on ='key', how = 'left', sort=True))     #  keys from left dataframe
#print("Right Join:\n", pd.merge(df1, df2, on ='key', how = 'right', sort=True))     # 
#Here left and outer is same and Right and Inner is same


# In[56]:


f1 =pd. DataFrame({'key' :['b', 'b', 'a', 'c', 'a', 'a'], 'data1': range(6)}) #has multiple rows labelled a and b
df2 = pd. DataFrame({'key' :['a', 'b', 'd'], 'data2': range(3)})
print("DataFrame1 : \n", df1, '\nDataFrame2 :\n', df2)

#Example of many to one merge situation
#No of rows will be 5X2 
#print("Inner Join:\n", pd.merge(df1, df2, on ='key', how = 'inner', sort=True))     # intersection of keys 
#print("Outer Join:\n", pd.merge(df1, df2, on ='key', how = 'outer', sort=True))     # union of keys
#print("Left Join:\n", pd.merge(df1, df2, on ='key', how = 'left', sort=True))     #  keys from left dataframe
print("Right Join:\n", pd.merge(df1, df2, on ='key', how = 'right', sort=True))     # 
#Here left and outer is same and Right and Inner is same


# In[57]:


series = pd.Series([2,3,4,5])
print(series[2])
series[2]=7.8
print(series[2])
print(series)


# In[59]:


dir( String)


# In[60]:


dir(Series)


# In[61]:


### Difference beytween axis =0 and axis =1
#Dataframe aggregation methods ignore nan values and find the sum
data = pd.DataFrame([[1,  4,  2],    
                   [2,   3, 5],     
                   [7,  4,  6]])  
print(data)
print(data.sum())    #sum by default is column sum axis =0  columnwise sum
print(data.sum(axis=1))   #roqwwise sum


# In[ ]:





# In[ ]:





# In[ ]:




