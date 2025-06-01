#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data  = pd.read_csv(r"C:\Users\Hemanth Pusuluri\Downloads\archive (18)\US_Accidents_March23.csv")


# In[ ]:





# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


data.columns


# In[6]:


data.shape


# In[7]:


data.isnull().sum()


# In[8]:


data.info()


# In[9]:


data.describe()


# In[10]:


city_value_counts = data['City'].value_counts()

print(city_value_counts)


# In[11]:


temp_value_counts = data['Temperature(F)'].value_counts()

print(temp_value_counts)


# In[12]:


temp_value_counts.count()


# In[13]:


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
num_data = data.select_dtypes(include=numerics)

print(num_data)


# In[14]:


# Correlation heatmap based on numeric features
plt.figure(figsize=(15 ,9))
sns.heatmap(num_data.corr() , annot=True)


# In[15]:


corr = num_data.corr()

print(corr)


# In[16]:


strong = []
week = []

for col1 in corr.columns:
    for col2 in corr.columns:
        if col1 != col2:
            value = corr.loc[col1,col2]
            if abs(value)>0.7:
                strong.append((col1,col2,value))
            elif abs(value)<0.3:
                week.append((col1,col2,value))

print("Strongly correlated pairs:")
for s in strong:
    print(s)

print("\nWeakly correlated pairs:")
for w in week:
    print(w)


# In[17]:


features_with_null_values = [features for features in data.columns if data[features].isnull().sum()>0]


# In[18]:


features_with_null_values


# In[19]:


missing_percent = data[features_with_null_values].isnull().sum().sort_values(ascending=False)/len(data)*100

print(missing_percent)


# In[20]:


plt.figure(figsize=(10,7))

sns.barplot(x=missing_percent.values,y=missing_percent.index)
plt.xlabel('Missing Percentage')
plt.ylabel('Features')
plt.title('Missing Data Percentage by Feature')
plt.show()


# In[21]:


states = data['State'].value_counts()

print(states)


# In[22]:


data['State'].unique()


# In[23]:


plt.figure(figsize=(20,10))
sns.barplot(y = states, x = states.index)
plt.title('Number of Accidents in Different States')
plt.xlabel('State')
plt.ylabel('Count')
plt.show()


# In[24]:


# Analyzing the cities columns 
df = data.groupby(['City'])['City'].count().sort_values(ascending=False)
#select top 15 
df_first_15 = df.iloc[:15]    

# Showing accident in different cities by visualization
plt.figure(figsize=(20, 10))
ax = sns.barplot(x=df_first_15.index, y=df_first_15)
plt.ylabel('Accidents')
plt.title("Number of of accidents in different Cities")
plt.show()


# In[25]:


# First, make a list of the features that have 'Time' in the name but not 'Timezone'
features_with_time = [features for features in data.columns if 'Time' in features and 'Timezone' not in features]
print(features_with_time)

# Convert the features to datetime format
for feature in features_with_time:
    data[feature] = pd.to_datetime(data[feature], errors='coerce')  # Correct feature reference

# Drop rows where the year is 2023
""" The data for the year 2023 is incomplete (Only three months), therefore, 
it has been excluded from the analysis."""
df = data[data['Start_Time'].dt.year != 2023]# Get the count of accidents per year
yearly_counts = data['Start_Time'].dt.year.value_counts().sort_index()
print(yearly_counts)

# Plot Accident counts over years
plt.figure(figsize=(10, 5))
sns.barplot(
    x=yearly_counts.index,
    y=yearly_counts.values,
    palette='pastel'
)
plt.title('Count of Accidents Over Time')
plt.xlabel('Year')


# In[26]:


# How many accidents on different weekdays
data['Weekday'] = data['Start_Time'].dt.weekday  # Monday=0, Sunday=6
df3 = data.groupby(['Weekday']).size().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
ax = sns.barplot(x=df3.index, y=df3.values, palette='pastel')
plt.ylabel('Accidents')
plt.title("Number of Accidents on Different Weekdays")
plt.xticks(df3.index, ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])  # Set labels for weekdays

plt.tight_layout()
plt.show()


# In[27]:


# Count the occurrences of each weather condition
weather_counts = data['Weather_Condition'].value_counts()

# Sort the counts in descending order and select the top 20
top_weather_conditions = weather_counts.sort_values(ascending=False).head(20)

# Create the bar plot
fig, ax = plt.subplots(figsize=(8, 5))
top_weather_conditions.plot(kind='bar', ax=ax)
ax.set(title='Weather Conditions at Time of Accident Occurrence',
       xlabel='Weather',
       ylabel='Accidents Count')
plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility
plt.show()


# In[28]:


# checking the number of accident on basis of temparture in from 2017-2021
df_2021 = df[data.Start_Time.dt.year == 2021]
df_2020 = df[data.Start_Time.dt.year == 2020]
df_2019 = df[data.Start_Time.dt.year == 2019]
df_2018 = df[data.Start_Time.dt.year == 2018]
df_2017 = df[data.Start_Time.dt.year == 2017]

# Set up a 2x3 grid for the subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # (2, 3) grid for 5 subplots

# List of DataFrames for each year
dfs = [df_2021, df_2020, df_2019, df_2018, df_2017]
years = [2021, 2020, 2019, 2018, 2017]

# Flatten axes to loop through them easily
axes = axes.flatten()

# Loop through each DataFrame and plot a histogram on the corresponding axis
for i, df_year in enumerate(dfs):
    # Create the histogram for temperature
    sns.histplot(df_year['Temperature(F)'], bins=12, kde=False, ax=axes[i])

    # Set the title for each subplot
    axes[i].set_title(f'Temperature vs Accident Count in {years[i]}')
    axes[i].set_xlabel('Temperature (F)')
    axes[i].set_ylabel('Accident Count')

fig.delaxes(axes[5])  # This removes the extra empty plot in the (2,3) grid

plt.tight_layout()
plt.show()


# In[29]:


# Count the occurrences of each severity level
severity_count = data['Severity'].value_counts()

# Create the pie chart
sns.set(style="whitegrid")
plt.figure(figsize=(10, 7))
pastel_colors = sns.color_palette("pastel", len(severity_count))

plt.pie(
    severity_count,
    labels=severity_count.index,
    autopct='%1.1f%%',
    colors=pastel_colors
)
plt.legend(title='Severity Levels', loc='best', bbox_to_anchor=(1, 0.5))
plt.title('Distribution of Incident Severity')
plt.show()


# In[30]:


data.duplicated().sum()


# In[31]:


from sklearn.preprocessing import StandardScaler
#Initialize the StandardScaler
scaler = StandardScaler()

#Fit and transform the numerical features
standardized_data = scaler.fit_transform(num_data)

#Convert the standardized data back to a DataFrame
standardized_df = pd.DataFrame(standardized_data, columns=num_data.columns)

print(standardized_df)


# In[32]:


sns.scatterplot(x=data.Start_Lng,y=data.Start_Lat,data=df,size=data.Severity, hue=data.Severity)


# In[33]:


sns.scatterplot(x=df.Start_Lng,y=df.Start_Lat,data=df,size=0.001)


# In[ ]:




