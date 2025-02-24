#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D


# In[2]:


# Load dataset
file_path = r"C:\Users\SHIVANI\Downloads\data (5) (1) (1) (1) (1) (1).xlsx"
df = pd.read_excel(file_path, engine="openpyxl")



# In[14]:


df.head()


# In[15]:


df.info


# In[16]:


df.columns


# In[17]:


df.shape


# In[18]:


duplicate_count = df.duplicated().sum()
duplicate_count


# In[19]:


missing_values_count = df.isnull().sum()

missing_values_count


# In[20]:


df.describe()


# In[21]:


unique_counts = df.nunique()
unique_values = {col: df[col].unique() for col in df.columns}
unique_counts, unique_values


# In[3]:


# Data Cleaning

df.columns = df.columns.str.strip()
df['time'] = pd.to_datetime(df['time'])
numeric_cols = [
    "Cyclone_Inlet_Gas_Temp", "Cyclone_Material_Temp", "Cyclone_Outlet_Gas_draft",
    "Cyclone_cone_draft", "Cyclone_Gas_Outlet_Temp", "Cyclone_Inlet_Draft"
]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
df.fillna(method='ffill', inplace=True)
df.drop_duplicates(inplace=True)
df = df[(np.abs(stats.zscore(df[numeric_cols])) < 3).all(axis=1)]
scaler = RobustScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
df['Temp_Diff'] = df['Cyclone_Inlet_Gas_Temp'] - df['Cyclone_Gas_Outlet_Temp']
df['Draft_Diff'] = df['Cyclone_Inlet_Draft'] - df['Cyclone_Outlet_Gas_draft']
print("Initial dataset shape:", df.shape)
df.columns = df.columns.str.strip()
df['time'] = pd.to_datetime(df['time'])

drop_cols = ['Unnamed: 0']
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

numeric_cols = [
    "Cyclone_Inlet_Gas_Temp", "Cyclone_Material_Temp", "Cyclone_Outlet_Gas_draft",
    "Cyclone_cone_draft", "Cyclone_Gas_Outlet_Temp", "Cyclone_Inlet_Draft"
]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

def categorize_temp(temp):
    if temp < 400:
        return 'Low'
    elif 400 <= temp < 700:
        return 'Medium'
    else:
        return 'High'

df['Temp Category'] = df['Cyclone_Inlet_Gas_Temp'].apply(categorize_temp)


def categorize_pressure(pressure):
    if pressure < -180:
        return 'Very Low'
    elif -180 <= pressure < -160:
        return 'Low'
    elif -160 <= pressure < -140:
        return 'Moderate'
    else:
        return 'High'

df['Pressure Category'] = df['Cyclone_Inlet_Draft'].apply(categorize_pressure)

df = df[(np.abs(stats.zscore(df[numeric_cols])) < 3).all(axis=1)]
print("Dataset shape after outlier removal:", df.shape)

scaler = RobustScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])


df['Temp_Diff'] = df['Cyclone_Inlet_Gas_Temp'] - df['Cyclone_Gas_Outlet_Temp']
df['Draft_Diff'] = df['Cyclone_Inlet_Draft'] - df['Cyclone_Outlet_Gas_draft']

print(df[['time', 'Cyclone_Inlet_Gas_Temp', 'Cyclone_Material_Temp',
       'Cyclone_Outlet_Gas_draft', 'Cyclone_cone_draft',
       'Cyclone_Gas_Outlet_Temp', 'Cyclone_Inlet_Draft',
       'Temp Category', 'Pressure Category']].head())


# In[23]:


df.head()


# In[24]:


df.columns


# In[4]:


# visualizaion
plt.figure(figsize=(8, 5))
sns.histplot(df['Cyclone_Inlet_Gas_Temp'], bins=20, kde=True, color='blue')
plt.title('Distribution of Cyclone Inlet Gas Temperature')
plt.xlabel('Cyclone Inlet Gas Temp')
plt.ylabel('Count')
plt.show()


# In[25]:


plt.figure(figsize=(10, 6))
for col in numeric_cols:
    sns.histplot(df[col], kde=True, label=col, alpha=0.5, bins=30)
plt.title("Histograms of Numeric Features (Including New Columns)")
plt.legend()
plt.savefig("combined_histograms.png")
plt.show()


# In[31]:


numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

plt.figure(figsize=(10, 6))
for col in numeric_cols:
    sns.histplot(df[col], kde=True, label=col, alpha=0.5, bins=30)
plt.title("Histograms of Numeric Features (Including New Columns)")
plt.legend()
plt.savefig("combined_histograms.png")
plt.show()


# In[26]:


plt.figure(figsize=(12, 6))
sns.boxplot(data=df[['Cyclone_Inlet_Gas_Temp', 'Cyclone_Gas_Outlet_Temp', 'Cyclone_Material_Temp']])
plt.title('Boxplot of Cyclone Temperatures')
plt.xlabel('Temperature Variables')
plt.ylabel('Value')
plt.show()


# In[27]:


plt.figure(figsize=(12, 6))
sns.lineplot(x=df['time'], y=df['Cyclone_Inlet_Gas_Temp'], label='Inlet Gas Temp')
sns.lineplot(x=df['time'], y=df['Cyclone_Gas_Outlet_Temp'], label='Outlet Gas Temp')
plt.title('Time Series of Gas Temperatures')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.legend()
plt.show()


# In[28]:


plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['Cyclone_Inlet_Gas_Temp'], y=df['Cyclone_Gas_Outlet_Temp'], alpha=0.6)
plt.title('Scatter Plot of Inlet vs Outlet Gas Temp')
plt.xlabel('Cyclone Inlet Gas Temp')
plt.ylabel('Cyclone Outlet Gas Temp')
plt.show()


# In[30]:


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['Cyclone_Inlet_Gas_Temp'], df['Cyclone_Gas_Outlet_Temp'], df['Temp_Diff'], c='red', alpha=0.7)
ax.set_xlabel('Inlet Gas Temp')
ax.set_ylabel('Outlet Gas Temp')
ax.set_zlabel('Temp Difference')
plt.title('3D Scatter Plot of Temperature Differences')
plt.show()


# In[34]:


# Line Plot of Draft Differences Over Time
plt.figure(figsize=(12, 6))
sns.lineplot(x=df['time'], y=df['Draft_Diff'])
plt.title('Time Series of Draft Differences')
plt.xlabel('Time')
plt.ylabel('Draft Difference')
plt.show()


# In[37]:


iso_forest = IsolationForest(contamination=0.01, random_state=42)
df['Anomaly'] = iso_forest.fit_predict(df[numeric_cols])
df['Anomaly'] = df['Anomaly'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')
if 'Anomaly' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(x=df['Anomaly'])
    plt.title('Anomaly Counts')
    plt.xlabel('Anomaly Status')
    plt.ylabel('Count')
    plt.show()
else:
    print("Error: 'Anomaly' column not found in the DataFrame.")


# In[ ]:





# In[38]:


iso_forest = IsolationForest(contamination=0.01, random_state=42)
df['Anomaly'] = iso_forest.fit_predict(df[numeric_cols])
df['Anomaly'] = df['Anomaly'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')
g = sns.FacetGrid(df, col="Anomaly")
g.map_dataframe(sns.histplot, x="Cyclone_Inlet_Gas_Temp")
plt.title("Distribution of Inlet Gas Temperature by Anomaly")
plt.show()


# In[39]:


pca = PCA(n_components=2)
df_pca = pca.fit_transform(df[numeric_cols])
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df_pca[:, 0], y=df_pca[:, 1], hue=df['Anomaly'])
plt.title("PCA Scatter Plot")
plt.savefig("pca_scatter.png")
plt.show()

iso_forest = IsolationForest(contamination=0.01, random_state=42)
df['Anomaly'] = iso_forest.fit_predict(df[numeric_cols])
df['Anomaly'] = df['Anomaly'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')
df.to_excel("processed_cyclone_data.xlsx", index=False)



# In[42]:


from pptx import Presentation


# In[28]:


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')


df['time'] = pd.to_datetime(df['time'], errors='coerce')
df['time_numeric'] = (df['time'] - pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')


clean_df = df.dropna(subset=['time_numeric', 'Cyclone_Inlet_Gas_Temp', 'Cyclone_Gas_Outlet_Temp'])

x = clean_df['time_numeric']
y = clean_df['Cyclone_Inlet_Gas_Temp']
z = clean_df['Cyclone_Gas_Outlet_Temp']

ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none')
ax.set_xlabel('Time')
ax.set_ylabel('Inlet Gas Temperature')
ax.set_zlabel('Outlet Gas Temperature')
plt.title('3D Surface Plot of Pressure and Temperature Over Time')
plt.show()


# In[ ]:




