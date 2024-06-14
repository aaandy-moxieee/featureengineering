import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
from sklearn.cluster import KMeans

plt.figure(figsize=(14,7))
plt.style.use('ggplot')

#filepath variable and read dataset

home_filepath = r'course data\housing.csv'
home_dataset = pd.read_csv(home_filepath)

#Feature extraction
X = home_dataset.loc[:,['MedInc','Latitude','Longitude']]
X.head(20)


#Cluster feature
kmeans = KMeans(n_clusters=6,max_iter=10, n_init= 3)
X['Cluster'] = kmeans.fit_predict(X)
X['Cluster'] = X['Cluster'].astype('category')

X.head(20)

#Visualization of clusters

sns.relplot(data=X , y='Latitude', x='Longitude', hue='Cluster' , height=7)

X['MedHouseVal'] = home_dataset['MedHouseVal']

sns.catplot(data=X, x='MedHouseVal' , y='Cluster' , kind='boxen', height=7, hue='Cluster')