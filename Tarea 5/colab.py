import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("fifa_players.csv")

data_cleaned = data.dropna()


features = data_cleaned[['age', 'height_cm', 'weight_kgs', 'overall_rating', 'potential', 'value_euro', 'wage_euro']]


scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)


dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(scaled_features)


data_cleaned['cluster'] = clusters


print(data_cleaned['cluster'].value_counts())

dbscan = DBSCAN(eps=1, min_samples=5)
clusters = dbscan.fit_predict(scaled_features)
data_cleaned.loc[:, 'cluster'] = clusters

sns.scatterplot(x=data_cleaned['age'], y=data_cleaned['overall_rating'], hue=data_cleaned['cluster'], palette='viridis')
plt.title('Clustering con DBSCAN')
plt.xlabel('Edad')
plt.ylabel('Valoración Global')
plt.show()