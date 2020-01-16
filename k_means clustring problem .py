#!/usr/bin/env python
# coding: utf-8

# In[130]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[131]:


dataset=pd.read_csv('data.kmeans_problem.csv')
dataset.info()


# In[132]:


x=dataset[['Annual Income (k$)', 'Spending Score (1-100)']]


# In[133]:


x.info()


# In[134]:


from sklearn.cluster import KMeans


# In[135]:


wcss = []


# In[136]:


for i in range(1, 11):
    kmeans = KMeans(n_clusters = i,init = 'k-means++', max_iter= 300,n_init = 10,random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11),wcss) 
plt.title('The Elbow Methode')
plt.xlabel('Number of cluster')
plt.ylabel('wcss')
plt.show()


# In[137]:


kmeans = KMeans(n_clusters = 5,init = 'k-means++', max_iter= 300,n_init = 10,random_state = 0)


# In[138]:


y_kmeans = kmeans.fit_predict(x)


# In[139]:


plt.scatter(x[y_kmeans == 0,0], x[y_kmeans==0,1], s = 15, c= 'red', label = 'Cluster_1')
plt.scatter(x[y_kmeans == 1,0], x[y_kmeans==1,1], s = 15, c= 'blue', label = 'Cluster_2')
plt.scatter(x[y_kmeans == 2,0], x[y_kmeans==2,1], s = 15, c= 'green', label = 'Cluster_3')
plt.scatter(x[y_kmeans == 3,0], x[y_kmeans==3,1], s = 15, c= 'cyan', label = 'Cluster_4')
plt.scatter(x[y_kmeans == 4,0], x[y_kmeans==4,1], s = 15, c= 'magenta', label = 'Cluster_5')


# In[ ]:


plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c ='yellow', label = 'Centroids')
plt.title('Clusters of clients')
plt.xlabel('Annual income')
plt.ylabel('spending score')
plt.legend()
plt.show()


# In[140]:


plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c ='yellow', label = 'Centroids')
plt.title('Clusters of clients')
plt.xlabel('Annual income')
plt.ylabel('spending score')
plt.legend()
plt.show()


# In[141]:


plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c ='yellow', label = 'Centroids')
plt.title('Clusters of clients')
plt.xlabel('Annual income')
plt.ylabel('spending score')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




