
# coding: utf-8

# In[39]:


import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.image as mpimg
from PIL import Image
import time
from scipy.spatial import distance
import sys


# In[40]:


filename = sys.argv[1]
image = mpimg.imread(filename)


# In[41]:


def findClosestCentroids(centroids,mapCentroids,temp):
#     for i in range(temp.shape[0]):
#         deltas = centroids - temp[i]
#         dist_2 = np.einsum('ij,ij->i', deltas, deltas)
# #         dist_2 = np.sum((centroids - temp[i])**2, axis=1)
#         mapCentroids[i] = np.argmin(dist_2)
#         mapCentroids[i] = np.sum(np.abs(centroids - temp[i]),axis=1).argmin()
    mapCentroids = (np.sum(np.abs(temp[np.newaxis, :] - centroids[:, np.newaxis]),axis=2)).argmin(axis=0)
    return mapCentroids


# In[42]:


def updateCentroids(k,temp,mapCentroids):
    count = np.zeros(k)
    newCentroids = [[0.0 for j in range(temp.shape[1])] for i in range(k)]
    newCentroids = np.asarray(newCentroids)
    
    for i in range(temp.shape[0]):
        count[mapCentroids[i]] += 1
        newCentroids[mapCentroids[i]] += temp[i]
    newCentroids = newCentroids / count[:, np.newaxis]
    for i in range(k):
        if count[i] == 0:
            newCentroids[i] = random.choice(temp)
            print("Help!")
#     print(count)
    return newCentroids


# In[43]:


def kmeans(k,image_data,iterations):
    totalstart = time.time()
    width, height, depth = image_data.shape
    temp = np.reshape(image_data,[width*height,depth])
    centroids = [ random.choice(temp) for i in range(k) ]
    centroids = np.asarray(centroids)
    mapCentroids = np.zeros(width*height,dtype=np.int)
    for i in range(iterations):
        print("Iteration Number", i+1)
#         print(centroids)
        print("Finding Closest Centroids")
        start = time.time()
        mapCentroids = findClosestCentroids(centroids,mapCentroids,temp)
#         print(mapCentroids)
        print("Took ",time.time() - start,"seconds")
        print("Updating Centroids")
        start = time.time()
        centroids = updateCentroids(k,temp,mapCentroids)
        print("Took ",time.time() - start,"seconds")
        print("\n")
    print("Mapping to output")
    output = np.empty_like(temp)
    for i in range(temp.shape[0]):
        output[i] = centroids[mapCentroids[i]]
    print("Total time taken",time.time() - totalstart)
    return output


# In[44]:


compressed = kmeans(32,image,10)                     # kmeans(No. of colors, image, No. of iterations)


# In[45]:


compressed = np.reshape(compressed,image.shape)


# In[46]:


rescaled = (255.0 / compressed.max() * (compressed - compressed.min())).astype(np.uint8)
im = Image.fromarray(rescaled)


# In[47]:


f, axarr = plt.subplots(1,2)
axarr[0].imshow(image)
axarr[1].imshow(im)
plt.show()


# In[48]:


# im.save('test.png')
im.save('test.jpg')

