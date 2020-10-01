m#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
# Enter path to the SEVIR data location
DATA_PATH    = '../../sevir_data_copy/data'
CATALOG_PATH = '../../sevir_data_copy/CATALOG.csv' 

# On some Linux systems setting file locking to false is also necessary:
import os
os.environ["HDF5_USE_FILE_LOCKING"]='FALSE' 


# In[14]:


import os
import h5py # needs conda/pip install h5py
import matplotlib.pyplot as plt


# In[8]:


# !pip install --upgrade scikit-image


# In[ ]:





# In[9]:


# !pip install ipympl --user


# In[10]:


def read_data( sample_event, img_type, data_path=DATA_PATH ):
    """
    Reads single SEVIR event for a given image type.
    
    Parameters
    ----------
    sample_event   pd.DataFrame
        SEVIR catalog rows matching a single ID
    img_type   str
        SEVIR image type
    data_path  str
        Location of SEVIR data
    
    Returns
    -------
    np.array
       LxLx49 tensor containing event data
    """
    fn = sample_event[sample_event.img_type==img_type].squeeze().file_name
    fi = sample_event[sample_event.img_type==img_type].squeeze().file_index
    with h5py.File(data_path + '/' + fn,'r') as hf:
        data=hf[img_type][fi] 
    return data

# ir069 = read_data(sample_event, 'ir069')

# # plot a frame from each img_type
# fig,axs = plt.subplots(1,1,figsize=(10,5))
# frame_idx = 30
# axs.imshow(ir069[:,:,frame_idx]), axs.set_title('IR 6.9')


# In[ ]:



import matplotlib.patches as patches
from math import sqrt
import cv2
import h5py # needs conda/pip install h5py
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
from joblib import Parallel, delayed
from tqdm import tqdm

def read_data( sample_event, img_type, data_path=DATA_PATH ):
    """
    Reads single SEVIR event for a given image type.
    
    Parameters
    ----------
    sample_event   pd.DataFrame
        SEVIR catalog rows matching a single ID
    img_type   str
        SEVIR image type
    data_path  str
        Location of SEVIR data
    
    Returns
    -------
    np.array
       LxLx49 tensor containing event data
    """
    fn = sample_event[sample_event.img_type==img_type].squeeze().file_name
    fi = sample_event[sample_event.img_type==img_type].squeeze().file_index
    if(type(fn) is not pd.Series):
        with h5py.File(data_path + '/' + fn,'r') as hf:
            data=hf[img_type][fi] 
        return data
    else:
        return None

def lght_to_grid(data):
    """
    Converts SEVIR lightning data stored in Nx5 matrix to an LxLx49 tensor representing
    flash counts per pixel per frame

    Parameters
    ----------
    data  np.array
       SEVIR lightning event (Nx5 matrix)

    Returns
    -------
    np.array 
       LxLx49 tensor containing pixel counts
    """
    FRAME_TIMES = np.arange(-120.0,125.0,5) * 60 # in seconds
    out_size = (48,48,len(FRAME_TIMES))
    if data.shape[0]==0:
        return np.zeros(out_size,dtype=np.float32)

    # filter out points outside the grid
    x,y=data[:,3],data[:,4]
    m=np.logical_and.reduce( [x>=0,x<out_size[0],y>=0,y<out_size[1]] )
    data=data[m,:]
    if data.shape[0]==0:
        return np.zeros(out_size,dtype=np.float32)

    # Filter/separate times
    # compute z coodinate based on bin locaiton times
    t=data[:,0]
    z=np.digitize(t,FRAME_TIMES)-1
    z[z==-1]=0 # special case:  frame 0 uses lght from frame 1

    x=data[:,3].astype(np.int64)
    y=data[:,4].astype(np.int64)

    k=np.ravel_multi_index(np.array([y,x,z]),out_size)
    n = np.bincount(k,minlength=np.prod(out_size))
    return np.reshape(n,out_size).astype(np.float32)

def read_lght_data( sample_event, data_path=DATA_PATH ):
    """
    Reads lght data from SEVIR and maps flash counts onto a grid  

    Parameters
    ----------
    sample_event   pd.DataFrame
        SEVIR catalog rows matching a single ID
    data_path  str
        Location of SEVIR data

    Returns
    -------
    np.array 
       LxLx49 tensor containing pixel counts for selected event

    """
    fn = sample_event[sample_event.img_type=='lght'].squeeze().file_name
    id = sample_event[sample_event.img_type=='lght'].squeeze().id
    if(type(fn) is pd.Series):
        return None
    with h5py.File(data_path + '/' + fn,'r') as hf:
        data      = hf[id][:] 
    return lght_to_grid(data)

# Read catalog
catalog = pd.read_csv(CATALOG_PATH,parse_dates=['time_utc'],low_memory=False)

# Desired image types
img_types = set(['vis','ir069','ir107','vil'])

# Group by event id, and filter to only events that have all desired img_types
events = catalog.groupby('id').filter(lambda x: img_types.issubset(set(x['img_type']))).groupby('id')
event_ids = list(events.groups.keys())
print('Found %d events matching' % len(event_ids),img_types)

#for event_id in event_ids[-15:-10]:
def create_lightning_cluster(event_id):
    # Grab a sample event and view catalog entries
    sample_event = events.get_group(event_id)
    #print(event_id)
    lght = read_lght_data(sample_event)
    ir069 = read_data(sample_event, 'ir069')
    if ir069 is None:
        return None
    if lght is None:
        return None
    # include lightning counts in plot
    #fig,axs = plt.subplots(1,2,figsize=(14,5))
    from sklearn.cluster import KMeans
    cluster_data = []
    rescaled_lightning_imgs = []
    for i in range(0,49):
        #fig,axs = plt.subplots(1,3,figsize=(14,5))
        frame_idx = i
        lightning_img = cv2.resize(lght[:,:,frame_idx], (192,192), interpolation=cv2.INTER_NEAREST)
#         axs[0].imshow(ir069[:,:,frame_idx]), axs[1].set_title('IR 6.9')
#         axs[1].imshow(np.absolute(lght[:,:,frame_idx] - lght[:,:,frame_idx-1])), axs[1].set_title('Lightning Diff')
#         axs[2].imshow(lightning_img)
        thresh_points = np.argwhere(lightning_img > np.percentile(lightning_img.flatten(), 1)/5)
        if(len(thresh_points)==0):
            cluster_data.append({})
            rescaled_lightning_imgs.append(lightning_img)
            continue
        #print(optimal_number_of_clusters(thresh_points))
        optimal_kmeans = None
        optimal_points = 1
        for num_cluster_points in range(1,25):
            kmeans = KMeans(num_cluster_points, n_init=10,max_iter=300, n_jobs=8)
            kmeans.fit(thresh_points)
            optimal_kmeans = kmeans
            optimal_points = num_cluster_points
            if(kmeans.inertia_ <12000):
                break
        #axs[2].set_title('Lightning with ' + str(optimal_points) +' clusters ')
        #print(len(thresh_points))
        min_max_dict = {}
        for point, label in zip(thresh_points, optimal_kmeans.labels_):
            if(label not in min_max_dict.keys()):
                min_max_dict[label] = {}
                min_max_dict[label]["count"] = 1
                min_max_dict[label]["min_x"] = point[1]
                min_max_dict[label]["max_x"] = point[1]
                min_max_dict[label]["min_y"] = point[0]
                min_max_dict[label]["max_y"] = point[0]
                min_max_dict["num_clusters"] = len(optimal_kmeans.labels_)
            else:
                min_max_dict[label]["count"] += 1
                if(point[1] < min_max_dict[label]["min_x"]):
                    min_max_dict[label]["min_x"] = point[1]
                if(point[0] < min_max_dict[label]["min_y"]):
                    min_max_dict[label]["min_y"] = point[0]
                if(point[1] > min_max_dict[label]["max_x"]):
                    min_max_dict[label]["max_x"] = point[1]
                if(point[0] > min_max_dict[label]["max_y"]):
                    min_max_dict[label]["max_y"] = point[0]
        cluster_data.append(min_max_dict)
        rescaled_lightning_imgs.append(lightning_img)
    return event_id, ir069, cluster_data, rescaled_lightning_imgs
#         for label in range(num_cluster_points):
#             start_x, start_y = min_max_dict[label]["min_x"], min_max_dict[label]["min_y"]
#             length_x, length_y = min_max_dict[label]["max_x"] - start_x, min_max_dict[label]["max_y"] - start_y
#             #print((start_x, start_y, length_x, length_y))
#             rect = patches.Rectangle((start_x-1, start_y-1),length_x+2,length_y+2,linewidth=1, edgecolor='r',facecolor='none')
#             axs[2].add_patch(rect)
    
        #axs[3].plot(sum_of_squares)
    # DIFFERENCES IN LIGHTNING

data = Parallel(n_jobs=20)(delayed(create_lightning_cluster)(event_ids[i]) for i in tqdm(range(len(event_ids))))


# In[ ]:


#import pickle
#with open("cluster_lght_data.pickle", "wb") as pkl_file:
import joblib
joblib.dump(data, "cluster_lght_data_2.pickle")


# In[ ]:





# In[ ]:




