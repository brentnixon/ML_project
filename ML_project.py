#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 10:36:32 2018

@author: brentan
"""

# import dependencies
#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pprint
import pickle as pkl

#%%

# set up working directory
#%%
os.chdir('/Users/brentan/Documents/DAND/Projects/ML/ud120-projects/final_project')
print('\n')
print("CWD:", os.getcwd(), '\n')
print("DIR Contents:")
pprint.pprint(os.listdir())

#%%

# load pickle file
#%%
# create open file object
en_pkl = open("/Users/brentan/Documents/DAND/Projects/ML/ud120-projects/final_project/final_project_dataset.pkl", "rb")
# unpickle file
en_data = pkl.load(en_pkl)

# load dict into data frame
en_df = pd.DataFrame(en_data)

# make rows columns and vice versa
en_df = en_df.transpose()

# make the index values (person names) be their own column
en_df = en_df.reset_index()
en_df.head()
 #%%
 

