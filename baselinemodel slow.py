# %%
from sklearn.model_selection import train_test_split
import pandas as pd
from useful_funcs import load_df  # , fft_river_df
import matplotlib.pyplot as plt
import plotly.express as px
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from time import time

# -- River height --
river_depth_df = load_df('ingested_data/river_depth_data.csv')
river_depth_df = river_depth_df.interpolate(method='linear').interpolate(
    method='linear', limit_direction='backward')

# -- Precipitation --
precip_df = load_df('ingested_data/precip_data.csv')
precip_df2 = load_df('ingested_data/precip_data2.csv')

# if na can only make better by looking at next nearest df
for riv in list(precip_df):
    hashmap_for_orig = precip_df[riv].isna()
    precip_df.loc[hashmap_for_orig,
                  riv] = precip_df2.loc[hashmap_for_orig, riv]
precip_df = precip_df.interpolate(method='linear').interpolate(
    method='linear', limit_direction='backward')

# -- Temperature --
temp_df = load_df('ingested_data/temp_data.csv')
temp_df2 = load_df('ingested_data/temp_data2.csv')

# if na can only make better by looking at next nearest df
for riv in list(temp_df):
    hashmap_for_orig = temp_df[riv].isna()
    temp_df.loc[hashmap_for_orig,
                riv] = temp_df2.loc[hashmap_for_orig, riv]
temp_df = temp_df.interpolate(method='linear').interpolate(
    method='linear', limit_direction='backward')

# %%
imperfect_final_rivs = []  # for train and validate
with open('eda_results/imperfect_final_rivs.txt', 'r') as in_file:
    imperfect_final_rivs = in_file.readline()[:-1].split(', ')
perfect_final_rivs = []  # for final test
with open('eda_results/perfect_final_rivs.txt', 'r') as in_file:
    perfect_final_rivs = in_file.readline()[:-1].split(', ')


# df name shortening:
# letter 1 = d (river depth), p (precipitation), t (temperature)
# letter 2 = i (imperfect), p (perfect)
di_df = river_depth_df.loc[:, imperfect_final_rivs]
pi_df = precip_df.loc[:, imperfect_final_rivs]
ti_df = temp_df.loc[:, imperfect_final_rivs]
dp_df = river_depth_df.loc[:, perfect_final_rivs]
pp_df = precip_df.loc[:, perfect_final_rivs]
tp_df = temp_df.loc[:, perfect_final_rivs]

# %%
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
train_rivs, validate_rivs = train_test_split(
    imperfect_final_rivs, test_size=0.33, random_state=42)
# %%
dtrain_df = di_df.loc[:, train_rivs]
normed_01_d_df = (dtrain_df - dtrain_df.min()) / dtrain_df.max()
normed_01_d_df.iloc[:, :5].plot()
plt.ylabel('Normalised depth from 0-1')
plt.xlabel('Time')
plt.show()
mean_std_d_df = (dtrain_df - dtrain_df.mean()) / dtrain_df.std()
# %%
# Check frequencies in data
# not clear there is pattern any apart from yearly
# fft_river_df(dtrain_df, 'depth')
# fft_river_df(pi_df.loc[:, train_rivs], 'precipitation')
# fft_river_df(ti_df.loc[:, train_rivs], 'temperature')
# %%
# Make training, validate, testing matrices for df in next step
X_train_arr = []    # train features
y_train_arr = []    # train target
X_val_arr = []      # validate features
y_val_arr = []      # validate target
X_test_arr = []      # test features
y_test_arr = []      # test target
data_days = 20
future_days = 14
rolling_period = data_days+future_days

# Considered this but 2x as fast for much uglier code so changed back
# https://stackoverflow.com/questions/47483579/how-to-use-numpy-as-strided-from-np-stride-tricks-correctly/47483615#47483615
# def custom_rolling(a, length):
#     return np.lib.stride_tricks.as_strided(a, (len(a) - (length-1), length), a.strides*2, writeable=False)
# consider from tqdm.contrib.concurrent import process_map in future so don't have to wait 30 min for data
for riv in tqdm(imperfect_final_rivs + perfect_final_rivs, 'Generate test data with sliding window'):
    if riv in imperfect_final_rivs:
        d = di_df[riv].rolling(rolling_period)
        p = pi_df[riv].rolling(rolling_period)
        t = ti_df[riv].rolling(rolling_period)
        # sliding window through depth, pressure and temp
        for idx, (rd, rp, rt) in enumerate(zip(d, p, t)):
            # rolling window initially not full
            if len(rd) != rolling_period:
                continue
            feature_data = np.concatenate(
                (rd.values[:data_days], rp.values[:data_days], rt.values[:data_days]))
            features = [riv, idx] + list(feature_data)
            if riv in train_rivs:
                X_train_arr.append(features)
                y_train_arr.append(rd.values[-1])
            else:
                X_val_arr.append(features)
                y_val_arr.append(rd.values[-1])
    else:
        d = dp_df[riv].rolling(rolling_period)
        p = pp_df[riv].rolling(rolling_period)
        t = tp_df[riv].rolling(rolling_period)
        # sliding window through depth, pressure and temp
        for idx, (rd, rp, rt) in enumerate(zip(d, p, t)):
            # rolling window initially not full
            if len(rd) != rolling_period:
                continue
            feature_data = np.concatenate(
                (rd.values[:data_days], rp.values[:data_days], rt.values[:data_days]))
            features = [riv, idx] + list(feature_data)
            X_test_arr.append(features)
            y_test_arr.append(rd.values[-1])

# %%
X_cols = ['river', 'river_day']
X_cols.append([f'd{idx}' for idx in range(20)])
X_cols.append([f'p{idx}' for idx in range(20)])
X_cols.append([f't{idx}' for idx in range(20)])
X_train_df = pd.DataFrame(X_train_arr, cols=X_cols)  # train features
y_train_df = pd.DataFrame(y_train_arr, cols=['y'])  # train target
print('Done')
X_val_df = pd.DataFrame(X_val_arr, cols=X_cols)     # validate features
y_val_df = d.DataFrame(y_val_arr, cols=['y'])       # validate target
print('Done')
X_test_df = pd.DataFrame(X_test_arr, cols=X_cols)   # test features
y_test_df = d.DataFrame(y_train_arr, cols=['y'])    # test target
# %%
