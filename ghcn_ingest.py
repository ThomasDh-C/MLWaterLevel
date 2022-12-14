import dask.dataframe as dd
from multiprocessing import Pool
import threading
import requests
import concurrent.futures
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.neighbors import BallTree
import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map
import time

# AIM: find precip and temp data for all the river height monitoring stations
# https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily
precip_temp_sites = open('input_data/ghcnd-inventory.txt')
lines = precip_temp_sites.readlines()
precip_temp_sites.close()

# --- Find the features of all site active between aug 8 2010 and aug 8 2020
# key = sitenumber
# val = {lat, lon, features active between 2009 and 2021}
all_sites = {}
for line in lines:
    if line[-1] == '\n':
        line = line[:-1]
    line_entries = [val for val in line.split(' ') if len(val) > 0]
    start_date, end_date = int(line_entries[4]), int(line_entries[5])
    if start_date <= 2009 and end_date >= 2021:
        sitenumber = line_entries[0]
        if sitenumber not in all_sites:
            all_sites[sitenumber] = {'lat': float(line_entries[1]),
                                     'lon': float(line_entries[2]),
                                     'features': [line_entries[3]]}
        else:
            all_sites[sitenumber]['features'].append(line_entries[3])

# %%
# --- Find sites that have precipitation and temp data
sites_with_pt = []
for sitenumber, siteobj in all_sites.items():
    f = siteobj['features']
    if 'PRCP' in f and 'TMAX' in f and 'TMIN' in f:
        row = [sitenumber, siteobj['lat'], siteobj['lon']]
        sites_with_pt.append(row)

# %%
# --- Set up orig sites (river height) + target sites (with precip and temp data)
# Orig sites
# list of good river sites with 90% of readings not missing
river_depth_df = pd.read_csv('ingested_data/river_depth_data.csv', index_col=0)
non_nas = river_depth_df.count()
max_days = 3654
good_rivers_bool = non_nas >= .9*max_days
good_river_sites = [river for good, river in zip(
    good_rivers_bool, list(river_depth_df)) if good]
# filter all recording sites to get only good river sites
# 1. get all sites + fix
df_raw_river_sites = pd.read_csv('input_data/recordingsites.tsv', sep='\t')
rec_sites_file = open('input_data/recordingsites.tsv')
lines = rec_sites_file.readlines()
df_raw_river_sites['site_no'] = [line.split('\t')[0] for line in lines[1:]]
rec_sites_file.close()
# 2. filter
df_river_sites = df_raw_river_sites.loc[:, [
    'site_no', 'dec_lat_va', 'dec_long_va']]
filter_river_sites = [
    cand in good_river_sites for cand in df_river_sites['site_no']]
df_river_sites = df_river_sites[filter_river_sites]
cols = ['station', 'lat', 'lon']
df_river_sites = pd.DataFrame(df_river_sites.values, columns=cols)

# target sites
df_pt_sites = pd.DataFrame(sites_with_pt, columns=cols)
# %%
# --- Find the closest target site to a orig river site
# we use haversine distance and a ball tree data structure
# code based on https://towardsdatascience.com/using-scikit-learns-binary-trees-to-efficiently-find-latitude-and-longitude-neighbors-909979bd929b

# have to use radians not degrees for haversine distance
for column in df_river_sites[["lat", "lon"]]:
    rad = np.deg2rad(df_river_sites[column].astype('float').values)
    df_river_sites[f'{column}_rad'] = rad
for column in df_pt_sites[["lat", "lon"]]:
    rad = np.deg2rad(df_pt_sites[column].astype('float').values)
    df_pt_sites[f'{column}_rad'] = rad

ball = BallTree(df_pt_sites[["lat_rad", "lon_rad"]].values, metric='haversine')
distances, indices = ball.query(
    df_river_sites[["lat_rad", "lon_rad"]].values, k=2)

earth_r = 3958.8
earth_distances = distances*earth_r  # in miles
# %%
# First closest site
df_river_sites['closest_pt_site'] = [
    df_pt_sites['station'][i[0]] for i in indices]
df_river_sites['dist_to_pt_site'] = earth_distances[:, 0]
# Second closest site
df_river_sites['closest_pt_site2'] = [
    df_pt_sites['station'][i[1]] for i in indices]
df_river_sites['dist_to_pt_site2'] = earth_distances[:, 1]

# %%
max1, max2 = max(df_river_sites['dist_to_pt_site']), max(
    df_river_sites['dist_to_pt_site2'])
binwidth = 5
bins_custom = bins = [idx for idx in range(
    0, int(max(max1, max2)//binwidth+1)*binwidth+1, binwidth)]
plt.hist(df_river_sites['dist_to_pt_site'],
         bins=bins_custom, alpha=0.5, label="First closest site")
plt.hist(df_river_sites['dist_to_pt_site2'],
         bins=bins_custom, alpha=0.5, label="Second closest site")
plt.title(
    'Histogram of distances from NWIS site to GHCN site ')
plt.xlabel(
    'Distance from river station to precipitation & temperature station (miles)')
plt.ylabel('Number of stations')
plt.legend()
plt.savefig('./presentation docs/nwis_to_ghcn_distance', dpi=200)
plt.show()
# %%
# --- Query NOAA for the daily values for those PT sites
# documentation https://www.ncei.noaa.gov/data/daily-summaries/doc/GHCND_documentation.pdf

complete_index = pd.date_range(start='2010-08-01', end='2020-08-01')
# closest sites
precip_df = pd.DataFrame(index=complete_index)
temp_df = pd.DataFrame(index=complete_index)
# 2nd closest site
precip_df2 = pd.DataFrame(index=complete_index)
temp_df2 = pd.DataFrame(index=complete_index)

for row in tqdm(df_river_sites[['station', 'closest_pt_site', 'closest_pt_site2']].values):
    site, pt_site, pt_site2 = row
    # pt_data = pd.read_csv(
    #     f'https://www.ncei.noaa.gov/data/daily-summaries/access/{pt_site}.csv')
    # pt_data.index = pd.to_datetime(pt_data['DATE'], infer_datetime_format=True, low_memory=False)
    pt_data2 = pd.read_csv(
        f'https://www.ncei.noaa.gov/data/daily-summaries/access/{pt_site2}.csv', low_memory=False)
    pt_data2.index = pd.to_datetime(
        pt_data2['DATE'], infer_datetime_format=True)

    # Handle precipitation
    # pt_data[site] = pt_data['PRCP'] / 100  # measured in hundredths of inch?
    # p_data = pt_data[[site]]
    # precip_df = precip_df.join(p_data)
    pt_data2[site] = pt_data2['PRCP'] / 100  # measured in hundredths of inch?
    p_data2 = pt_data2[[site]]
    precip_df2 = precip_df2.join(p_data2)

    # Handle temperature
    # divide by 10 as measured in tenths ...
    # pt_data[site] = (pt_data['TMAX'] + pt_data['TMIN'])/2 / 10
    # t_data = pt_data[[site]]
    # temp_df = temp_df.join(t_data)
    pt_data2[site] = (pt_data2['TMAX'] + pt_data2['TMIN'])/2 / 10
    t_data2 = pt_data2[[site]]
    temp_df2 = temp_df2.join(t_data2)

# %%

# %%
# --- Export for future files to use
# df_river_sites.to_csv('ingested_data/river_to_tp_map.csv')
# precip_df.to_csv('ingested_data/precip_data.csv')
# temp_df.to_csv('ingested_data/temp_data.csv')
precip_df2.to_csv('ingested_data/precip_data2.csv')
temp_df2.to_csv('ingested_data/temp_data2.csv')
