# %%
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd

# 3569 rivers
# 3654 datapoints max per river
river_depth_df = pd.read_csv('input_data/river_depth_data.csv', index_col=0)
# %%
non_nas = river_depth_df.count()
plt.hist(non_nas, bins=50)
plt.show()

max_days = 3654
print('Number of rivers with perfect recording:', sum(non_nas == max_days))
print('Number of rivers with 90% recordings:', sum(non_nas >= .9*max_days))

# %%
# --- Location of these rivers

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
filter_river_sites = [
    cand in good_river_sites for cand in df_raw_river_sites['site_no']]
df_raw_river_sites = df_raw_river_sites[filter_river_sites]

# %%
# https://plotly.com/python/scatter-plots-on-maps/
fig = px.scatter_geo(df_raw_river_sites,
                     lat='dec_lat_va',
                     lon='dec_long_va',
                     color="state_cd",  # which column to use to set the color of markers
                     hover_name="station_nm",  # column added to hover information
                     )
fig.show()

# %%
