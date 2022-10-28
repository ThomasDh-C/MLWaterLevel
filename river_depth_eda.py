# %%
# --- Import and analyse the data ingested from NWIS
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd

# 3569 river stations
# 3654 datapoints max per river
river_depth_df = pd.read_csv('ingested_data/river_depth_data.csv', index_col=0)
river_depth_df.index = pd.DatetimeIndex(river_depth_df.index)

# %%
# --- Investigate how much data is missing from all the river stations
# --- Set threshold for % missing data to deem a station unusable (more than 10%)
non_nas = river_depth_df.count()
max_days = 3654
plt.hist(max_days-non_nas, bins=50)
plt.ylabel('Number of stations')
plt.xlabel('Number of missing days')
plt.title('Histogram of total missing days from stations over the 10 years')
plt.show()

print('Number of river stations with perfect recording:', sum(non_nas == max_days))
print('Number of river stations with 90% recordings:', sum(non_nas >= .9*max_days))

# %%
# --- Visualising location of these good rivers
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

# Display on map
# https://plotly.com/python/scatter-plots-on-maps/
fig = px.scatter_geo(df_raw_river_sites,
                     lat='dec_lat_va',
                     lon='dec_long_va',
                     color="state_cd",  # which column to use to set the color of markers
                     hover_name="station_nm",  # column added to hover information
                     )
fig.show()

# %%
# --- Visualising quantity of missing data for selected stations
plt.hist(max_days-river_depth_df.loc[:, good_rivers_bool].count())
plt.ylabel('Number of stations')
plt.xlabel('Number of days missing data over 10 years')
plt.title('Distribution of missing data for selected stations')
plt.show()
# %%
# --- Station 50148890 has 307 missing days - Where are they?
# Most missed days at the start
missing_days = max_days-river_depth_df.loc[:, good_rivers_bool].count()
# t
plt.plot(river_depth_df.index, river_depth_df['50148890'])
plt.ylabel('River Gauge')
plt.xlabel('Date')
plt.title('Station 50148890\'s missing data')
plt.show()

# %%
# --- Visualise missing data across all stations
dup_river_depth_df = river_depth_df.loc[:, good_rivers_bool].copy()
dup_river_depth_df = dup_river_depth_df.fillna(-1)
dup_river_depth_df[dup_river_depth_df != -1] = 1  # data exists = 1
dup_river_depth_df[dup_river_depth_df == -1] = 0  # missing data = 0

plt.imshow(dup_river_depth_df, interpolation='none')
plt.ylabel('Date (0 = Aug 2010)')
plt.xlabel('River recording station')
plt.title('Missing data shown in purple, existing in yellow')
plt.show()
# %%

good_river_depth = river_depth_df.loc[:, good_rivers_bool].copy()
riv = good_river_depth['01036390']

# return [first idx, last idx] of longest contiguous non-na
# https://stackoverflow.com/questions/41494444/pandas-find-longest-stretch-without-nan-values


def pir(x):
    # pad with np.nan
    x = np.append(np.nan, np.append(x, np.nan))
    # find where null
    w = np.where(np.isnan(x))[0]
    # diff to find length of stretch
    # argmax to find where largest stretch
    a = np.diff(w).argmax()
    # return original positions of boundary nulls
    return w[[a, a + 1]] + np.array([0, -2])


def length_contig(x):
    a, b = pir(x)
    return b-a+1


lengths_contig = good_river_depth.apply(length_contig)
plt.hist(lengths_contig, bins=50)
plt.show()
# %%

lengths = []
h_changes = []
riv_name = []
for riv in list(good_river_depth):
    x = good_river_depth[riv]
    w = np.where(np.isnan(x))[0]
    w_not = np.where(~np.isnan(x))[0]
    first_non_na, last_non_na = x[w_not[0]], x[w_not[-1]]
    temp_l = 1
    start_h, end_h = first_non_na, first_non_na
    for idx, loc in enumerate(w):
        if temp_l == 1 and loc != 0:
            start_h = x[loc-1]
        if idx == len(w)-1 or w[idx+1] != loc+1:
            lengths.append(temp_l)
            temp_l = 1
            end_h = x[loc+1] if loc != len(x)-1 else last_non_na
            h_changes.append(end_h-start_h)
            riv_name.append(riv)
        temp_l += 1
plt.hist(lengths, bins=50, range=(0, 10))
plt.show()
# %%
# Gage height, feet
plt.hist(h_changes, range=(-1, 1))
plt.show()
# %%
px.scatter(x=lengths, y=np.absolute(h_changes), hover_name=riv_name)

# %%
gaps_df = pd.DataFrame()
gaps_df['River'] = riv_name
gaps_df['Length_na'] = lengths
gaps_df['Ft_change'] = h_changes
gaps_df['Abs_ft_change'] = np.absolute(h_changes)
gaps_df['color'] = 'Resevoir/ Large gap'
filter_rivs = np.logical_and(
    gaps_df['Length_na'] < 90, gaps_df['Abs_ft_change'] < 30)
gaps_df.loc[filter_rivs, 'color'] = 'Fixable gap'
bad_rivs = gaps_df[gaps_df['color'] == 'Resevoir/ Large gap']['River'].unique()

print(f'Filter captures {100*sum(filter_rivs)/len(filter_rivs):.2f}% of gaps')
print(
    f'Number of rivers to remove - {len(bad_rivs)} of {len(list(good_river_depth))}')
px.scatter(gaps_df, x='Length_na', y='Abs_ft_change',
           color='color', hover_name='River')


# %%
great_rivs = set(good_river_depth) - set(bad_rivs)
with open('eda_results/great_depth_rivs.txt', 'w') as out:
    print(', '.join(great_rivs), file=out)

# %%
perfect_rivs_filter = good_river_depth.count() == max_days
perfect_rivs = list(good_river_depth.loc[:, perfect_rivs_filter])
# %%

with open('eda_results/perfect_depth_rivs.txt', 'w') as out:
    print(', '.join(perfect_rivs), file=out)
