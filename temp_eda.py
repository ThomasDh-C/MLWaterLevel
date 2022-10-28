# %%
# --- Import and analyse the data ingested from NWIS
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
from useful_funcs import find_sparse_nas

# %%
# Assess temp_data
temp_df = pd.read_csv('ingested_data/temp_data.csv', index_col=0)
temp_df.index = pd.DatetimeIndex(temp_df.index)
temp_df2 = pd.read_csv('ingested_data/temp_data2.csv', index_col=0)
temp_df2.index = pd.DatetimeIndex(temp_df2.index)

# if na can only make better by looking at next nearest df
for riv in list(temp_df):
    hashmap_for_orig = temp_df[riv].isna()
    temp_df.loc[hashmap_for_orig,
                riv] = temp_df2.loc[hashmap_for_orig, riv]

# %%
plt.hist(temp_df.count())
plt.show()
# %%
na_df = find_sparse_nas(temp_df)
# %%
plt.hist(na_df['length_nas'], bins=40, range=(0, 40))
plt.show()

# %%
na_df['color'] = 'Large gap in temp data'
filter_rivs = na_df['length_nas'] < 7
na_df.loc[filter_rivs, 'color'] = 'Fixable gaps'
bad_rivs = na_df[na_df['color'] ==
                 'Large gap in temp data']['riv_names'].unique()

print(f'Filter captures {100*sum(filter_rivs)/len(filter_rivs):.2f}% of gaps')
print(
    f'Number of rivers to remove - {len(bad_rivs)} of {len(list(temp_df))}')
px.scatter(na_df, x='length_nas', y='abs_val_change',
           hover_name='riv_names', color='color')

# %%
good_rivs = set(temp_df) - set(bad_rivs)
with open('eda_results/good_temp_rivs.txt', 'w') as out:
    print(', '.join(good_rivs), file=out)

# %%
temp_df = temp_df[list(good_rivs)]
most_na_col_idx = np.argmin(temp_df.count())
most_na_riv = list(temp_df)[most_na_col_idx]

most_na_col_pre = temp_df.iloc[:, most_na_col_idx]
most_na_col_pre_filt = most_na_col_pre.isna()
# https://machinelearningmastery.com/resample-interpolate-time-series-data-python/
temp_df = temp_df.interpolate(method='linear').interpolate(
    method='linear', limit_direction='backward')
most_na_col_post = temp_df.iloc[:, most_na_col_idx]

# %%
plt.scatter(most_na_col_pre.index, most_na_col_pre, s=2, label='orig')
plt.scatter(most_na_col_post.index[most_na_col_pre_filt],
            most_na_col_post[most_na_col_pre_filt], s=2, label='interpolation added')
plt.title(f'River {most_na_riv} air temperature before & after interpolation')
plt.legend()
plt.show()
# %%
