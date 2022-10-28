# %%
# --- Import and analyse the data ingested from NWIS
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
from useful_funcs import find_sparse_nas

# %%
# Assess precip_data
precip_df = pd.read_csv('ingested_data/precip_data.csv', index_col=0)
precip_df.index = pd.DatetimeIndex(precip_df.index)
precip_df2 = pd.read_csv('ingested_data/precip_data2.csv', index_col=0)
precip_df2.index = pd.DatetimeIndex(precip_df2.index)

# if na can only make better by looking at next nearest df
for riv in list(precip_df):
    hashmap_for_orig = precip_df[riv].isna()
    precip_df.loc[hashmap_for_orig,
                  riv] = precip_df2.loc[hashmap_for_orig, riv]
# %%

# %%
plt.hist(precip_df.count())
plt.show()
# %%
na_df = find_sparse_nas(precip_df)
# %%
plt.hist(na_df['length_nas'], bins=40, range=(0, 40))
plt.show()

# %%
na_df['color'] = 'Large gap in precip data'
filter_rivs = na_df['length_nas'] < 7
na_df.loc[filter_rivs, 'color'] = 'Fixable gaps'
bad_rivs = na_df[na_df['color'] ==
                 'Large gap in precip data']['riv_names'].unique()

print(f'Filter captures {100*sum(filter_rivs)/len(filter_rivs):.2f}% of gaps')
print(
    f'Number of rivers to remove - {len(bad_rivs)} of {len(list(precip_df))}')
px.scatter(na_df, x='length_nas', y='abs_val_change',
           hover_name='riv_names', color='color')

# %%
good_rivs = set(precip_df) - set(bad_rivs)
with open('eda_results/good_precip_rivs.txt', 'w') as out:
    print(', '.join(good_rivs), file=out)

# %%
# %%
precip_df = precip_df[list(good_rivs)]
most_na_col_idx = np.argmin(precip_df.count())
most_na_riv = list(precip_df)[most_na_col_idx]

most_na_col_pre = precip_df.iloc[:, most_na_col_idx]
most_na_col_pre_filt = most_na_col_pre.isna()
# https://machinelearningmastery.com/resample-interpolate-time-series-data-python/
precip_df = precip_df.interpolate(method='linear').interpolate(
    method='linear', limit_direction='backward')
most_na_col_post = precip_df.iloc[:, most_na_col_idx]

# %%
plt.plot(most_na_col_pre.index, most_na_col_pre, label='orig')
plt.scatter(most_na_col_post.index[most_na_col_pre_filt],
            most_na_col_post[most_na_col_pre_filt], s=2, label='added', color='orange')
plt.title(f'River {most_na_riv} precipitation before & after filling na')
plt.legend()
plt.show()
# %%
