import dataretrieval.nwis as nwis
import pandas as pd
from tqdm import tqdm

rec_sites_file = open('input_data/recordingsites.tsv')
lines = rec_sites_file.readlines()
all_sites = [line.split('\t')[0] for line in lines[1:]]
rec_sites_file.close()


complete_index = pd.date_range(start='2010-08-01', end='2020-08-01')
river_depth_df = pd.DataFrame(index=complete_index)
for site in tqdm(all_sites, 'Collecting river depth data for each site'):
    temp_river_depth_df = nwis.get_record(
        sites=site, service='dv', start='2010-08-01', end='2020-08-01', parameterCd=['00065'])
    if len(temp_river_depth_df) > 0:
        temp_depth_only = pd.DataFrame()
        if '00065_Mean' in list(temp_river_depth_df):
            temp_depth_only = temp_river_depth_df.loc[:, ['00065_Mean']]
            temp_depth_only = temp_depth_only.rename(
                columns={'00065_Mean': site})
        elif '00065_Maximum' in list(temp_river_depth_df) and '00065_Minimum' in list(temp_river_depth_df):
            temp_depth_only = temp_river_depth_df.loc[:, []]
            temp_depth_only[site] = (
                temp_river_depth_df['00065_Maximum'] + temp_river_depth_df['00065_Minimum'])/2
        else:
            continue

        # https://stackoverflow.com/questions/16628819/convert-pandas-timezone-aware-datetimeindex-to-naive-timestamp-but-in-certain-t
        temp_depth_only.index = temp_depth_only.index.tz_localize(None)
        # https://realpython.com/pandas-merge-join-and-concat/#pandas-join-combining-data-on-a-column-or-index
        river_depth_df = river_depth_df.join(temp_depth_only)
river_depth_df.to_csv('ingested_data/river_depth_data.csv')
