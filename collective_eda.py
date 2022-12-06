from sklearn.model_selection import train_test_split
import plotly.express as px
import pandas as pd

perfect_depth_rivs = []
great_depth_rivs = []
good_precip_rivs = []
good_temp_rivs = []

with open('eda_results/perfect_depth_rivs.txt', 'r') as in_file:
    perfect_depth_rivs = in_file.readline()[:-1].split(', ')

with open('eda_results/great_depth_rivs.txt', 'r') as in_file:
    great_depth_rivs = in_file.readline()[:-1].split(', ')

with open('eda_results/good_precip_rivs.txt', 'r') as in_file:
    good_precip_rivs = in_file.readline()[:-1].split(', ')

with open('eda_results/good_temp_rivs.txt', 'r') as in_file:
    good_temp_rivs = in_file.readline()[:-1].split(', ')

with open('eda_results/perfect_depth_rivs.txt', 'r') as in_file:
    perfect_depth_rivs = in_file.readline()[:-1].split(', ')

final_rivs = set(great_depth_rivs) & set(
    good_precip_rivs) & set(good_temp_rivs)
perfect_final_rivs = set(final_rivs) & set(perfect_depth_rivs)
imperfect_final_rivs = final_rivs - perfect_final_rivs

# with open('eda_results/perfect_final_rivs.txt', 'w') as out:
#     print(', '.join(perfect_final_rivs), file=out)
# with open('eda_results/imperfect_final_rivs.txt', 'w') as out:
#     print(', '.join(imperfect_final_rivs), file=out)

train_rivs, validate_rivs = train_test_split(
    list(imperfect_final_rivs), test_size=0.33, random_state=42)

# filter all recording sites to get only good river sites
# 1. get all sites + fix
nwis_raw_sites = pd.read_csv('input_data/recordingsites.tsv', sep='\t')
nwis_sites_file = open('input_data/recordingsites.tsv')
lines = nwis_sites_file.readlines()
nwis_raw_sites['site_no'] = [line.split('\t')[0] for line in lines[1:]]
nwis_sites_file.close()
nwis_sites = nwis_raw_sites.loc[:, [
    'site_no', 'dec_lat_va', 'dec_long_va']]
filter_river_sites = [cand in final_rivs for cand in nwis_sites['site_no']]
nwis_sites = nwis_sites[filter_river_sites]
cols = ['station', 'lat', 'lon']
nwis_sites = pd.DataFrame(nwis_sites.values, columns=cols)

type_of_site = []
for station in nwis_sites['station']:
    if station in train_rivs:
        type_of_site.append('train')
    elif station in validate_rivs:
        type_of_site.append('validate')
    elif station in perfect_final_rivs:
        type_of_site.append('test')
nwis_sites['type'] = type_of_site

fig = px.scatter_geo(nwis_sites, lat='lat', lon='lon', color="type")
fig.show()
