import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import gc
from concurrent.futures import ProcessPoolExecutor

max_days = 3654


def load_df(df_dir):
    temp_df = pd.read_csv(df_dir, index_col=0)
    temp_df.index = pd.DatetimeIndex(temp_df.index)
    return temp_df


def fft_river_df(df, var):
    # Based on code from https://www.tensorflow.org/tutorials/structured_data/time_series
    ffts = [tf.signal.rfft(df[riv]) for riv in df]
    # https://numpy.org/doc/stable/reference/generated/numpy.mean.html
    fft = np.abs(np.mean(ffts, axis=0))
    fft_lower = np.abs(np.percentile(ffts, 75, axis=0))
    fft_upper = np.abs(np.percentile(ffts, 25, axis=0))

    f_per_dataset = np.arange(0, len(fft))
    n_samples_d = len(df)
    years_per_dataset = n_samples_d/365.2524
    f_per_year = f_per_dataset/years_per_dataset
    plt.step(f_per_year, fft, c='black', label='Mean')
    plt.step(f_per_year, fft_lower, alpha=0.3,
             c='lightblue', label='25th Percentile')
    plt.step(f_per_year, fft_upper, alpha=0.3,
             c='darkblue', label='75th Percentile')
    plt.xscale('log')
    # skip first two as usually large and not shown on graph
    ymax = max(*fft[2:], *fft_lower[2:], *fft_upper[2:])+20
    plt.ylim(0, ymax)
    plt.xlim([0.05, max(plt.xlim())])
    plt.xticks([0.1, 1, 365.2524], labels=['1/Decade', '1/Year', '1/day'])
    plt.xlabel('Frequency (log scale)')
    plt.title(f'FFT of river {var}')
    plt.legend()
    plt.savefig(f'./presentation docs/fft_{var}.png', dpi=200)
    plt.show()


def find_sparse_nas(input_df):
    """Takes in a df of river cols and returns df of all contiguous na sequences"""
    idx_contig_na_days = []  # [(start_idx_na, end_idx_na), ...]
    changes_over_contig_na = []  # [(start_val, end_val)]
    riv_names = []
    for riv in list(input_df):
        vals = input_df[riv]
        w_na = np.where(np.isnan(vals))[0]
        w_notna = np.where(~np.isnan(vals))[0]
        if len(w_notna) == 0:
            idx_contig_na_days.append((0, len(vals)-1))
            changes_over_contig_na.append((0, 0))
            riv_names.append(riv)
            continue
        first_non_na, last_non_na = vals[w_notna[0]], vals[w_notna[-1]]
        temp_l = 1
        start_val, end_val = first_non_na, first_non_na
        for idx, loc in enumerate(w_na):
            if temp_l == 1 and loc != 0:
                start_val = vals[loc-1]
            if idx == len(w_na)-1 or w_na[idx+1] != loc+1:
                idx_contig_na_days.append((loc-temp_l+1, loc))
                temp_l = 1
                end_val = vals[loc+1] if loc != len(vals)-1 else last_non_na
                changes_over_contig_na.append((start_val, end_val))
                riv_names.append(riv)
            temp_l += 1
    temp = [[*a, *b, riv]
            for a, b, riv in zip(idx_contig_na_days, changes_over_contig_na, riv_names)]
    na_df = pd.DataFrame(temp, columns=[
        'start_idx_na', 'end_idx_na', 'start_val', 'end_val', 'riv_names'])
    na_df['length_nas'] = na_df['end_idx_na'] - na_df['start_idx_na'] + 1
    na_df['val_change'] = na_df['end_val'] - na_df['start_val']
    na_df['abs_val_change'] = np.absolute(na_df['val_change'])

    return na_df


def parse_data_dfs():
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
    return di_df, pi_df, ti_df, dp_df, pp_df, tp_df


def make_timeseries(all_rivs, train_rivs, validate_rivs, slide_window_riv, data_days):
    slide_train_arr = []    # train features
    slide_val_arr = []      # validate features
    slide_test_arr = []      # test features

    # https://stackoverflow.com/questions/37804279/how-can-we-use-tqdm-in-a-parallel-execution-with-joblib
#     all_slide_window_data = process_map(slide_window_riv, all_rivs,
#                                         max_workers=20, chunksize=5)
    # https://github.com/tqdm/tqdm/blob/master/tqdm/contrib/concurrent.py
    # https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor
    executor = ProcessPoolExecutor(max_workers=16) #, max_tasks_per_child=100
    # https://tqdm.github.io/docs/tqdm/#__init__
    all_slide_window_data = list(tqdm(executor.map(slide_window_riv, all_rivs, chunksize=8), total=len(all_rivs), desc="Multiprocess all sliding windows"))
    executor.shutdown()
    _ = gc.collect()
    
    for all_features in tqdm(all_slide_window_data, 'Reassigning river data to correct frame'):
        for features in all_features:
            riv = features[1]
            if riv in train_rivs:
                slide_train_arr.append(features)
            elif riv in validate_rivs:
                slide_val_arr.append(features)
            else:
                slide_test_arr.append(features)
    
    # create dfs and sort because all out of order!
    slide_cols = ['y', 'river', 'river_day']
    for idx in range(data_days):
        slide_cols.append(f'd{idx}')
    for idx in range(data_days):
        slide_cols.append(f'p{idx}')
    for idx in range(data_days):
        slide_cols.append(f't{idx}')

    X_train_df = pd.DataFrame(slide_train_arr, columns=slide_cols)
    print('Training data made')
    X_train_df.sort_values(by=['river', 'river_day'], inplace=True)
    print('Training data sorted')

    X_val_df = pd.DataFrame(slide_val_arr, columns=slide_cols)
    print('Validation data made')
    X_val_df.sort_values(by=['river', 'river_day'], inplace=True)
    print('Validation data sorted')

    X_test_df = pd.DataFrame(slide_test_arr, columns=slide_cols)
    print('Test data made')
    X_test_df.sort_values(by=['river', 'river_day'], inplace=True)
    print('Test data sorted')

    return X_train_df, X_val_df, X_test_df
