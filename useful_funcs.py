import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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
