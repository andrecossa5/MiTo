""""
Miscellaneous utilities.
"""

import os 
import sys
import re
import time 
import random
import string
import pickle
import json
from shutil import rmtree
import logging
import numpy as np
import pandas as pd
import scanpy as sc


##


path_assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets')


##


logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'  # Custom format
)


##


class TimerError(Exception):
    """
    A custom exception used to report errors in use of Timer class.
    """

class Timer:
    """
    A custom Timer class.
    """
    def __init__(self):
        self._start_time = None

    def start(self):
        """
        Start a new timer.
        """
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")
        self._start_time = time.perf_counter()

    def stop(self):
        """
        Stop the timer, and report the elapsed time.
        """
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")
        elapsed_time = time.perf_counter() - self._start_time

        if elapsed_time > 100:
            unit = 'min'
            elapsed_time = elapsed_time / 60
        elif elapsed_time > 1000:
            unit = 'h'
            elapsed_time = elapsed_time / 3600
        else:
            unit = 's'

        self._start_time = None

        return f'{round(elapsed_time, 2)} {unit}'


##


def make_folder(path, name, overwrite=True):
    """
    A function to create a new {name} folder at the {path} path.
    """
    os.chdir(path)
    if not os.path.exists(name) or overwrite:
        rmtree(os.path.join(path, name), ignore_errors=True)
        os.makedirs(name)
    else:
        pass


##


def update_params(d_original, d_passed):
    for k in d_passed:
        if k in d_original:
            pass
        else:
            print(f'{k}:{d_passed[k]} kwargs added...')
        d_original[k] = d_passed[k]
        
    return d_original


##


def one_hot_from_labels(y):
    """
    My one_hot encoder from a categorical variable.
    """
    if len(y.categories) > 2:
        Y = np.concatenate(
            [ np.where(y == x, 1, 0)[:, np.newaxis] for x in y.categories ],
            axis=1
        )
    else:
        Y = np.where(y == y.categories[0], 1, 0)
    
    return Y


##


def rescale(x):
    """
    Max/min rescaling.
    """    
    if np.min(x) != np.max(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))
    else:
        return x
    

##


def ji(x, y):
    """
    Jaccard Index between two list-like objs.
    """
    x = set(x)
    y = set(y)
    ji = len(x&y) / len(x|y)

    return ji


##


def flatten_dict(d):
    result = {}
    for key, value in d.items():
        if isinstance(value, dict):
            result.update(flatten_dict(value))
        else:
            result[key] = value
    return result


##


def format_tuning(path_tuning):
    """
    Format tuning dataframe.
    """

    assert os.path.exists(path_tuning)
    options = pd.read_csv(os.path.join(path_tuning, 'all_options_final.csv'))
    metrics = pd.read_csv(os.path.join(path_tuning, 'all_metrics_final.csv'))
    df = pd.merge(
        options.pivot(index=['sample', 'job_id'], values='value', columns='option').reset_index(),
        metrics.pivot(index=['sample', 'job_id'], values='value', columns='metric').reset_index(),
        on=['sample', 'job_id']
    )
    options = options['option'].unique().tolist()
    metrics = metrics['metric'].unique().tolist()

    return df, metrics, options


##


def extract_kwargs(args, only_tree=False):
    """
    Extract preprocessing parameters from CLI and tuning information. 
    """
    
    path_tuning = args.path_tuning if hasattr(args, 'path_tuning') else None

    if path_tuning is not None and args.job_id is not None:
        
        path_options = os.path.join(path_tuning, 'all_options_final.csv')
        if os.path.exists(path_options):
            
            df_options = pd.read_csv(path_options).loc[lambda x: x['job_id'] == args.job_id]
            d = { k:v for k,v in zip(df_options['option'],df_options['value']) }

            if not only_tree:

                cell_filter = d['cell_filter']
                min_cell_number = int(d['min_cell_number'])
                lineage_column = d['lineage_column']
                filtering = d['filtering']
                bin_method = d['bin_method']
                min_n_var = int(d['min_n_var'])
                kwargs = {
                    'min_cell_number' : min_cell_number,
                    'lineage_column' : lineage_column,
                    'filtering' : filtering,
                    'bin_method' : bin_method,
                    'min_n_var' : min_n_var,
                    'path_dbSNP' : args.path_dbSNP, 
                    'path_REDIdb' : args.path_REDIdb,
                    'ncores' : args.ncores
                }
                filtering_kwargs = {
                    'min_cov' : int(d['min_cov']),
                    'min_var_quality': int(d['min_var_quality']),
                    'min_frac_negative' : float(d['min_frac_negative']),
                    'min_n_positive' : int(d['min_n_positive']),
                    'af_confident_detection' : float(d['af_confident_detection']),
                    'min_n_confidently_detected' : int(d['min_n_confidently_detected']),
                    'min_mean_AD_in_positives' : float(d['min_mean_AD_in_positives']),
                    'min_mean_DP_in_positives' : float(d['min_mean_DP_in_positives']) 
                }
                binarization_kwargs = {
                    't_prob' : float(d['t_prob']), 
                    't_vanilla' : float(d['t_vanilla']),
                    'min_AD' : int(d['min_AD']),
                    'min_cell_prevalence' : float(d['min_cell_prevalence']),
                    'k' : int(d['k']), 
                    'gamma' :  float(d['gamma']), 
                    'resample' : False
                }
                tree_kwargs = {'solver':d['solver'], 'metric':d['metric']}
            
            else:

                cell_filter = None; kwargs = None; 
                filtering_kwargs = None; binarization_kwargs = None
                tree_kwargs = {'solver':d['solver'], 'metric':d['metric']}

        else:
            raise ValueError(f'{path_options} does not exists!')

    else:

        if not only_tree:

            cell_filter = args.cell_filter if args.cell_filter != "None" else None
            kwargs = {
                'min_cell_number' : args.min_cell_number,
                'lineage_column' : args.lineage_column,
                'filtering' : args.filtering if args.filtering != "None" else None,
                'bin_method' : args.bin_method,
                'min_n_var' : args.min_n_var,
                'path_dbSNP' : args.path_dbSNP, 
                'path_REDIdb' : args.path_REDIdb,
                'ncores' : args.ncores
            }
            filtering_kwargs = {
                'min_cov' : args.min_cov,
                'min_var_quality': args.min_var_quality,
                'min_frac_negative' : args.min_frac_negative,
                'min_n_positive' : args.min_n_positive,
                'af_confident_detection' : args.af_confident_detection,
                'min_n_confidently_detected' : args.min_n_confidently_detected,
                'min_mean_AD_in_positives' : args.min_mean_AD_in_positives,
                'min_mean_DP_in_positives' : args.min_mean_DP_in_positives 
            }
            binarization_kwargs = {
                't_prob' : args.t_prob, 
                't_vanilla' : args.t_vanilla,
                'min_AD' : args.min_AD,
                'min_cell_prevalence' : args.min_cell_prevalence,
                'k' : args.k, 
                'gamma' : args.gamma, 
                'resample' : False
            }
            tree_kwargs = {'solver':args.solver, 'metric':args.metric}
        
        else:

            cell_filter = None; kwargs = None; 
            filtering_kwargs = None; binarization_kwargs = None
            tree_kwargs = {'solver':args.solver, 'metric':args.metric}

    # Set filtering kwargs to None (use defaults) filtering is not MiTo                 # FIX THIS
    filtering_kwargs = filtering_kwargs if kwargs['filtering'] == 'MiTo' else None

    return cell_filter, kwargs, filtering_kwargs, binarization_kwargs, tree_kwargs


##


def rank_items(df, groupings, metrics, weights, metric_annot):

    df_agg = df.groupby(groupings, dropna=False)[metrics].mean().reset_index()

    for metric_type in metric_annot:
        colnames = []
        for metric in metric_annot[metric_type]:
            colnames.append(f'{metric}_rescaled')
            if metric in ['n_dbSNP', 'n_REDIdb']:
                df_agg[metric] = -df_agg[metric]
            df_agg[f'{metric}_rescaled'] = (df_agg[metric] - df_agg[metric].min()) / \
                                           (df_agg[metric].max() - df_agg[metric].min())

        x = df_agg[colnames].mean(axis=1)
        df_agg[f'{metric_type} score'] = (x - x.min()) / (x.max() - x.min())

    x = np.sum(df_agg[ [ f'{k} score' for k in metric_annot ] ] * np.array([ weights[k] for k in metric_annot ]), axis=1)
    df_agg['Overall score'] = (x - x.min()) / (x.max() - x.min())
    df_agg = df_agg.sort_values('Overall score', ascending=False)

    return df_agg


##


def load_mut_spectrum_ref():
    df = pd.read_csv(os.path.join(path_assets, 'weng2024_mut_spectrum_ref.csv'), index_col=0)
    return df


##


def load_mt_gene_annot():
    df = pd.read_csv(os.path.join(path_assets, 'formatted_table_wobble.csv'), index_col=0)
    df['mut'] = df['Position'].astype(str) + '_' + df['Reference'] + '>' + df['Variant']
    return df


##
