"""
Miscellaneous utilities.
"""

import inspect
import logging
import os
import sys
import time
from importlib.resources import files

import numpy as np
import pandas as pd

##


# Assets ship inside the package, so they resolve identically from a source
# checkout, a wheel install and a zipped egg.
path_assets = str(files('mito.assets'))


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
            raise TimerError("Timer is running. Use .stop() to stop it")
        self._start_time = time.perf_counter()

    def stop(self, pretty=True):
        """
        Stop the timer, and report the elapsed time.
        """
        if self._start_time is None:
            raise TimerError("Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        if pretty:
            if elapsed_time > 100:
                unit = 'min'
                elapsed_time = elapsed_time / 60
            elif elapsed_time > 1000:
                unit = 'h'
                elapsed_time = elapsed_time / 3600
            else:
                unit = 's'
            formatted_time = f'{round(elapsed_time, 2)} {unit}'

        else:
            formatted_time = round(elapsed_time, 2)

        return formatted_time

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


def extract_kwargs(args, path_tuning=None, job_id=None):
    """
    Parameters for the three pipeline entry points, from CLI arguments and, optionally,
    from one row of a tuning table.

    Only the arguments that `mito.pp.filter_cells`, `mito.pp.filter_afm` and
    `mito.tl.annotate_clones` actually accept are returned, so a stale or misspelled
    option fails here instead of deep in the call. A tuning row overrides the CLI.

    Parameters
    ----------
    args : argparse.Namespace or dict
        Parsed CLI arguments.
    path_tuning : str, optional
        Folder with `all_options_final.csv`, whose `job_id` column selects one
        parameter combination. Default is None (use `args` only).
    job_id : str, optional
        Row of the tuning table to use. Default is None.

    Returns
    -------
    dict
        {"filter_cells": {...}, "filter_afm": {...}, "annotate_clones": {...},
         "build_tree": {...}}, each holding only that function's parameters.
    """

    from mito.pp import filter_afm, filter_cells
    from mito.tl import annotate_clones, build_tree

    d = vars(args).copy() if not isinstance(args, dict) else dict(args)
    path_tuning = path_tuning if path_tuning is not None else d.get('path_tuning')
    job_id = job_id if job_id is not None else d.get('job_id')

    if path_tuning is not None and job_id is not None:
        path_options = os.path.join(path_tuning, 'all_options_final.csv')
        if not os.path.exists(path_options):
            raise ValueError(f'{path_options} does not exist!')
        options = pd.read_csv(path_options).query('job_id == @job_id')
        if options.empty:
            raise ValueError(f'job_id {job_id} is not in {path_options}')
        d.update(options.iloc[0].to_dict())

    out = {}
    for name, fun in [('filter_cells', filter_cells), ('filter_afm', filter_afm),
                      ('annotate_clones', annotate_clones), ('build_tree', build_tree)]:
        accepted = set(inspect.signature(fun).parameters) - {'afm', 'tree', 'copy'}
        out[name] = { k:v for k, v in d.items() if k in accepted and v is not None }

    return out


##


def load_mut_spectrum_ref():
    df = pd.read_csv(os.path.join(path_assets, 'weng2024_mut_spectrum_ref.csv.gz'), index_col=0)
    return df


##


def load_mt_gene_annot():
    df = pd.read_csv(os.path.join(path_assets, 'formatted_table_wobble.csv.gz'), index_col=0)
    df['mut'] = df['Position'].astype(str) + '_' + df['Reference'] + '>' + df['Variant']
    return df


##


def load_common_dbSNP():
    common = pd.read_csv(os.path.join(path_assets, 'dbSNP_MT.txt'), index_col=0, sep='\t')
    common = common['pos'].astype('str') + '_' + common['REF'] + '>' + common['ALT'].map(lambda x: x.split('|')[0])
    common = common.to_list()
    return common


##


def load_edits_REDIdb():
    edits = pd.read_csv(os.path.join(path_assets, 'REDIdb_MT.txt.gz'), index_col=0, sep='\t')
    edits = edits.query('nSamples>100')
    edits = edits['Position'].astype('str') + '_' + edits['Ref'] + '>' + edits['Ed']
    edits = edits.to_list()
    return edits


##


