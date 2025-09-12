""""
Miscellaneous utilities.
"""

import os 
import sys
import time 
import pickle
from shutil import rmtree
import logging
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


##


_cell_filters = ['filter1', 'filter2']
_var_filters = [
    'baseline',
    'CV',
    'miller2022', 
    'weng2024',
    'MQuad', 
    'MiTo',
    'GT_enriched'
    # DEPRECATED
    # 'ludwig2019', 
    # 'velten2021', 
    # 'seurat', 
    # 'MQuad_optimized',
    # 'density',
    # 'GT_stringent'
]

# Try to find assets directory in multiple locations
def _find_assets_path():
    # First try relative path for development
    dev_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../assets')
    if os.path.exists(dev_path):
        return dev_path
    
    # Try conda environment assets directory
    import sys
    if hasattr(sys, 'prefix'):
        conda_assets = os.path.join(sys.prefix, 'assets')
        if os.path.exists(conda_assets):
            return conda_assets
    
    # Try installed package location in site-packages
    import site
    for site_dir in site.getsitepackages():
        assets_path = os.path.join(site_dir, 'assets')
        if os.path.exists(assets_path):
            return assets_path
    
    # Fallback to user site directory
    user_assets = os.path.join(site.getusersitepackages(), 'assets')
    if os.path.exists(user_assets):
        return user_assets
    
    # If nothing found, return the development path anyway
    return dev_path

path_assets = _find_assets_path()


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

    def stop(self, pretty=True):
        """
        Stop the timer, and report the elapsed time.
        """
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")
        
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
                metric = d['metric']
                min_n_var = int(d['min_n_var'])
                filter_dbs = d['filter_dbs']
                filter_moran = d['filter_moran']
                kwargs = {
                    'min_cell_number' : min_cell_number,
                    'lineage_column' : lineage_column,
                    'filtering' : filtering,
                    'bin_method' : bin_method,
                    'min_n_var' : min_n_var,
                    'ncores' : args.ncores,
                    'metric' : metric,
                    'spatial_metrics' : args.spatial_metrics,
                    'filter_moran' : filter_moran
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
                filtering_kwargs = filtering_kwargs if kwargs['filtering'] == 'MiTo' else {} 
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

            cell_filter = args.cell_filter
            kwargs = {
                'min_cell_number' : args.min_cell_number,
                'lineage_column' : args.lineage_column,
                'filtering' : args.filtering if args.filtering in _var_filters else None,
                'bin_method' : args.bin_method,
                'min_n_var' : args.min_n_var,
                'filter_dbs' : True if args.filter_dbs == 'true' else False,
                'ncores' : args.ncores,
                'metric' : args.metric,
                'spatial_metrics' : True if args.spatial_metrics == 'true' else False,
                'filter_moran' : True if args.filter_moran == 'true' else False,
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
            filtering_kwargs = filtering_kwargs if kwargs['filtering'] == 'MiTo' else {}   
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


def load_common_dbSNP():
    common = pd.read_csv(os.path.join(path_assets, 'dbSNP_MT.txt'), index_col=0, sep='\t')
    common = common['pos'].astype('str') + '_' + common['REF'] + '>' + common['ALT'].map(lambda x: x.split('|')[0])
    common = common.to_list()
    return common


##


def load_edits_REDIdb():
    edits = pd.read_csv(os.path.join(path_assets, 'REDIdb_MT.txt'), index_col=0, sep='\t')
    edits = edits.query('nSamples>100')
    edits = edits['Position'].astype('str') + '_' + edits['Ref'] + '>' + edits['Ed']
    edits = edits.to_list()
    return edits


##


def subsample_afm(afm, n_clones=3, ncells=100, freqs=np.array([.3,.3,.4])):

    assert 1-np.array(freqs).sum() <= .05
    assert len(freqs) == n_clones

    clones_sorted = afm.obs['GBC'].value_counts().index
    clones = clones_sorted[:n_clones].to_list()

    cells = []
    for clone, f in zip(clones, freqs):
        afm_clone = afm[afm.obs.query('GBC==@clone').index,:].copy()
        afm_clone = afm_clone[(afm_clone.layers['bin']>0).sum(axis=1).flatten()>2,
                              (afm_clone.layers['bin']>0).sum(axis=0).flatten()>=2]
        n_cells_clone = min(round(ncells*f), afm_clone.shape[0])
        cells.extend(
            np.random.choice(afm_clone.obs_names, n_cells_clone, replace=False).tolist()
        )

    afm_subsample = afm[cells].copy()
    
    return afm_subsample


##


def select_jobs(df, sample, n_cells, n_GBC_groups, frac_unassigned):
    """
    Select jobs, and choose one for clonal inference benchmarking
    """
    df_selected = (
        df.loc[
            (df['sample'] == sample) & \
            (df['n_cells'] >= n_cells) & \
            (df['n_GBC_groups'] >= n_GBC_groups) & \
            (df['frac_unassigned'] <= frac_unassigned)  
        ]
    )
    df_selected = (
        df_selected[[
            'job_id', 'pp_method', 'bin_method', 'af_confident_detection', 'min_cell_number', 'metric',
            'ARI', 'corr', 'NMI', 'AUPRC', 'n_cells', 'unassigned', 'n_vars', 'n_GBC_groups', 'n MiTo clone',
        ]]
    )
    df_final = df_selected.sort_values('ARI', ascending=False).head(5)

    return df_selected, df_final


##


def extract_bench_df(path):

    L = []
    for folder,_,files in os.walk(path):
        for file in files:
            if file.endswith('pickle'):
                with open(os.path.join(folder, file), 'rb') as f:
                    d = pickle.load(f)
                d['n_inferred'] = d['labels'].loc[lambda x: ~x.isna()].unique().size
                del d['labels']
                L.append(d)
    df_bench = pd.DataFrame(L)

    return df_bench


def perturb_AD_counts(a, perc_sites=.75, theta=1, add=True):
    """
    Perturb AD and .X layers of afm.
    """
    afm = a.copy()
    AD_new = afm.layers['AD'].copy()

    n_vars = AD_new.shape[1]
    n_sites = int(np.round(n_vars * perc_sites))
    idx = np.random.choice(np.arange(n_vars), n_sites)

    for i in idx:
        ad = afm.layers['AD'][:,i].toarray().flatten()
        dp = afm.layers['site_coverage'][:,i].toarray().flatten()
        p_fit = np.sum(ad) / np.sum(dp)
        p_noise = theta * p_fit
        if add:
            new_ad = ad + (dp * p_noise)
        else:
            new_ad = ad - (dp * p_noise)

        AD_new[:,i] = new_ad

    corr = np.corrcoef(afm.layers['AD'].toarray().flatten(), AD_new.toarray().flatten())[0,1]
    afm.layers['AD'] = AD_new
    afm.X = AD_new / (afm.layers['DP'].toarray() + .000001)

    return afm, corr


##