"""
mito.pp.filter_overdispersed_variants, overdispersion_stats, cap_candidates

The candidate stage of molecule-count assays (redeemR Filter-2): a variant is worth
testing when its per-cell counts are over-dispersed relative to one binomial, i.e.
concentrated in the cells that carry it rather than spread like sequencing error.
"""

import numpy as np
import pytest

import mito as mt
from conftest import N_CLONES, build_afm
from mito.pp.variant_filters import BLACKLIST_REGIONS, overdispersion_stats

N_MARKERS = 2 * N_CLONES       # build_afm plants the markers in the first columns


def test_keeps_the_clonal_markers_and_drops_most_of_the_noise(afm):
    markers = set(afm.var_names[:N_MARKERS])
    mt.pp.filter_overdispersed_variants(afm, ancestral=False)
    kept = set(afm.var_names)
    assert markers <= kept
    assert len(kept - markers) <= 3


def test_records_the_statistics_it_decided_on(afm):
    mt.pp.filter_overdispersed_variants(afm)
    assert {'od_stat', 'od_q'} <= set(afm.var.columns)
    assert (afm.var['od_q'] <= 0.05).all()


def test_noise_is_not_overdispersed():
    afm = build_afm(markers_per_clone=0, seed=7)
    tab = overdispersion_stats(afm)
    assert (tab['q'] > 0.05).mean() >= 0.9


def test_a_two_cell_marker_seen_once_per_cell_stays_testable():
    afm = build_afm(seed=3)
    AD = afm.layers['AD'].toarray()
    j = afm.shape[1] - 1
    AD[:, j] = 0
    AD[:2, j] = 1                              # one molecule in each of two cells
    afm.layers['AD'] = type(afm.layers['AD'])(AD)
    tab = overdispersion_stats(afm)
    assert tab['n_carriers'].iloc[j] == 2
    assert np.isfinite(tab['stat'].iloc[j])


def test_blacklisted_regions_are_removed(afm):
    start, _ = BLACKLIST_REGIONS[0]
    afm.var['pos'] = afm.var['pos'].values.copy()
    afm.var.loc[afm.var_names[0], 'pos'] = start
    name = afm.var_names[0]
    mt.pp.filter_overdispersed_variants(afm, ancestral=False)
    assert name not in afm.var_names


def test_the_cap_keeps_the_most_overdispersed_and_says_so(afm, caplog):
    stat = overdispersion_stats(afm)['stat']
    top = set(stat.sort_values(ascending=False).index[:5])
    with caplog.at_level('WARNING'):
        mt.pp.cap_candidates(afm, max_candidates=5)
    assert set(afm.var_names) == top
    assert 'max_candidates' in caplog.text


def test_the_cap_does_nothing_below_the_limit(afm):
    n = afm.shape[1]
    mt.pp.cap_candidates(afm, max_candidates=n)
    assert afm.shape[1] == n


def test_filter_afm_resolves_the_stage_from_the_assay():
    a = build_afm()
    mt.pp.filter_afm(a)
    assert a.uns['mito']['filter_afm']['params']['candidates']['method'] == 'overdispersion'
    b = build_afm(scLT_system='mtscATAC', pp_method='mgatk', targeted=False)
    mt.pp.filter_afm(b)
    assert b.uns['mito']['filter_afm']['params']['candidates']['method'] == 'thresholds'


def test_filter_afm_caps_the_candidates():
    a = build_afm()
    mt.pp.filter_afm(a, max_candidates=4, qc_alpha=1)
    flow = a.uns['mito']['filter_afm']['flow'].set_index('stage')
    assert flow.loc['candidate cap', 'n_vars_out'] == 4
    assert a.shape[1] <= 4


def test_filter_afm_rejects_an_unknown_stage(afm):
    with pytest.raises(ValueError, match='candidates'):
        mt.pp.filter_afm(afm, candidates='magic')



def test_the_signal_filter_runs_after_the_thresholds_stage_only():
    a = build_afm()
    mt.pp.filter_afm(a)
    b = build_afm()
    mt.pp.filter_afm(b, candidates='thresholds')
    assert 'signal-to-background filter' not in set(a.uns['mito']['filter_afm']['flow']['stage'])
    assert 'signal-to-background filter' in set(b.uns['mito']['filter_afm']['flow']['stage'])
    assert b.uns['mito']['filter_afm']['params']['signal']['min_snr'] == 10


def test_known_artefacts_are_filtered_for_rna_assays_only():
    a = build_afm()
    mt.pp.filter_afm(a)
    b = build_afm(scLT_system='ReDeeM', pp_method='redeem-v', targeted=False)
    mt.pp.filter_afm(b)
    assert a.uns['mito']['filter_afm']['params']['candidates']['filter_artefacts'] is True
    assert b.uns['mito']['filter_afm']['params']['candidates']['filter_artefacts'] is False
