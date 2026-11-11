"""Merged mgatk tables of the 7 SU-L-005 samples (Liu et al., 2026, GSE302113) -> one MiTo AFM.

Usage: build_afm.py --mgatk-dir <folder> --out <merged_ma.h5ad>

<folder> holds the mgatk tables of the seven samples concatenated into one (A, C, G, T and coverage .txt.gz, refAllele.txt),
with the cells named {barcode}_{GSM}, so that the reads per molecule are estimated once on the pool. Every alternative allele at a site
is a candidate variant (`multi_allelic_sites=True`): the dominant allele of the pool can be an error allele.
"""
import argparse

import anndata
import mito as mt

anndata.settings.allow_write_nullable_strings = True

SAMPLES = {
    'GSM9096489': 'tumor_CD3+',
    'GSM9096490': 'tumor_CD3-',
    'GSM9096491': 'NILT_CD3+',
    'GSM9096492': 'NILT_CD3-',
    'GSM9096493': 'NILT_CD45-',
    'GSM9096494': 'PBMC_CD3+',
    'GSM9096495': 'PBMC_CD3-',
}

ap = argparse.ArgumentParser()
ap.add_argument('--mgatk-dir', required=True)
ap.add_argument('--out', required=True)
args = ap.parse_args()

afm = mt.io.make_afm(args.mgatk_dir, scLT_system='mtscATAC', pp_method='mgatk', multi_allelic_sites=True)
afm.obs['sample'] = afm.obs_names.str.rsplit('_', n=1).str[1]
afm.obs['tissue_sort'] = afm.obs['sample'].map(SAMPLES)
afm.write(args.out)
print(afm.shape, afm.uns.get('molecules'))
