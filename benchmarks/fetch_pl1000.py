"""
Fetch the pL1000 ground-truth benchmark (Gao, Weng et al., 2026; MitoDrift paper).

Same cells profiled by ReDeeM (raw redeemV output and the redeemR Filter-2 matrix) and
by mtscATAC (mgatk), with per-cell LARRY barcodes as ground truth, plus MitoDrift's own
tree and the redeemR reference tables needed to mirror its preprocessing.

    figshare 10.6084/m9.figshare.31323481 (CC BY 4.0)
    github.com/chenweng1991/redeemR, branch redeemR2.0, data/

Every figshare file is checked against the md5 the figshare API reports, and is skipped
if already present and intact, so the script can be re-run safely.

Usage: python benchmarks/fetch_pl1000.py [--out data_test/bench/pL1000] [--all]
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.request

ARTICLE = 31323481
REDEEMR = 'https://raw.githubusercontent.com/chenweng1991/redeemR/redeemR2.0/data'

# Files used by the benchmarks. --all adds the other redeemV stringency levels and the
# R objects, which nothing here reads.
FILES = [
    # ground truth, MitoDrift output, redeemR Filter-2 matrix
    'README.md',
    'pL1000_barcode_annot_final.csv',
    'pL1000_barcode_annot_cleaned.csv',
    'tree_annotated.newick',
    'pL1000_full_mut_dat.csv',
    # raw redeemV output
    'RawGenotypes.Sensitive.StrandBalance',
    'QualifiedTotalCts',
    'StrandBiaseBlackList',
    'VariantsGTSummary.RDS',
    'chrM_refAllele.txt',
    # mgatk output, and the variant calls MitoDrift used on it
    'pL1000_mgatk.A.txt.gz',
    'pL1000_mgatk.C.txt.gz',
    'pL1000_mgatk.G.txt.gz',
    'pL1000_mgatk.T.txt.gz',
    'pL1000_mgatk.coverage.txt.gz',
    'pL1000_mgatk.depthTable.txt',
    'pL1000_mgatk.variant_stats.tsv.gz',
    'pL1000_mgatk.mutations.fr2_mc3.tsv',
]
REDEEMR_FILES = ['ATACWhite.rda', 'RNAWhite.rda', 'mitomap_freq.rda']


def md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def fetch(url, dest):
    tmp = dest + '.part'
    with urllib.request.urlopen(url) as r, open(tmp, 'wb') as f:
        while block := r.read(1 << 22):
            f.write(block)
    os.replace(tmp, dest)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='data_test/bench/pL1000')
    p.add_argument('--all', action='store_true')
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with urllib.request.urlopen(f'https://api.figshare.com/v2/articles/{ARTICLE}') as r:
        meta = json.load(r)
    files = {f['name']: f for f in meta['files']}
    wanted = list(files) if args.all else FILES

    # Two files share the name chrM_refAllele.txt (redeemV and mgatk): keep the first,
    # and check that they are identical rather than assume it.
    seen = set()
    failed = []
    for name in wanted:
        if name in seen:
            continue
        seen.add(name)
        f = files[name]
        dest = os.path.join(args.out, name)
        if os.path.exists(dest) and md5(dest) == f['computed_md5']:
            print(f'ok      {name}', flush=True)
            continue
        print(f'fetch   {name} ({f["size"]/1e6:.1f} MB)', flush=True)
        fetch(f['download_url'], dest)
        if md5(dest) != f['computed_md5']:
            failed.append(name)
            print(f'MD5 MISMATCH {name}', flush=True)
        else:
            print(f'ok      {name}', flush=True)

    dup = [f for f in meta['files'] if f['name'] == 'chrM_refAllele.txt']
    if len({f['computed_md5'] for f in dup}) > 1:
        print('NOTE: the two chrM_refAllele.txt files differ; the first was kept', flush=True)

    for name in REDEEMR_FILES:
        dest = os.path.join(args.out, name)
        if not os.path.exists(dest):
            print(f'fetch   redeemR {name}', flush=True)
            fetch(f'{REDEEMR}/{name}', dest)
        print(f'ok      redeemR {name}', flush=True)

    if failed:
        sys.exit(f'md5 mismatch: {failed}')
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
