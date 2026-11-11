"""Stage the data of the Liu 2026 (SU-L-005) tutorial for Zenodo.

Usage: prepare_zenodo.py --atac <loose merged.h5ad> --afm <merged_ma.h5ad> --reference-dir <dir> --out <folder>

Writes <out>/{atac_merged.h5ad, mt_merged_ma.h5ad, reference_*, checksums.md5}; the names are flat because a Zenodo record has no folders:
  * atac_merged.h5ad    the loose ATAC object, copied as it is (snapATAC2 already stores it compressed);
  * mt_merged_ma.h5ad   the merged AFM restricted to the cells of the ATAC object (the other cells are never used)
                        and written with gzip: the dense coverage layer (DP) is most of the 5.7 GB of the original file;
  * reference_*         the outputs of the tutorial, to check a run against.
"""
import argparse
import hashlib
import shutil
from pathlib import Path

import anndata
import numpy as np
import snapatac2 as snap

anndata.settings.allow_write_nullable_strings = True

ap = argparse.ArgumentParser()
ap.add_argument('--atac', required=True)
ap.add_argument('--afm', required=True)
ap.add_argument('--reference-dir', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--gzip-level', type=int, default=4)
ap.add_argument('--reference-only', action='store_true', help='rewrite the reference files and the checksums only')
args = ap.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

if not args.reference_only:
    # loose ATAC object
    shutil.copyfile(args.atac, out / 'atac_merged.h5ad')
    atac = snap.read(args.atac)
    cells = set(atac.obs_names)
    atac.close()

    # AFM: only the cells of the ATAC object
    afm = anndata.read_h5ad(args.afm, backed='r')
    keep = np.flatnonzero(afm.obs_names.isin(cells))
    print(f'AFM: {afm.n_obs:,} cells, {len(keep):,} in the ATAC object', flush=True)
    afm = afm[keep].to_memory()
    afm.write_h5ad(out / 'mt_merged_ma.h5ad', compression='gzip', compression_opts=args.gzip_level)

# reference outputs of the notebook: the cell type and the clone of each cell, and the enrichment table
joined = anndata.read_h5ad(Path(args.reference_dir) / 'atac_annotated_clones.h5ad').obs
joined[['cell_type', 'MiTo_clone', 'MT_status']].astype(str).to_csv(out / 'reference_cells.csv')
shutil.copyfile(Path(args.reference_dir) / 'clone_celltype_enrichment.csv', out / 'reference_clone_celltype_enrichment.csv')

# checksums
with open(out / 'checksums.md5', 'w') as f:
    for path in sorted(p for p in out.rglob('*') if p.is_file() and p.name not in ('checksums.md5', 'README.md', '.DS_Store')):
        digest = hashlib.md5()
        with open(path, 'rb') as g:
            for block in iter(lambda: g.read(1 << 24), b''):
                digest.update(block)
        f.write(f'{digest.hexdigest()}  {path.name}\n')
print(open(out / 'checksums.md5').read())
