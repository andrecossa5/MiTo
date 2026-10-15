# Tutorial data

Small, real inputs for the "Loading data" tutorial, in each pre-processing tool's own
layout: 50 cells each, 10 from each of the 5 largest ground-truth clones. Written by
`benchmarks/make_doc_subsets.py`; the tables are otherwise untouched.

| Folder | Assay, tool | Source | Ground truth |
|---|---|---|---|
| `maester_MDA_clones/` | MAESTER, maegatk (nf-MiTo layout) | MDA_clones, MiTo paper data (Zenodo 10.5281/zenodo.17225334) | lentiviral barcodes (`GBC`) |
| `mtscATAC_pL1000/` | mtscATAC, mgatk (positions 1-3,000) | pL1000, MitoDrift data (figshare 10.6084/m9.figshare.31323481) | LARRY barcodes |
| `redeem_pL1000/` | ReDeeM, RedeemV (positions 1-3,000) | the same cells and deposit | LARRY barcodes |
