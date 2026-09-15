import numpy as np, pandas as pd
from refined_bench import as_variants


def front_metrics(c, cols, B, gbc, GT):
    vars_, M = as_variants(c, cols, B)
    gb = gbc.loc[c.obs_names].values
    spec, clones = [], set()
    for j, v in enumerate(vars_):
        pos = M[:, j]
        if pos.sum() < 3:
            continue
        modal = pd.Series(gb[pos]).value_counts().index[0]
        spec.append(pos[gb == modal].sum()/pos.sum())
        if v in GT:
            clones.add(modal)
    return dict(n_vars=len(vars_), n_chars=int(B.shape[1]), GT_enriched_kept=len(set(vars_) & GT),
                GT_enriched_total=len(GT), call_clone_specificity=round(float(np.mean(spec)), 3) if spec else np.nan,
                GBC_clones_marked=len(clones))
