"""
Clonal annotation: from a cell phylogeny to discrete clone labels.
"""

import logging

import numpy as np
import pandas as pd
from anndata import AnnData
from cassiopeia.data import CassiopeiaTree
from scipy import stats

from mito.pp._graph import knn_kernel
from mito.ut.provenance import record

from .cutting import evidence_cut

##


def clone_support(X: np.array, B: np.array, k: int = 30) -> np.array:
    """
    Neighbourhood corroboration of each cell's calls.

    `support[i, j]` is the weighted fraction of cell i's neighbourhood carrying character
    j, so it asks whether a cell's own call is backed by the cells around it - the only
    corroboration available when a cell has a single call.
    """

    W = knn_kernel(X, k=k, normalize=True)

    # A weighted mean of 0/1 values: a fraction, up to floating-point error
    return np.clip(W @ (B>0).astype(float), 0.0, 1.0)


##


def rescue_unassigned(
    labels: pd.Series,
    afm: AnnData,
    alpha: float = 0.01,
    min_prev_in: float = 0.5,
    max_prev_out: float = 0.1,
    min_cells: int = 5
    ) -> pd.Series:
    """
    Assign left-over cells to a clone by pooling their reads over that clone's markers.

    A cell with no significant call at any single variant can still carry one or two
    alternative reads at each of its clone's markers. For every unassigned cell and clone,
    the reads summed over the clone's marker set are compared with the reads expected from
    each variant's own error rate (Poisson, one-sided). The cell joins its best clone only
    if that clone is significant at `alpha` / n_clones AND no other clone reaches `alpha`,
    so reads supporting several clones (e.g. only a shared ancestral marker) leave it
    unassigned.

    Parameters
    ----------
    labels : pd.Series
        Clone labels over afm.obs_names, "unassigned" where none.
    afm : AnnData
        Filtered AFM, genotyped, with per-variant "error_rate" in .var.
    alpha : float, optional
        Significance of the pooled-read test. Default is 0.01.
    min_prev_in, max_prev_out : float, optional
        A marker of a clone is carried by at least `min_prev_in` of its cells and at most
        `max_prev_out` of the other assigned cells. Defaults are 0.5 and 0.1.
    min_cells : int, optional
        Minimum clone size to define a marker set. Default is 5.

    Returns
    -------
    pd.Series
        Updated labels.
    """

    lab = labels.copy()
    B = afm.layers['bin'].toarray()>0
    AD = afm.layers['AD'].toarray().astype(float)
    DP = afm.layers['DP']
    COV = (DP.toarray() if hasattr(DP, 'toarray') else np.asarray(DP)).astype(float)
    bg = afm.var['error_rate'].values

    ok = (lab!='unassigned').values
    markers = {}
    for L in pd.unique(lab[ok]):
        inL = (lab==L).values
        if inL.sum()<min_cells:
            continue
        prev_in = B[inL].mean(0)
        prev_out = B[ok & ~inL].mean(0) if (ok & ~inL).any() else np.zeros(B.shape[1])
        M = np.flatnonzero((prev_in>=min_prev_in) & (prev_out<=max_prev_out))
        if M.size:
            markers[L] = M

    un = np.flatnonzero((lab=='unassigned').values)
    if not markers or un.size == 0:
        return lab

    Ls = list(markers)
    obs = np.column_stack([AD[un][:,markers[L]].sum(1) for L in Ls])
    exp = np.column_stack([(COV[un][:,markers[L]]*bg[markers[L]][None,:]).sum(1) for L in Ls])
    pv = np.where(obs>0, stats.poisson.sf(obs-1, np.maximum(exp, 1e-12)), 1.0)
    best = pv.argmin(1)
    p_best = pv[np.arange(un.size), best]
    p_second = np.sort(pv, axis=1)[:,1] if len(Ls)>1 else np.ones(un.size)
    assign = (p_best<=alpha/len(Ls)) & (p_second>alpha)
    lab.iloc[un[assign]] = np.array(Ls, dtype=object)[best[assign]]
    logging.info(f'Rescue: {int(assign.sum())} cells assigned from pooled reads')

    return lab


##


def annotate_clones(
    tree: CassiopeiaTree,
    afm: AnnData,
    alpha: float = 0.01,
    min_in: float = 0.25,
    ratio: float = 3.0,
    self_min_in: float = 0.85,
    min_cells: int = 5,
    min_supported: int = 2,
    one_sided: bool = False,
    tau: float = 0.25,
    rescue: bool = False,
    key_added: str = 'MiTo_clone',
    copy: bool = False
    ) -> AnnData | None:
    """
    Infer discrete clone labels from a MT-SNVs phylogeny.

    Two steps:

    1. **evidence cut** (`mito.tl.evidence_cut`): the tree is cut where, and only where,
       the characters pay for the split - a clade is subdivided only if at least
       `min_supported` of its children carry markers specific against their siblings.
    2. **abstention**: a cell keeps its label only if one of its calls is corroborated by
       its neighbourhood (weighted fraction of carriers >= `tau`). Cells that fail are
       labelled "unassigned" rather than forced into the nearest clade.

    This replaces the grid search over similarity percentile, mutation enrichment and
    merging thresholds of the published annotator, which needed per-dataset overrides to
    find any solution at all.

    Parameters
    ----------
    tree : CassiopeiaTree
        Cell phylogeny from `mito.tl.build_tree`.
    afm : AnnData
        The filtered AFM the tree was built from.
    alpha, min_in, ratio, self_min_in, min_cells, min_supported, one_sided
        Cut parameters. See `mito.tl.evidence_cut`.
    tau : float, optional
        Neighbourhood corroboration required to keep a label; 0 disables abstention.
        Default is 0.25.
    rescue : bool, optional
        Try to assign left-over cells from pooled reads over clone markers
        (`mito.tl.rescue_unassigned`). Default is False.
    key_added : str, optional
        Column added to afm.obs and tree.cell_meta. Default is "MiTo_clone".
    copy : bool, optional
        Return a modified copy of `afm` instead of updating it in place. Default is False.

    Returns
    -------
    AnnData | None
        Annotated AFM if `copy` is True, otherwise None. Adds afm.obs[`key_added`] and
        afm.obs["clone_support"], the same two columns in tree.cell_meta, and
        .uns["mito"]["annotate_clones"], whose "clones" table gives each clone's size, the
        tree node it was cut at, and its marker characters.
    """

    afm = afm.copy() if copy else afm
    if 'bin' not in afm.layers:
        raise ValueError('annotate_clones needs genotypes in afm.layers["bin"]: run mito.pp.filter_afm.')

    cells = list(afm.obs_names)
    B = afm.layers['bin'].toarray()>0
    X = afm.X.toarray()

    labels = evidence_cut(
        tree, B.astype(float), cells, alpha=alpha, min_in=min_in, ratio=ratio,
        self_min_in=self_min_in, min_cells=min_cells, min_supported=min_supported,
        one_sided=one_sided
    )
    labels = labels.loc[cells]

    support = clone_support(X, B)
    best_support = np.where(B, support, 0.0).max(1)
    if tau:
        labels = labels.mask(best_support<tau, 'unassigned')

    if rescue:
        labels = rescue_unassigned(labels, afm)

    # Rename the clades that survived the cut: tree node names are unreadable, and are
    # not stable across runs. Clones are numbered by size, largest first.
    sizes = labels[labels!='unassigned'].value_counts()
    node_of = {f'MT-{i}':node for i, node in enumerate(sizes.index, start=1)}
    labels = labels.map({v:k for k, v in node_of.items()}).fillna('unassigned')

    # One row per clone: size, the tree node it was cut at, and its marker characters
    ok = (labels!='unassigned').values
    rows = []
    for L in sorted(pd.unique(labels[ok]), key=lambda x: int(x.split('-')[1])):
        inL = (labels==L).values
        prev_in = B[inL].mean(0)
        prev_out = B[ok & ~inL].mean(0) if (ok & ~inL).any() else np.zeros(B.shape[1])
        rows.append(dict(
            clone=str(L), n_cells=int(inL.sum()), node=str(node_of[L]),
            markers=';'.join(afm.var_names[(prev_in>=0.5) & (prev_out<=0.1)])
        ))
    clones = pd.DataFrame(rows).set_index('clone') if rows else pd.DataFrame()

    afm.obs[key_added] = pd.Categorical(labels.values)
    afm.obs['clone_support'] = best_support
    tree.cell_meta[key_added] = labels.reindex(tree.cell_meta.index).values
    tree.cell_meta['clone_support'] = pd.Series(best_support, index=cells).reindex(tree.cell_meta.index).values
    record(afm, 'annotate_clones', {
        'params' : {
            'alpha':alpha, 'min_in':min_in, 'ratio':ratio, 'self_min_in':self_min_in,
            'min_cells':min_cells, 'min_supported':min_supported, 'one_sided':one_sided,
            'tau':tau, 'rescue':rescue
        },
        'clones' : clones,
        'frac_assigned' : float(ok.mean())
    })

    logging.info(
        f'Clonal annotation: {int(labels[ok].nunique())} clones, '
        f'{ok.sum()}/{len(cells)} cells assigned ({ok.mean():.1%})'
    )

    return afm if copy else None


##
