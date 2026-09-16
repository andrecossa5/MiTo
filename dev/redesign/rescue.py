"""Assign unassigned cells to clones by pooling their reads over each clone's marker set."""
import numpy as np
import pandas as pd
from scipy import stats


def clone_markers(labels, B, min_prev_in=0.5, max_prev_out=0.1, min_cells=5):
    """
    Marker set of every label: characters carried by >= min_prev_in of the label's cells and
    <= max_prev_out of the other assigned cells. Nested labels share their parent's markers.
    labels: array of labels over B's rows ('unassigned' for none); B: cells x characters (bool).
    """
    lab = np.asarray(labels, dtype=object)
    ok = lab != 'unassigned'
    out = {}
    for L in pd.unique(lab[ok]):
        inL = lab == L
        if inL.sum() < min_cells:
            continue
        prev_in = B[inL].mean(0)
        prev_out = B[ok & ~inL].mean(0) if (ok & ~inL).any() else np.zeros(B.shape[1])
        M = np.flatnonzero((prev_in >= min_prev_in) & (prev_out <= max_prev_out))
        if M.size:
            out[L] = M
    return out


def read_rescue(labels, B, AD, COV, background, alpha=0.05, **marker_kwargs):
    """
    Read-level rescue: counts a cell's reads ONLY against candidate clones' marker sets.

    A cell joins label L if (i) it has alt reads on L's markers, (ii) those reads beat the
    markers' background (Poisson, one-sided, p <= alpha), and (iii) it has NO alt read on the
    markers of any other label. Reads are never turned into per-cell calls that enter the tree:
    on MDA_PT and MDA_lung, looser per-cell genotypes put single reads on other clones'
    markers too and broke the tree (clones recovered 11 -> 2-4).
    Labels sharing markers (nested clades) both show the reads, so such cells stay unassigned.

    Returns (new labels, per-cell table of the rescue decisions).
    """
    lab = np.asarray(labels, dtype=object).copy()
    markers = clone_markers(lab, B > 0, **marker_kwargs)
    if not markers:
        return lab, pd.DataFrame()
    Ls = list(markers)
    un = np.flatnonzero(lab == 'unassigned')
    AD_u = AD[un].astype(float); COV_u = COV[un].astype(float)
    obs = np.column_stack([AD_u[:, markers[L]].sum(1) for L in Ls])
    exp = np.column_stack([(COV_u[:, markers[L]]*background[markers[L]][None, :]).sum(1) for L in Ls])
    with_reads = obs > 0
    only_one = with_reads.sum(1) == 1
    best = obs.argmax(1)
    p_best = np.where(only_one, stats.poisson.sf(obs[np.arange(len(un)), best] - 1, np.maximum(exp[np.arange(len(un)), best], 1e-12)), 1.0)
    assign = only_one & (p_best <= alpha)
    lab[un[assign]] = np.array(Ls, dtype=object)[best[assign]]
    detail = pd.DataFrame(dict(cell_index=un, labels_with_reads=with_reads.sum(1), best_label=np.array(Ls, dtype=object)[best],
                               reads_best=obs[np.arange(len(un)), best], p_best=p_best, assigned=assign))
    return lab, detail


def pooled_rescue(labels, B, AD, COV, background, alpha=0.01, **marker_kwargs):
    """
    Pooled-evidence rescue of cells the clone caller left unassigned.

    A cell with no significant call at any single variant can still carry one or two alt
    reads at each of its clone's markers. For every unassigned cell and label L, the reads
    summed over L's markers are compared with the reads expected from each variant's own
    background (Poisson, one-sided: P(sum AD >= observed | sum COV * background)). The cell
    joins its best label only if that label is significant at alpha / n_labels AND no other
    label reaches alpha, so reads supporting several labels (e.g. only a shared parent
    marker) leave it unassigned.

    Returns (new labels, per-cell table of the rescue decisions).
    """
    lab = np.asarray(labels, dtype=object).copy()
    markers = clone_markers(lab, B > 0, **marker_kwargs)
    if not markers:
        return lab, pd.DataFrame()
    Ls = list(markers)
    un = np.flatnonzero(lab == 'unassigned')
    AD_u = AD[un].astype(float); COV_u = COV[un].astype(float)
    obs = np.column_stack([AD_u[:, markers[L]].sum(1) for L in Ls])
    exp = np.column_stack([(COV_u[:, markers[L]]*background[markers[L]][None, :]).sum(1) for L in Ls])
    pv = np.where(obs > 0, stats.poisson.sf(obs - 1, np.maximum(exp, 1e-12)), 1.0)
    best = pv.argmin(1)
    p_best = pv[np.arange(len(un)), best]
    p_second = np.sort(pv, axis=1)[:, 1] if len(Ls) > 1 else np.ones(len(un))
    assign = (p_best <= alpha/len(Ls)) & (p_second > alpha)
    lab[un[assign]] = np.array(Ls, dtype=object)[best[assign]]
    detail = pd.DataFrame(dict(cell_index=un, best_label=np.array(Ls, dtype=object)[best], p_best=p_best, p_second=p_second,
                               reads_best=obs[np.arange(len(un)), best], expected_best=exp[np.arange(len(un)), best], assigned=assign))
    return lab, detail
