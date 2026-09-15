"""
Variant QC that replaces Moran I and the hard four-gamete deletion.

Two ideas, both local:

  local_autocorrelation  Are a variant's carriers enriched among each other's
                         nearest neighbours, on a cosine AF graph that leaves the
                         variant itself out? Moran I averages autocorrelation over
                         the whole population, so a variant compact in one clade
                         and absent elsewhere scores the same as one weakly spread
                         everywhere. Here the statistic is computed over carriers
                         only, against a permutation null of random carrier sets
                         of the same size on the same graph. A recurrent variant is
                         compact in every clade where it arose, so it passes;
                         scattered noise does not.

  split_recurrent        MT-SNVs can recur, so a four-gamete conflict between two
                         locally compact variants is more likely homoplasy than
                         error. The carriers of a conflicting variant are grouped
                         by the OTHER markers they carry: carriers in different
                         lineages share none. Each group large enough becomes its
                         own character, which restores infinite sites for the
                         split characters instead of deleting a real marker.
"""
import numpy as np
from itertools import combinations
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

from caller import _cosine_D


def local_autocorrelation(X, B, k_max=30, k_min=5, n_perm=200, seed=0):
    """
    Per-variant local autocorrelation of carriers on a leave-one-out AF graph.

    Returns (enrichment, pvalue). enrichment is the mean share of a carrier's k
    nearest neighbours that are carriers too, over the share expected at the
    variant's prevalence. k scales with the carrier count (a 10-cell clone cannot
    fill a 30-cell neighbourhood), and the permutation null uses the same k.
    """
    rng = np.random.default_rng(seed)
    n, m = X.shape
    G = X @ X.T
    sq = (X**2).sum(1)
    enr = np.full(m, np.nan)
    pv = np.ones(m)
    for j in range(m):
        C = np.flatnonzero(B[:, j] > 0)
        nc = C.size
        if nc < 3:
            continue
        k = int(np.clip(nc, k_min, k_max))
        D = _cosine_D(X, G, sq, j)
        np.fill_diagonal(D, np.inf)
        idx = np.argpartition(D, k, axis=1)[:, :k]
        member = np.zeros(n, bool)
        member[C] = True
        obs = member[idx[C]].mean()
        null = np.empty(n_perm)
        for p in range(n_perm):
            R = rng.choice(n, nc, replace=False)
            mem = np.zeros(n, bool); mem[R] = True
            null[p] = mem[idx[R]].mean()
        enr[j] = obs/max((nc-1)/(n-1), 1e-12)
        pv[j] = (1 + (null >= obs).sum())/(1 + n_perm)
    return enr, pv


def neighbourhood_concordance(X, B, k=15, n_perm=200, min_placed=3.0, alpha=0.05, seed=0, n_rounds=1):
    """Iterated wrapper: variants rejected in a round no longer place cells in the next.

    A carrier placed only through a noise call is placed in a random lineage, so it
    turns a sole marker's carriers into apparently scattered ones. Removing
    rejected variants from the placement profiles returns those carriers to
    'unplaced'. Removal is monotone (a rejected variant never places again), so the
    loop cannot oscillate; it stops early when no decision changes.
    """
    placing = np.ones(B.shape[1], bool)
    prev = None
    for _ in range(n_rounds):
        res = _concordance_round(X, B, placing, k, n_perm, min_placed, alpha, seed)
        dec = res[0]
        if prev is not None and (dec == prev).all():
            break
        prev = dec
        placing &= dec != 'reject'
    return res


def _concordance_round(X, B, placing, k, n_perm, min_placed, alpha, seed):
    """
    Rejection test for scattered variants on the kNN graph, safe for sole markers.

    For variant j the graph is built WITHOUT j, from allele frequencies masked to
    called genotypes (so a cell is placed only by calls it actually has, not by
    background AF). Each carrier i gets:

      w_i   placement weight: mean cosine similarity to its k neighbours in that
            graph. A carrier with no other call has an empty profile, w_i = 0.
      N_i   its neighbourhood, including itself.

    Concordance = weighted mean overlap |N_i ∩ N_i'| / (k+1) over carrier pairs,
    weights w_i w_i'. Markers (sole-marked clones excepted), subclone markers and
    recurrent variants have carriers inside one or a few lineages, so their
    neighbourhoods overlap; scattered noise carriers sit in unrelated lineages.

    Decision:
      undecided  placed mass sum(w) over carriers < min_placed: the graph holds no
                 evidence about j (sole marker) -> KEEP
      keep       concordance significantly above random carrier sets of the same
                 size (permutation, p <= alpha)
      reject     otherwise: placed carriers are as scattered as random cells

    Returns DataFrame-ready arrays: decision, concordance, null mean, p, placed mass.
    """
    rng = np.random.default_rng(seed)
    Bb = B > 0
    n, m = Bb.shape
    Xc = np.where(Bb, X, 0.0)
    Xc[:, ~placing] = 0.0                                  # rejected variants do not place cells
    G = Xc @ Xc.T
    sq = (Xc**2).sum(1)
    decision = np.empty(m, dtype=object)
    conc = np.full(m, np.nan); null_mean = np.full(m, np.nan); pv = np.ones(m); mass = np.zeros(m)
    eye = np.arange(n)

    def stat(S, A, w):
        if S.size < 2:
            return np.nan
        As = A[S]
        ov = (As @ As.T).toarray()
        ww = np.outer(w[S], w[S])
        np.fill_diagonal(ww, 0.0)
        den = ww.sum()
        return float((ov*ww).sum()/den/(k+1)) if den > 0 else np.nan

    for j in range(m):
        C = np.flatnonzero(Bb[:, j])
        if C.size < 2:
            decision[j] = 'undecided'
            continue
        D = _cosine_D(Xc, G, sq, j)
        np.fill_diagonal(D, np.inf)
        idx = np.argpartition(D, k, axis=1)[:, :k]
        sim = 1.0 - np.take_along_axis(D, idx, 1)
        w = np.clip(sim.mean(1), 0.0, 1.0)
        w[np.sqrt(np.maximum(sq - Xc[:, j]**2, 0)) <= 1e-12] = 0.0      # no other call: unplaced
        mass[j] = w[C].sum()
        if mass[j] < min_placed:
            decision[j] = 'undecided'
            continue
        rows = np.repeat(eye, k+1)
        cols_ = np.column_stack([eye, idx]).ravel()
        A = csr_matrix((np.ones(rows.size), (rows, cols_)), shape=(n, n))
        obs = stat(C, A, w)
        null = np.array([stat(rng.choice(n, C.size, replace=False), A, w) for _ in range(n_perm)])
        null = null[np.isfinite(null)]
        conc[j], null_mean[j] = obs, null.mean() if null.size else np.nan
        pv[j] = (1 + (null >= obs).sum())/(1 + null.size)
        decision[j] = 'keep' if pv[j] <= alpha else 'reject'
    return decision, conc, null_mean, pv, mass


def enrichment_test(X, B, k=15, min_odds=4.0, bf_reject=10.0, min_informative=0.5, n_rounds=1):
    """
    Prevalence-robust rejection test for scattered variants: odds ratio + Bayes factor.

    For variant j, on the kNN graph built WITHOUT j from allele frequencies masked
    to called genotypes, pool over carriers i and their neighbours k:

      s_ik   similarity to the neighbour (0 when i has no other call)
      N      sum s_ik                  similarity-weighted neighbour trials
      a      sum s_ik * [k carries j]  similarity-weighted carrier neighbours
      p      (|C|-1)/(n-1)             prevalence of j

    Effect size is the ODDS RATIO of the local share q = a/N against p. Unlike q/p it
    stays meaningful at any prevalence (q/p is capped at 1/p, so a marker of a clone
    holding half the cells can never reach q/p = 2).

    Decision compares two hypotheses on the neighbour counts:
      H0  q = p                        (scattered: neighbours carry j at random)
      H1  odds(q) = min_odds * odds(p) (marker-like enrichment)
    Under H0 neighbours ARE independent draws, so neighbour-level binomial counts are
    the right likelihood for the hypothesis being rejected; the correlation among
    neighbours only exists under H1, for variants that are kept anyway.

      reject  informativeness >= min_informative AND BF(H0:H1) >= bf_reject
      keep    otherwise

    informativeness = carriers' mean neighbour similarity / all cells' mean: near 0 for
    sole markers (carriers have no position without j), so they are never rejected.
    Returns decision, informativeness, odds ratio, log10 BF(H0:H1), prevalence.
    """
    Bb = B > 0
    n, m = Bb.shape
    placing = np.ones(m, bool)
    prev = None
    for _ in range(n_rounds):
        Xc = np.where(Bb, X, 0.0)
        Xc[:, ~placing] = 0.0
        G = Xc @ Xc.T
        sq = (Xc**2).sum(1)
        dec = np.empty(m, dtype=object); inf_ = np.full(m, np.nan); odds = np.full(m, np.nan)
        lbf = np.full(m, np.nan); pj = np.full(m, np.nan)
        for j in range(m):
            C = np.flatnonzero(Bb[:, j])
            if C.size < 2 or C.size >= n - 1:
                dec[j] = 'keep' if C.size < 2 else 'reject'
                continue
            D = _cosine_D(Xc, G, sq, j)
            np.fill_diagonal(D, np.inf)
            idx = np.argpartition(D, k, axis=1)[:, :k]
            S = np.clip(1.0 - np.take_along_axis(D, idx, 1), 0.0, 1.0)
            S[np.sqrt(np.maximum(sq - Xc[:, j]**2, 0)) <= 1e-12] = 0.0
            ms = S.mean(1)
            inf_[j] = ms[C].mean()/max(ms.mean(), 1e-12)
            Sc = S[C]
            N = Sc.sum(); a = (Sc*Bb[idx[C], j]).sum()
            p = (C.size - 1)/(n - 1); pj[j] = p
            q = (a + 0.5)/(N + 1.0)
            odds[j] = (q/(1-q))/(p/(1-p))
            p1 = min_odds*p/(1 - p + min_odds*p)
            lbf[j] = (a*np.log(p/p1) + (N - a)*np.log((1-p)/(1-p1)))/np.log(10)
            reject = (inf_[j] >= min_informative) and (lbf[j] >= np.log10(bf_reject))
            dec[j] = 'reject' if reject else 'keep'
        if prev is not None and (dec == prev).all():
            break
        prev = dec
        placing &= dec != 'reject'
    return dec, inf_, odds, lbf, pj


def _incompatible(x, y, min_gamete):
    return min(int((x & y).sum()), int((x & ~y).sum()), int((~x & y).sum())) > min_gamete


def split_recurrent(B, min_gamete=10, min_group=5):
    """
    Split conflicting variants into one character per lineage of carriers.

    Carriers of variant j are linked when they share another called variant; the
    connected components are candidate lineages. Carriers with no other call are
    attached to the component they share the most variants' neighbourhood with --
    here simply the largest, since they carry no information to place them.
    A variant is split only if >= 2 components reach `min_group` cells; calls in
    smaller components are kept with the largest group.

    Returns (B_split, origin) where origin[c] is the source column of character c.
    """
    Bb = B > 0
    n, m = Bb.shape
    conflicted = np.zeros(m, bool)
    for i, j in combinations(range(m), 2):
        if _incompatible(Bb[:, i], Bb[:, j], min_gamete):
            conflicted[i] = conflicted[j] = True
    cols, origin = [], []
    for j in range(m):
        C = np.flatnonzero(Bb[:, j])
        if not conflicted[j] or C.size < 2*min_group:
            cols.append(Bb[:, j]); origin.append(j)
            continue
        other = np.delete(Bb[C], j, axis=1).astype(np.int32)
        share = other @ other.T                       # carriers sharing another variant
        ncomp, lab = connected_components(csr_matrix(share > 0), directed=False)
        has_other = other.sum(1) > 0
        sizes = np.bincount(lab[has_other], minlength=ncomp) if has_other.any() else np.zeros(ncomp, int)
        big = np.flatnonzero(sizes >= min_group)
        if big.size < 2:
            cols.append(Bb[:, j]); origin.append(j)
            continue
        largest = big[np.argmax(sizes[big])]
        for g in big:
            col = np.zeros(n, bool)
            in_g = (lab == g) & has_other
            if g == largest:
                in_g |= ~np.isin(lab, big) | ~has_other   # unplaceable and tiny groups
            col[C[in_g]] = True
            cols.append(col); origin.append(j)
    return np.column_stack(cols).astype(np.int8), np.array(origin)


def split_recurrent_graph(X, B, k=15, min_gamete=10, min_group=5, max_profile_similarity=0.5):
    """
    Split conflicting variants by kNN-graph lineage of their carriers.

    Linking carriers through shared variants fails as soon as any broad variant
    (present in several clones) exists: it connects every lineage into one
    component (measured on MDA_clones: no variant split). Here two carriers are
    linked only if one is among the other's k nearest neighbours on the cosine AF
    graph that leaves variant j out -- cells of different lineages are not each
    other's neighbours, whatever variants they happen to share.

    Groups of >= min_group carriers become characters; carriers in smaller groups
    follow the group holding most of their carrier neighbours, else the largest.
    """
    Bb = B > 0
    n, m = Bb.shape
    conflicted = np.zeros(m, bool)
    for i, j in combinations(range(m), 2):
        if _incompatible(Bb[:, i], Bb[:, j], min_gamete):
            conflicted[i] = conflicted[j] = True
    G = X @ X.T
    sq = (X**2).sum(1)
    cols, origin, info = [], [], {}
    for j in range(m):
        C = np.flatnonzero(Bb[:, j])
        if not conflicted[j] or C.size < 2*min_group:
            cols.append(Bb[:, j]); origin.append(j)
            continue
        D = _cosine_D(X, G, sq, j)
        np.fill_diagonal(D, np.inf)
        kk = min(k, n-1)
        idx = np.argpartition(D, kk, axis=1)[:, :kk]
        pos = -np.ones(n, int); pos[C] = np.arange(C.size)
        nb = pos[idx[C]]                                  # carrier-neighbours, -1 if not a carrier
        r, cc = np.nonzero(nb >= 0)
        A = csr_matrix((np.ones(r.size), (r, nb[r, cc])), shape=(C.size, C.size))
        ncomp, lab = connected_components(A, directed=False)
        sizes = np.bincount(lab, minlength=ncomp)
        big = np.flatnonzero(sizes >= min_group)
        if big.size < 2:
            cols.append(Bb[:, j]); origin.append(j)
            continue
        # LINEAGE GUARD: groups inside one clone fragment the graph through dropout
        # but carry the same other markers (measured on MDA: a clean single-clone
        # marker split 59/43). Merge groups whose mean profile on the OTHER
        # variants is similar; only lineage-distinct groups become characters.
        Xo = np.delete(X[C], j, axis=1)
        cent = np.vstack([Xo[lab == g].mean(0) for g in big])
        nrm = np.linalg.norm(cent, axis=1, keepdims=True)
        S = (cent @ cent.T)/np.maximum(nrm @ nrm.T, 1e-12)
        _, merged = connected_components(csr_matrix(S >= max_profile_similarity), directed=False)
        remap = {g: big[np.flatnonzero(merged == merged[i])[np.argmax(sizes[big][merged == merged[i]])]]
                 for i, g in enumerate(big)}
        lab = np.array([remap.get(x, x) for x in lab])
        sizes = np.bincount(lab, minlength=ncomp)
        big = np.flatnonzero(sizes >= min_group)
        if big.size < 2:
            cols.append(Bb[:, j]); origin.append(j)
            continue
        info[j] = sorted(sizes[big].tolist(), reverse=True)
        final = lab.copy()
        for q in np.flatnonzero(~np.isin(lab, big)):
            votes = np.bincount(lab[nb[q][nb[q] >= 0]], minlength=ncomp)[big] if (nb[q] >= 0).any() else np.zeros(big.size)
            final[q] = big[int(votes.argmax())] if votes.sum() > 0 else big[int(sizes[big].argmax())]
        for g in big:
            col = np.zeros(n, bool); col[C[final == g]] = True
            cols.append(col); origin.append(j)
    return np.column_stack(cols).astype(np.int8), np.array(origin), info


def resolve_conflicts(B, score, min_gamete=10):
    """Greedy four-gamete resolution: drop the less locally-compact character of a conflict."""
    Bb = B > 0
    m = Bb.shape[1]
    inc = np.zeros((m, m), bool)
    for i, j in combinations(range(m), 2):
        if _incompatible(Bb[:, i], Bb[:, j], min_gamete):
            inc[i, j] = inc[j, i] = True
    alive = np.ones(m, bool)
    while True:
        sub = inc[np.ix_(alive, alive)]
        if not sub.any():
            break
        idx = np.flatnonzero(alive)
        conflicted = idx[sub.any(1)]
        alive[conflicted[int(np.argmin(score[conflicted]))]] = False
    return alive
