"""
Main MiTo class for MT-SNVs single-cell phylogenies annotation.
"""

from mito_utils.utils import *
from mito_utils.preprocessing import *
from mito_utils.phylo import *


##


class MiToTreeAnnotator():
    """
    MiTo tree annotation class. Performs clonal inference from an arbitrary
    MT-SNVs-based phylogeny.
    """

    def __init__(self, tree):
        """
        Initialize class slots from input CassiopeiaTree.
        """

        # Slots
        self.tree = tree.copy()
        self.T = None
        self.M = None
        self.mut_df = None
        self.solutions = None
        self.ordered_muts = None
        self.clone_df = None
        self.clonal_nodes = None
        self.S_aggregate = None
        self.params = {}

    ##

    def get_T(self, with_root=True):
        """
        Compute the "cell assignment" matrix, T.
        T is a cell x clade (internal node) binary matrix mapping each cell i to every clade j.
        """
        clades = get_clades(self.tree, with_root=with_root, with_singletons=True)
        T = np.array([[x in clades[clade] for x in self.tree.leaves] for clade in clades]).astype(int)
        T = (
            pd.DataFrame(T, index=list(clades.keys()), columns=self.tree.leaves)
            .T.loc[self.tree.layers['transformed'].index]
        )
        self.T = T

    ##

    def get_M(self, alpha=0.05):
        """
        Compute the "mutation enrichment" matrix, M.
        M is a mut x clade matrix storing for each mutation i and clade j the enrichment 
        value defined as -log10(pval) from a Fisher's Exact test.
        """
        P = {}
        muts = self.tree.layers['transformed'].columns

        # For each internal node (clade)
        for lineage_column in self.T.columns:
            target_ratio_array = np.zeros(muts.size)
            oddsratio_array = np.zeros(muts.size)
            pvals = np.zeros(muts.size)

            # For each mutation
            for i in range(muts.size):
                test_mut = self.tree.layers['transformed'].values[:, i] == 1
                test_lineage = self.T[lineage_column].values == 1
                n_mut_lineage = np.sum(test_mut & test_lineage)
                n_mut_no_lineage = np.sum(test_mut & ~test_lineage)
                n_no_mut_lineage = np.sum(~test_mut & test_lineage)
                n_no_mut_no_lineage = np.sum(~test_mut & ~test_lineage)
                target_ratio_array[i] = n_mut_lineage / (np.sum(test_mut) + 1e-8)
                # Fisher's exact test
                oddsratio, pvalue = fisher_exact(
                    [
                        [n_mut_lineage, n_mut_no_lineage],
                        [n_no_mut_lineage, n_no_mut_no_lineage],
                    ],
                    alternative='greater',
                )
                oddsratio_array[i] = oddsratio
                pvals[i] = pvalue

            # Adjust p-values (FDR correction)
            pvals = multipletests(pvals, alpha=alpha, method="fdr_bh")[1]
            P[lineage_column] = pvals

        M = pd.DataFrame(P, index=muts)
        M = -np.log10(M)

        self.M = M

    ##

    def resolve_ambiguous_clones(self, df_predict, s_treshold=.7, add_to_meta=False):
        """
        Final clonal resolution process. 
        Tries to merge similar clones iteratively.
        First, the (raw) AF matrix is aggregated at the clonal level using MiTo clones.
        Then, clone-clone similarities are comuputed using (1-) weighted jaccard distances
        among these aggregated MT-SNVs clonal profiles. At each round, the tiniest "ambiguous"
        clone is selected for merging with its smallest "interacting clone". If the merge is 
        successfull, the clonal assignment table is updated and the process go through other 
        merging rounds, until no ambiguous clones remain. Unresolved clones (if any) are 
        annotated as NaNs in the final MiTo clone column which is appended to self.tree.cell_meta.
        """

        self.params['merging_treshold'] = s_treshold
        remained_unresolved = []
        try_merge = True
        n_trials = 0

        # Here we go
        while try_merge:

            logging.info(f'n trials: {n_trials}')

            # Aggregate
            X_agg = (
                self.tree.layers['raw']
                .join(df_predict[['MiTo clone']])
                .groupby('MiTo clone')
                .median()
            )
            X_agg = X_agg.loc[:,np.any(X_agg>0, axis=0)]

            # Compute similarity of putative clones (i.e., aggregated profiles)
            w = np.nanmedian(np.where(X_agg>0, X_agg, np.nan), axis=0)
            S_agg = 1-weighted_jaccard((X_agg>0).astype(int), w=w)
            self.S_aggregate = pd.DataFrame(S_agg, index=X_agg.index, columns=X_agg.index)

            # Spot interacting clones 
            S_agg_long = (
                pd.DataFrame(
                    np.triu(S_agg, k=1), 
                    index=X_agg.index.to_list(), 
                    columns=X_agg.index.to_list()
                )
                .melt(ignore_index=False)
                .reset_index()
            )
            S_agg_long.columns = ['clone1', 'clone2', 'similarity']
            S_agg_long = (
                S_agg_long
                .query('clone1!=clone2 and similarity>=@s_treshold')
                .reset_index(drop=True).drop_duplicates()
            )
            if S_agg_long.shape[0] == 0:
                logging.info('No ambiguous clones remaining!')
                try_merge = False
            else:
                logging.info(f'n {S_agg_long.shape[0]} ambiguous clonal interactions found')

            # Attempt merging, starting from the tiniest ambiguous clone

            # Gather clonal stats
            clone_df = (
                df_predict
                .loc[:,['MiTo clone', 'lca', 'n cells', 'muts']]
                .reset_index(drop=True).drop_duplicates()
                .sort_values('n cells')
                .dropna()
                .rename(columns={'MiTo clone':'old'})
                .set_index('old')
            )
            self.clone_df = clone_df

            # Find tiniest ambiguous clone
            if try_merge:
                ambiguous_clones = set(S_agg_long['clone1']) | set(S_agg_long['clone2'])
                logging.info(f'n {len(ambiguous_clones)} ambiguous clones')
                for clone in clone_df.index:
                    if clone in ambiguous_clones and not clone in remained_unresolved:
                        try_merge = True
                        logging.info(f'Merge clone: {clone}')
                        break
                    else:
                        try_merge = False

            if try_merge:

                # Find the tiniest ambiguous clone interactor
                interactions = S_agg_long.query('clone1==@clone or clone2==@clone')
                other_clones = []
                for i in range(interactions.shape[0]):
                    int_clones = interactions.iloc[i,:].values[:2]
                    other_clones.append(int_clones[int_clones!=clone][0])
                int_clone = clone_df.loc[other_clones].sort_values('n cells').index[0]

                # Find lca new clone
                lca_clone = clone_df.loc[clone, 'lca']
                lca_int_clone = clone_df.loc[int_clone, 'lca']
                if lca_clone == lca_int_clone:
                    new_lca = lca_clone
                else:
                    new_lca = self.tree.find_lca(*[lca_clone, lca_int_clone])

                # Merge clones mutations 
                clone_muts = set(clone_df.loc[clone, 'muts'].split(';'))
                int_clone_muts = set(clone_df.loc[int_clone, 'muts'].split(';'))
                new_clone_muts = ';'.join(list(clone_muts | int_clone_muts))

                # Check merging possibility
                cells_merged_clone = (
                    df_predict
                    .loc[ lambda x: (x['MiTo clone'] == clone) | (x['MiTo clone'] == int_clone)]
                    .index
                )
                cells_clone = df_predict.loc[lambda x: x['MiTo clone'] == clone].index
                cells_int_clone = df_predict.loc[lambda x: x['MiTo clone'] == int_clone].index

                # test1 = len(set(new_clone_cells) - set(cells_merged_clone)) == 0
                test2 =  len(set(cells_merged_clone) - (set(cells_clone) | set(cells_int_clone)) ) == 0

                if test2: # and test1
                    df_predict.loc[cells_merged_clone, 'MiTo clone'] = int_clone
                    df_predict.loc[cells_merged_clone, 'median cell similarity'] = self.tree.get_attribute(new_lca, 'similarity')
                    df_predict.loc[cells_merged_clone, 'n cells'] = cells_merged_clone.size
                    df_predict.loc[cells_merged_clone, 'lca'] = new_lca
                    df_predict.loc[cells_merged_clone, 'muts'] = new_clone_muts
                    logging.info(f'Successfully merged clone {clone} and {int_clone}')
                else:
                    logging.info(f'{clone} and {int_clone} cannot be merged')
                    remained_unresolved.append(clone)

                n_trials += 1

            else:

                logging.info(f'Finished merging!')

        # Put final, unresolved_clones as NaNs
        df_predict.loc[df_predict['MiTo clone'].isin(remained_unresolved), 'MiTo clone'] = np.nan
        sizes = df_predict['MiTo clone'].value_counts()
        mapping = { x : f'MT-{i}' for i,x in enumerate(sizes.index) }
        df_predict['MiTo clone'] = df_predict['MiTo clone'].map(mapping)
        self.clonal_nodes = df_predict.loc[lambda x: ~x['MiTo clone'].isna(), 'lca'].unique().tolist()

        # Check if the df_predict info should be added to self.tree.cell_meta
        if add_to_meta:
            self.tree.cell_meta = self.tree.cell_meta.join(df_predict)

        return df_predict['MiTo clone'], df_predict['median cell similarity']

    ##  

    def extract_mut_order(self, pval_tresh=.01):
        """
        Extract diagonal-order of MT-SNVs using mutation assignments, to create a ordered list of
        MT-SNVs for plotting.
        """

        assert (self.M.index == self.tree.layers['transformed'].columns).all()

        # Define a MT-SNVs treshold
        mut_df = (10**(-self.M.max(axis=1))<=pval_tresh).to_frame('assignment')
        mut_df['prevalence'] = self.tree.layers['transformed'].sum(axis=0) / len(self.tree.leaves)
        mut_df['top_node'] = self.M.columns[self.M.values.argmax(axis=1)]

        top_nodes = mut_df.loc[mut_df['assignment'], 'top_node'].to_list()

        mutation_order = []
        for node in self.tree.depth_first_traverse_nodes():
            if node in top_nodes:
                mut = mut_df.loc[mut_df['top_node']==node].index[0]
                mutation_order.append(mut)

        unassigned_muts = mut_df.loc[~mut_df['assignment']].sort_values('prevalence').index.to_list()
        mutation_order += unassigned_muts

        self.ordered_muts = mutation_order
        self.mut_df = mut_df

    ##

    def infer_clones(self, similarity_percentile=85, mut_enrichment_treshold=5, merging_treshold=.7, add_to_meta=False):
        """
        A MT-SNVs-specific re-adaptation of the recursive approach described in the MethylTree paper 
        (... et al., 2025).
        """

        logging.info('Prep data for clonal inference.')

        # Prep lists for recursion
        tree_list = []
        df_list = []

        # Get tree topology and mutation_enrichment tables
        if self.T is None or self.M is None: 
            self.get_T()
            self.get_M()

        # Set usable_mutations
        is_enriched = (self.M.values>=mut_enrichment_treshold).any(axis=1)
        mut_enrichment = self.M.loc[is_enriched,:]
        usable_mutations = set(mut_enrichment.index)
        self.params['mut_enrichment_treshold'] = mut_enrichment_treshold
        self.params['n total mutations'] = self.tree.layers['raw'].shape[1]
        self.params['n usable_mutations'] = len(usable_mutations)

        # Convert the dissimilarity into similarity matrix
        D = self.tree.get_dissimilarity_map()
        S = 1-D
        # Calculate similarity treshold
        similarity_treshold = np.percentile(S.values, similarity_percentile)
        self.params['similarity_treshold'] = similarity_treshold

        # Set median similarity among clade cells as tree node attributes
        clades = get_clades(self.tree, with_singletons=True)
        for node in clades:
            cells = list(clades[node])
            s = np.median(S.loc[cells,cells].values)
            self.tree.set_attribute(node, 'similarity', s)

        ##

        # Internal, recursive functions =========================================================== #

        def _find_clones(tree, node, usable_mutations):

            if not usable_mutations:
                _collect_clones(tree, node)
                return

            if tree.is_leaf(node):
                _collect_clones(tree, node)
            elif (
                ((mut_enrichment.loc[list(usable_mutations), node] >= mut_enrichment_treshold).any()) & \
                (tree.get_attribute(node, 'similarity') >= similarity_treshold) & \
                (not tree.is_leaf(node))
            ):
                triggered = {
                    mut for mut in usable_mutations \
                    if mut_enrichment.loc[mut,node] >= mut_enrichment_treshold 
                }
                tree.set_attribute(node, 'muts', list(triggered))
                new_usable = usable_mutations - triggered
                _refine_clones(tree, node, new_usable)
            else:
                for child in tree.children(node):
                    _find_clones(tree, child, usable_mutations)

        ##

        def _collect_clones(tree, node_tmp):
            tree_list.append(node_tmp)
            leaves = tree.leaves_in_subtree(node_tmp)
            df_tmp = pd.DataFrame({"cell": leaves})
            df_tmp["MiTo clone"] = f"MT-{len(df_list)}"
            df_tmp["median cell similarity"] = tree.get_attribute(node_tmp, 'similarity')
            df_tmp["n cells"] = len(leaves)
            df_tmp['lca'] = tree.find_lca(*leaves) if len(leaves)>1 else np.nan
            try:
                df_tmp['muts'] = ';'.join(tree.get_attribute(node_tmp, 'muts'))
            except:
                pass
            df_list.append(df_tmp)

        ##

        def _collect_clade_info(tree, node_tmp, usable_mutations, level=0, data_list=[], name="0"):
            valid_tmp = (
                ((mut_enrichment.loc[list(usable_mutations), node_tmp] >= mut_enrichment_treshold).any()) & \
                (tree.get_attribute(node_tmp, 'similarity') >= similarity_treshold) & \
                (not tree.is_leaf(node_tmp))                       
            )
            data_list.append([level, int(valid_tmp), len(tree.leaves_in_subtree(node_tmp)), name])
            for j0, child_tmp in enumerate(tree.children(node_tmp)):
                _collect_clade_info(
                    tree, child_tmp, usable_mutations, level=level+1, data_list=data_list, name=f"{name},{j0}"
                )

        ##

        def _add_clade_info(tree, node_tmp, target_level, target_name_list, level=0, name="0"):

            valid_list = []
            invalid_list = []

            for target_name in target_name_list:
                condition_1 = target_name == name          # this sub-tree matches our target names
                condition_2 = target_name.startswith(name) # this sub-tree is along the branch that leads to the target sub-tree, therefore acceptable
                condition_3 = name.startswith(target_name) # this sub-tree is contained within one of the target sub-trees
                valid_list.append(condition_1)
                invalid_list.append(condition_2 or condition_3)

            valid = np.sum(valid_list) > 0

            if not valid:
                valid = np.sum(invalid_list) == 0

            if valid:
                _collect_clones(tree, node_tmp)
                if name not in target_name_list:
                    target_name_list.append(name)

            if level < target_level:
                for j0, child_tmp in enumerate(tree.children(node_tmp)):
                    _add_clade_info(
                        tree,
                        child_tmp,
                        target_level,
                        target_name_list,
                        level=level + 1,
                        name=f"{name},{j0}",
                    )
            else:
                return

        ##

        def _refine_clones(tree, node, usable_mutations):

            if not usable_mutations:
                _collect_clones(tree, node)
                return

            data_list = []
            _collect_clade_info(tree, node, usable_mutations, level=0, data_list=data_list)
            df_tmp_orig = pd.DataFrame(
                np.array(data_list), columns=["level", "score", "cell_N", "name"]
            )
            df_tmp_orig["level"] = df_tmp_orig["level"].astype(int)
            df_tmp_orig["score"] = df_tmp_orig["score"].astype(float)
            df_tmp_orig["cell_N"] = df_tmp_orig["cell_N"].astype(int)

            df_tmp = (
                df_tmp_orig.groupby("level")
                .agg({"score": "mean", "cell_N": "sum"})
                .reset_index()
            )
            df_tmp = df_tmp[(df_tmp["score"] == 1)]
            target_level = df_tmp["level"].max()
            target_name_list = df_tmp_orig[df_tmp_orig["level"] == target_level]["name"].to_list()

            _add_clade_info(tree, node, target_level, target_name_list)

        # ============================================================================================ #

        # Fire recursion!
        logging.info('Recursive tree split...')
        tree_ = self.tree.copy()
        _find_clones(tree_, tree_.root, usable_mutations)

        # Get results
        df_predict = pd.concat(df_list, ignore_index=True)
        df_predict = df_predict.set_index('cell')
        df_predict.loc[df_predict['muts'].isna(), 'MiTo clone'] = np.nan

        # Resolve over-clustering
        logging.info('Solve potential over-clustering...')
        labels, similarities = self.resolve_ambiguous_clones(df_predict, s_treshold=merging_treshold, add_to_meta=add_to_meta)

        return labels, similarities
    
    ##

    def clonal_inference(
        self, 
        similarity_tresholds=[ 85, 90, 95, 99 ],
        mut_enrichment_tresholds=[ 3, 5, 10 ],
        merging_treshold=[ .25, .5, .75 ],
        max_fraction_unassigned=.05,
        weight_silhouette=.3,
        weight_n_clones=.4,
        weight_similarity=.3
        ):
        """
        Optimize tresholds for self.infer_clones and pick clonal labels with
        best silhouette scores across the attempted splits.
        """

        T = Timer()
        T.start()

        # Grid-search
        from itertools import product
        combos = list(product(similarity_tresholds, mut_enrichment_tresholds, merging_treshold))
        logging.info(f'Start Grid Search. n hyper-parameter combinations to explore: {len(combos)}')
        print('\n')

        silhouettes = [] 
        unassigned = []
        n_clones = []
        similarities = []

        i = 0
        for s,m,j in combos:
            logging.info(f'Grid Search: {i}/{len(combos)}')
            logging.info(f'Perform clonal inference with params: similarity_percentile={s}, mut_enrichment_treshold={m}, merging_treshold={j}')
            labels, sim = self.infer_clones(
                similarity_percentile=s, mut_enrichment_treshold=m, merging_treshold=j, add_to_meta=False
            )
            test = labels.isna()
            labels = labels.loc[~test]
            D = self.tree.get_dissimilarity_map().loc[labels.index,labels.index]
            assert (D.index == labels.index).all()
            sil = silhouette_score(X=D.values, labels=labels.values, metric='precomputed') if labels.unique().size>2 else 0
            silhouettes.append(sil)
            unassigned.append(test.sum()/labels.size)
            n_clones.append(labels.unique().size)
            similarities.append(sim.mean())
            print('\n')
            i += 1

        # Pick optimal combination, and perform final splitting
        self.solutions = (
            pd.DataFrame({'silhouette':silhouettes, 'unassigned':unassigned, 
                          'n_clones':n_clones, 'similarity':similarities})
            .assign(
                sil_rescaled = lambda x: rescale(x['silhouette']),
                sim_rescaled = lambda x: rescale(x['similarity']),
                n_clones_rescaled = lambda x: rescale(-x['n_clones']),
                score = lambda x: 
                    weight_silhouette * x['sil_rescaled'] + \
                    weight_n_clones * x['n_clones_rescaled'] + \
                    weight_similarity * x['similarity']
            )
            .sort_values('score', ascending=False)
        )
        chosen = (
            self.solutions
            .query('unassigned<=@max_fraction_unassigned')
            .sort_values('score', ascending=False)
            .index[0]
        )
        s, m, j = combos[chosen]
        
        # Final round
        logging.info(f'Hyper-params chosen: similarity_percentile={s}, mut_enrichment_treshold={m}, mut_enrichment_treshold={j}')
        print('\n')
        _,_ = self.infer_clones(
            similarity_percentile=s, mut_enrichment_treshold=m, merging_treshold=j, add_to_meta=True
        )

        # Retrieve mutation order for plotting
        self.extract_mut_order()

        print('\n')
        logging.info(f'MiTo clonal inference finished. {T.stop()}')


##