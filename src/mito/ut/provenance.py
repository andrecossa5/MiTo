"""
Provenance: what was run on an AFM, and with which parameters.
"""

from anndata import AnnData

from mito import __version__

##


def record(afm: AnnData, step: str, params: dict):
    """
    Add one step's parameters to the object's provenance, under .uns["mito"][step].

    Every public function that changes an AFM leaves its record here, so a stage run on
    its own and the same stage run through `mito.pp.filter_afm` are documented in the
    same place, and the whole history of an object is one dictionary.
    """

    if 'mito' not in afm.uns:
        afm.uns['mito'] = {'version':__version__}
    afm.uns['mito'][step] = params


##
