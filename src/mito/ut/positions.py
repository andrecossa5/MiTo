"""
MT-genome feature annotation.
"""

import numpy as np
import pandas as pd


##


patterns = [ 'A>C', 'T>G', 'A>T', 'A>G', 'G>A', 'C>G', 'C>A', 'T>A', 'G>C', 'G>T', 'N>T', 'C>T', 'T>C' ]

transitions = [pattern for pattern in patterns if pattern in ['A>G', 'G>A', 'C>T', 'T>C']]

transversions = [pattern for pattern in patterns if pattern not in transitions and 'N' not in pattern]

all_mt_genes_positions = [
    ["MT-ND1", 3307, 4262], ["MT-ND2", 4470, 5511], ["MT-CO1", 5904, 7445],
    ["MT-CO2", 7586, 8269], ["MT-ATP8", 8366, 8572], ["MT-ATP6", 8527, 9207],
    ["MT-CO3", 9207, 9990], ["MT-ND3", 10059, 10404], ["MT-ND4L", 10470, 10766],
    ["MT-ND4", 10760, 12137], ["MT-ND5", 12337, 14148], ["MT-ND6", 14149, 14673],
    ["MT-CYB", 14747, 15887], ["MT-TF", 577, 647], ["MT-TV", 1602, 1670],
    ["MT-TL1", 3230, 3304], ["MT-TI", 4263, 4331], ["MT-TQ", 4329, 4400],
    ["MT-TM", 4402, 4469], ["MT-TW", 5512, 5579], ["MT-TA", 5587, 5655],
    ["MT-TN", 5657, 5729], ["MT-TC", 5761, 5826], ["MT-TY", 5826, 5891],
    ["MT-TS1", 7518, 7585], ["MT-TD", 7513, 7585], ["MT-TK", 8295, 8364],
    ["MT-TG", 9991, 10058], ["MT-TR", 10405, 10469], ["MT-TH", 12138, 12206],
    ["MT-TS2", 12207, 12265], ["MT-TL2", 12266, 12336], ["MT-TE", 14674, 14742],
    ["MT-TT", 15888, 15953], ["MT-TP", 15956, 16023], ["12S rRNA", 648, 1601],
    ["16S rRNA", 1671, 3229]
]

MAESTER_genes_positions = [
    ["12S rRNA", 648, 1601],
    ["16S rRNA", 1671, 3229],
    ["MT-ND1", 3307, 4262],
    ["MT-ND2", 4470, 5511],
    ["MT-CO1", 5904, 7445],
    ["MT-CO2", 7586, 8269],
    ["MT-ATP8", 8366, 8572],
    ["MT-ATP6", 8527, 9207],
    ["MT-CO3", 9207, 9990],
    ["MT-ND3", 10059, 10404],
    ["MT-ND4L", 10470, 10766],
    ["MT-ND4", 10760, 12137],
    ["MT-ND5", 12337, 14148],
    ["MT-ND6", 14149, 14673],
    ["MT-CYB", 14747, 15887]
]

df_ = pd.DataFrame(MAESTER_genes_positions)
n_target_sites_maester = np.sum(df_[2]-df_[1])


##


def mask_mt_sites(site_list):
    """
    Function to mask all sites outside of known MT-genes bodies.
    """

    mask = []
    for pos in site_list:
        pos = int(pos)
        t = [ pos>=start and pos<=end for _, start, end in MAESTER_genes_positions ]
        if any(t):
            mask.append(True)
        else:
            mask.append(False)

    return np.array(mask)


##