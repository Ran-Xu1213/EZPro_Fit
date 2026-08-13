# codon/utils/codon_preference.py
import torch
import numpy as np
from codon.utils.codon_const import codon_order, codon_to_res

# ??(Saccharomyces cerevisiae)?????(????)
YEAST_CODON_USAGE = {
    'UUU': 26.1, 'UUC': 18.4, 'UUA': 26.2, 'UUG': 27.2,
    'UCU': 23.5, 'UCC': 14.2, 'UCA': 18.7, 'UCG': 8.6,
    'UAU': 18.8, 'UAC': 14.8, 'UAA': 1.0, 'UAG': 0.5,
    'UGU': 8.1, 'UGC': 4.8, 'UGA': 0.7, 'UGG': 10.4,
    'CUU': 12.3, 'CUC': 5.4, 'CUA': 13.4, 'CUG': 10.5,
    'CCU': 13.5, 'CCC': 6.8, 'CCA': 18.3, 'CCG': 5.3,
    'CAU': 13.6, 'CAC': 7.8, 'CAA': 27.3, 'CAG': 12.1,
    'CGU': 6.4, 'CGC': 2.6, 'CGA': 3.0, 'CGG': 1.7,
    'AUU': 30.1, 'AUC': 17.2, 'AUA': 17.2, 'AUG': 20.9,
    'ACU': 20.3, 'ACC': 12.7, 'ACA': 17.8, 'ACG': 8.0,
    'AAU': 35.7, 'AAC': 24.8, 'AAA': 42.0, 'AAG': 30.8,
    'AGU': 14.2, 'AGC': 9.8, 'AGA': 21.3, 'AGG': 9.2,
    'GUU': 22.2, 'GUC': 11.8, 'GUA': 11.9, 'GUG': 10.8,
    'GCU': 21.2, 'GCC': 12.6, 'GCA': 16.2, 'GCG': 6.2,
    'GAU': 37.6, 'GAC': 20.2, 'GAA': 45.6, 'GAG': 19.2,
    'GGU': 23.9, 'GGC': 9.8, 'GGA': 10.9, 'GGG': 6.0
}

# ???(E. coli)?????(????)
ECOLI_CODON_USAGE = {
    'UUU': 22.0, 'UUC': 16.5, 'UUA': 13.2, 'UUG': 13.0,
    'UCU': 8.2, 'UCC': 8.8, 'UCA': 7.2, 'UCG': 8.5,
    'UAU': 16.5, 'UAC': 12.2, 'UAA': 2.0, 'UAG': 0.2,
    'UGU': 4.6, 'UGC': 6.1, 'UGA': 1.0, 'UGG': 15.3,
    'CUU': 11.1, 'CUC': 10.8, 'CUA': 3.9, 'CUG': 52.6,
    'CCU': 7.0, 'CCC': 5.5, 'CCA': 8.3, 'CCG': 22.9,
    'CAU': 12.9, 'CAC': 9.5, 'CAA': 15.3, 'CAG': 29.5,
    'CGU': 20.8, 'CGC': 21.9, 'CGA': 3.6, 'CGG': 5.1,
    'AUU': 30.4, 'AUC': 24.9, 'AUA': 4.2, 'AUG': 27.1,
    'ACU': 8.9, 'ACC': 23.2, 'ACA': 7.7, 'ACG': 14.1,
    'AAU': 17.0, 'AAC': 21.6, 'AAA': 33.3, 'AAG': 10.5,
    'AGU': 6.3, 'AGC': 15.8, 'AGA': 2.1, 'AGG': 1.2,
    'GUU': 18.3, 'GUC': 14.8, 'GUA': 11.0, 'GUG': 26.3,
    'GCU': 15.8, 'GCC': 25.4, 'GCA': 20.5, 'GCG': 32.9,
    'GAU': 32.6, 'GAC': 19.1, 'GAA': 39.5, 'GAG': 17.8,
    'GGU': 24.5, 'GGC': 27.7, 'GGA': 8.2, 'GGG': 10.8
}

def create_codon_preference_matrix(organism='yeast'):
    """?????""
    usage_dict = YEAST_CODON_USAGE if organism == 'yeast' else ECOLI_CODON_USAGE
    
    # NA??ＶRNA??
    dna_to_rna = {'A': 'A', 'T': 'U', 'C': 'C', 'G': 'G'}
    
    preference_matrix = torch.zeros(len(codon_order))
    
    for dna_codon, idx in codon_order.items():
        if dna_codon == 'UNK':
            preference_matrix[idx] = 0.1  # ??
            continue
            
        # NA??
        rna_codon = ''.join([dna_to_rna.get(base, base) for base in dna_codon])
        
        if rna_codon in usage_dict:
            preference_matrix[idx] = usage_dict[rna_codon] / 1000.0  # ??0-1
        else:
            preference_matrix[idx] = 0.01  # ????
    
    return preference_matrix

def get_organism_id_mapping():
    """???D??"""
    return {
        'yeast': 0,
        'ecoli': 1,
        'other': 2
    }

