# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 14:09:37 2021

@author: Sameitos
"""

import argparse

from compound_featuring import mol_to_bit

parser = argparse.ArgumentParser(description='Compound featuring via rdkit package')


parser.add_argument('--data_name',
                    required = True,
                    type = str,
                    metavar = 'dn',
                    help = ' Name of file of SMILES data')

parser.add_argument('--save_data',
                    type = str,
                    default = 'bitvector.tsv',
                    metavar = 'sd',
                    help = 'Name of file where fingerprits are stored. Its format can be .csv, .txt and .tsv')

parser.add_argument('--save_idx',
                    type = str,
                    default = 'indices.txt',
                    metavar = 'si',
                    help = 'Name of file where indices of data points are stored. It is beacuse some data points can be lost during process.'

parser.add_argument('--bits',
                    type = int,
                    default = 1024,
                    metavar = 'bits',
                    help = 'Number of dimensions of fingerprints')


if __name__ == '__main__':
    args = parser.parse_args()
    mol_to_bit(args.data_name,
               args.save_data,
               args.save_idx,
               args.bits)
