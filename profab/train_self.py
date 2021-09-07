# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 23:07:30 2021

@author: Sameitos
"""

  
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 13:22:08 2021
@author: Sameitos
"""
import sys
import argparse


from .main_setup import model_all

parser = argparse.ArgumentParser(description='Benchmark Platform arguments')

parser.add_argument('--dataset_name',
                    required = True,
                    type = str,
                    default = 'nr_data',
                    metavar = 'dN',
                    help = 'folder name model stored')

parser.add_argument('--scaler_type',
                    type = str,
                    default = 'MinMax_Scaler',
                    metavar = 'ST',
                    help = 'scaling type to scale dataset (default: MinMax_Scaler)')

parser.add_argument('--learning_type',
    type = str,
    default = 'Classification',
    metavar = 'LT',
    help = 'learning type for dataset (default: Classification)')

parser.add_argument('--machine_type',
    type = str,
    default = 'random_forest',
    metavar = 'MT',
    help = 'machine type to train dataset(default: random_forest)')

parser.add_argument('--ratio',
    type = float,
    default = 0.2,
    metavar = 'Rt',
    help = 'ratio to split data(default: 0.2')

parser.add_argument('--cv',
    type = None,
    default = None,
    metavar = 'cv',
    help = 'cross validation (default: None)')


if __name__ == '__main__':
    args = parser.parse_args()
    model_all(args.dataset_name,
              args.learning_type,
              args.scaler_type,
              args.machine_type,
              args.ratio,args.cv)

