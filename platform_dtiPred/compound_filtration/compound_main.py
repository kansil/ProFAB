<<<<<<< HEAD

import argparse
from compound_setup import form_features

parser = argparse.ArgumentParser(description='Compound Feature Forming arguments')

parser.add_argument(
	'--dataset_name',
	type = str,
	default = 'nr_data',
	metavar = 'dN',
	help = 'folder name model stored')

parser.add_argument(
	'--cluster',
	type = bool,
	default = False,
	metavar = 'dN',
	help = 'will clustering be done')

if __name__ == '__main__':
	args = parser.parse_args()
	print(args)
	form_features(args.dataset_name,args.cluster)

