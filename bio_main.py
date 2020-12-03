import sys
import argparse

sys.path.append('bioactivity_dataset/bin')
from main_setup import model_training_scoring

parser = argparse.ArgumentParser(description='Benchmark Platform arguments')

parser.add_argument(
	'--dataset_name',
	type = str,
	default = 'nr_data',
	metavar = 'dN',
	help = 'folder name model stored')

parser.add_argument(
	'--split_type',
	type = str,
	default = 'random_split',
	metavar = 'ST',
	help = 'splitter to split dataset (default: random_split)')

parser.add_argument(
	'--scaler_type',
	type = str,
	default = 'MinMax_Scaler',
	metavar = 'ST',
	help = 'scaling type to scale dataset (default: MinMax_Scaler)')

parser.add_argument(
	'--learning_type',
	type = str,
	default = 'Regression',
	metavar = 'LT',
	help = 'learning type for dataset (default: Regression)')

parser.add_argument(
	'--machine_type',
	type = str,
	default = 'random_forest',
	metavar = 'MT',
	help = 'machine type to train dataset(default: random_forest)')

parser.add_argument(
	'--ratio',
	type = float,
	default = 0.2,
	metavar = 'Rt',
	help = 'ratio to split data(default: 0.2')
parser.add_argument(
	'--cv',
	type = NoneType,
	default = None,
	metavar = 'cv',
	help = 'cross validation (default: None)')


if __name__ == '__main__':
	args = parser.parse_args()
	print(args)
    model_training_scoring(args.dataset_name,
    	args.learning_type,
    	args.scaler_type,
    	args.split_types,
    	args.machine_type,
    	args.ratio,
    	args.cv)





