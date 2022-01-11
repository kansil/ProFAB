# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 18:25:15 2022

@author: Sameitos
"""

from ..feature_extraction_module import feature_extracter
from ..feature_extraction_module.utils import bcolors

def extract_protein_feature(protein_feature,
                          place_protein_id,
                          input_folder,
                          fasta_file_name):


    feat_ext = feature_extracter(protein_feature,
                                 place_protein_id,
                                 input_folder,
                                 fasta_file_name)

    if protein_feature in feat_ext.POSSUM_desc_list:

        feat_ext.extract_POSSUM_feature()

    elif protein_feature in feat_ext.iFeature_desc_list:

        feat_ext.extract_iFeature_feature()

    else:

        print(f"{bcolors.FAIL}Protein Feature extraction method is not in either POSSUM or iFeature{bcolors.ENDC}")
