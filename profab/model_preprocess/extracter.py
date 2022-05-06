# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 18:25:15 2022

@author: Sameitos
"""

from ..utils import feature_extracter
from ..utils import bcolors

def extract_protein_feature(protein_feature,
                          place_protein_id,
                          input_folder,
                          fasta_file_name
                          ):
    '''
     The feature_extracter class is designed to extract features by employing POSSUM and iFeature python-based tools.
     POSSUM (Position-Specific Scoring matrix-based feature generator for machine learning),
     a versatile toolkit with an online web server that can generate 21 types of PSSM-based feature descriptors,
     thereby addressing a crucial need for bioinformaticians and computational biologists.
     iFeature, a versatile Python-based toolkit for generating various numerical feature representation schemes for
     both protein and peptide sequences. iFeature is capable of calculating and extracting a comprehensive spectrum
     of 18 major sequence encoding schemes that encompass 53 different types of feature descriptors.
     Parameters:
         protein_feature: {string}, (default = 'aac_pssm'): one of the 21 PSMM-based protein descriptors in POSSUM.
                          aac_pssm, d_fpssm, smoothed_pssm, ab_pssm, pssm_composition, rpm_pssm,
                          s_fpssm, dpc_pssm, k_separated_bigrams_pssm, eedp, tpc, edp, rpssm,
                          pse_pssm, dp_pssm, pssm_ac, pssm_cc, aadp_pssm, aatp, medp , or all_POSSUM
                          all_POSSUM: it extracts the features of all (21) POSSUM protein descriptors
                          or
                          one of the 18 protein descriptors in iFeature.
                          AAC, PAAC, APAAC, DPC, GAAC, CKSAAP, CKSAAGP, GDPC, Moran, Geary,
                          NMBroto, CTDC, CTDD, CTDT, CTriad, KSCTriad, SOCNumber, QSOrder, or all_iFeature
                          all_iFeature: it extracts the features of all (18) iFeature protein descriptors
         place_protein_id: {int}, (default = 1): It indicates the place of protein id in fasta header.
                           e.g. fasta header: >sp|O27002|....|....|...., seperate the header wrt. '|' then >sp is
                           in the zeroth position, protein id in the first(1) position.
        input_folder: {string}, (default = 'input_folder'}: it is the path to the folder that contains the fasta file.
        fasta_file_name: {string}, (default ='sample'): it is the name of the fasta file exclude the '.fasta' extension.
        
     Returns:
          output_file: {string}, path to output file where extracted features are written.
    '''

    feat_ext = feature_extracter(protein_feature,
                                 place_protein_id,
                                 input_folder,
                                 fasta_file_name)

    if protein_feature in feat_ext.POSSUM_desc_list:

        return feat_ext.extract_POSSUM_feature()

    elif protein_feature in feat_ext.iFeature_desc_list:

        return feat_ext.extract_iFeature_feature()

    else:

        print(f"{bcolors.FAIL}Protein Feature extraction method is not in either POSSUM or iFeature{bcolors.ENDC}")
