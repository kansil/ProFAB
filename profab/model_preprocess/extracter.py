# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 18:25:15 2022

@author: Sameitos
"""

from profab.utils.deep_extracter import t5_features, bert_features
from profab.utils.feature_extraction_module.feature_extracter import feature_extracter
from profab.utils.feature_extraction_module.utils import bcolors
import category_encoders as ce


def extract_protein_feature(protein_feature,
                          place_protein_id,
                          input_folder,
                          fasta_file_name
                          ):
    '''
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



def extract_deep_feature(seq_data, pretrained_model= 't5_xl',take_avg = False,max_len = -1):
    '''
    Description:
        This function is to transform protein sequences into continuous data
        by using RostLab pretrained models with "transformers" Python package
    Parameters:
        seq_data: {np.array, list}, protein sequence data. 
        pretrained_model: {str}, {'prot_t5_xl_uniref50','prot_bert'}, 
            default= "prot_t5_xl_uniref50", pretrained model that RostLab
            provides to "transformers" package
        take_avg: {bool}, default = False, if True, average of vectors will be returned
        max_len: {int}, default = -1, Max sequence lenght to embed
    Return:
        embedded_data: {np.array}, transformed continous data.
    
    '''


    if pretrained_model == 't5_xl':
        return t5_features(seq_data,take_avg,max_len) 
    elif pretrained_model == 'bert':
        return bert_features(seq_data,take_avg,max_len)

def onehotencoder():
    pass









