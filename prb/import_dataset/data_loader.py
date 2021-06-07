# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 22:32:37 2021

@author: Sameitos
"""

from . import data_importer

class DTI(data_importer.rgs_data_loader):
    
    def __init__(self,protein_feature = 'paac',
                 set_type = 'random',
                 ratio = None):
        
        self.protein_feature = protein_feature
        self.set_type = set_type
        self.ratio = ratio
        super().__init__(ratio = self.ratio, protein_feature = self.protein_feature,
                         set_type = self.set_type)
    
class ECNO(data_importer.cls_data_loader):
    
    def __init__(self,protein_feature = 'paac',
                 set_type = 'random',
                 ratio = None,
                 label = None,
                 pre_determined = False):
        super().__init__(ratio = ratio, protein_feature = protein_feature,
                         set_type = set_type,label = label,
                         pre_determined = pre_determined, main_set = 'ec_dataset') 

class GOID(data_importer.cls_data_loader):
    
    def __init__(self,protein_feature = 'paac',
                 set_type = 'random',
                 ratio = None,
                 label = None,
                 pre_determined = False):
        
        super().__init__(ratio = ratio, protein_feature = protein_feature,
                         set_type = set_type,label = label,
                         pre_determined = pre_determined, main_set = 'go_dataset')       
