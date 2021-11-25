# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 22:32:37 2021

@author: Sameitos
"""

from . import data_importer

class DTI(data_importer.rgs_data_loader):
    '''
    DTI is a function to import drug-target interaction data. It gives X data 
    and y data separately
    Parameters:
        ratio: {None, float, list}, (default = None): used to split data 
                into train, test, validation set as given values. 
        protein_faeture: {'paac','aac','gaac','ctriad','ctdt','soc_number'},
                (default = 'paac'): numerical features of protein sequences
        set_type: {'random','target','temporal'}, (default = 'random'):
                split type of data, random:random splitting, target:
                similarity based splitting, temporal: splitting according to
                annotation time
    '''
    def __init__(self,protein_feature = 'paac',
                 set_type = 'random',
                 ratio = None):
        
        self.protein_feature = protein_feature
        self.set_type = set_type
        self.ratio = ratio
        super().__init__(ratio = self.ratio, protein_feature = self.protein_feature,
                         set_type = self.set_type)
    
class ECNO(data_importer.cls_data_loader):
    '''
    ECNO is a function to import enzyme commssion number data. It gives X data 
    and y data separately 
    Parameters:
        ratio: {None, float, list}, (default = None): used to split data 
                into train, test, validation set as given values. 
        protein_faeture: {'paac','aac','gaac','ctriad','ctdt','soc_number'},
                (default = 'paac'): numerical features of protein sequences
        set_type: {'random','target','temporal'}, (default = 'random'):
                split type of data, random:random splitting, target:
                similarity based splitting, temporal: splitting according to
                annotation time
        pre_determined: bool, (default = False), if False, data is given
                according to ratio type, If True, already splitted data will
                provided.
        label: {None, 'positive','negative'}, (default = None): If None, data
                is given directly, if 'negative', only negative set is given,
                If 'positive', only positive set is given.
    '''
    def __init__(self,protein_feature = 'paac',
                 set_type = 'random',
                 ratio = None,
                 label = None,
                 pre_determined = True):

        super().__init__(ratio = ratio, protein_feature = protein_feature,
                         set_type = set_type,label = label,
                         pre_determined = pre_determined, main_set = 'ec_dataset') 

class GOID(data_importer.cls_data_loader):
    '''
    GOID is a function to import gene ontology term data. It gives X data and
    y data separately.
    Parameters:
        ratio: {None, float, list}, (default = None): used to split data 
                into train, test, validation set as given values. 
        protein_faeture: {'paac','aac','gaac','ctriad','ctdt','soc_number'},
                (default = 'paac'): numerical features of protein sequences
        set_type: {'random','target','temporal'}, (default = 'random'):
                split type of data, random:random splitting, target:
                similarity based splitting, temporal: splitting according to
                annotation time
        pre_determined: bool, (default = False), if False, data is given
                according to ratio type, If True, already splitted data will
                provided.
        label: {None, 'positive','negative'}, (default = None): If None, data
                is given directly, if 'negative', only negative set is given,
                If 'positive', only positive set is given.
    '''
    def __init__(self,protein_feature = 'paac',
                 set_type = 'random',
                 ratio = None,
                 label = None,
                 pre_determined = False):
        
        super().__init__(ratio = ratio, protein_feature = protein_feature,
                         set_type = set_type,label = label,
                         pre_determined = pre_determined, main_set = 'go_dataset')       
