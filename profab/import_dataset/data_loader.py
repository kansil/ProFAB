# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 22:32:37 2021

@author: Sameitos
"""

from . import data_importer

class DTI(data_importer.rgs_data_loader):
    
    """!!!!THIS FUNCTION IS OUT OF USE FOR NOW. AS DTI DATASETS ARE IN PIPE,
       !!!!DTI IMPORTER WILL BE READY TO USE!!!!"""

    '''
    DTI is a function to import drug-target interaction data. It gives X data 
    and y data separately
    Parameters:
        ratio: {None, float, list}, (default = 0.2): used to split data 
                into train, test, validation sets as given values. If left None, 
                only X and y data can be obtained while float value gives train 
                and test set. If ratio = a (float), then test will be a% of total 
                data size. If ratio = [a,b] where a and b are in (0,1), train, test 
                and validation sets are formed according to them. For example, 
                If a = 0.2 and b = 0.1, train fraction is 0.7, test fraction is 0.2 
                and validation fraction is 0.1 of all dataset size.
        protein_faeture: {'paac','aac','gaac','ctriad','ctdt','soc_number', kpssm},
                (default = 'paac'): numerical features of protein sequences
        set_type: {'random','target'}, (default = 'random'):
                split type of data, random:random splitting, target:
                similarity based splitting
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
        ratio: {None, float, list}, (default = 0.2): used to split data 
                into train, test, validation sets as given values. If left None, 
                only X and y data can be obtained while float value gives train 
                and test set. If ratio = a (float), then test will be a% of total 
                data size. If ratio = [a,b] where a and b are in (0,1), 
                train, test and validation sets are formed according to them. For example, 
                If a = 0.2 and b = 0.1, train fraction is 0.7, test fraction is 0.2 
                and validation fraction is 0.1 of all dataset size. If set_type = 'temporal', 
                then ratio = None automatically.
        protein_faeture: {'paac','aac','gaac','ctriad','ctdt','soc_number','kpssm'},
                (default = 'paac'): numerical features of protein sequences
        set_type: {'random','similarity','temporal'}, (default = 'random'):
                split type of data, random:random splitting, target:
                similarity based splitting, temporal: splitting according to
                annotation time
        pre_determined: bool, (default = False), if False, data is given
                according to ratio type, If True, already splitted data will
                be provided.
        label: {None, 'positive','negative'}, (default = None): If None, data
                is given directly, if 'negative', only negative set is given,
                If 'positive', only positive set is given.
    '''
    def __init__(self,protein_feature = 'paac',
                 set_type = 'random',
                 ratio = 0.2,
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
        ratio: {None, float, list}, (default = 0.2): used to split data 
                into train, test, validation sets as given values. If left None, 
                only X and y data can be obtained while float value gives train 
                and test set. If ratio = a (float), then test will be a% of total 
                data size. If ratio = [a,b] where a and b are in (0,1), 
                train, test and validation sets are formed according to them. For example, 
                If a = 0.2 and b = 0.1, train fraction is 0.7, test fraction is 0.2 
                and validation fraction is 0.1 of all dataset size. If set_type = 'temporal', 
                then ratio = None automatically.
        protein_faeture: {'paac','aac','gaac','ctriad','ctdt','soc_number','kpssm'},
                (default = 'paac'): numerical features of protein sequences
        set_type: {'random','similarity','temporal'}, (default = 'random'):
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
                 ratio = 0.2,
                 label = None,
                 pre_determined = False):
        
        super().__init__(ratio = ratio, protein_feature = protein_feature,
                         set_type = set_type,label = label,
                         pre_determined = pre_determined, main_set = 'go_dataset')       

class SelfGet(data_importer.casual_importer):
    
    '''
    Description:
        This function is to provide users to import their datasets with
        specified delimiter. The format of data should be like that if 
        delimiter is comma separated and name == True:
            
            Name(or ID),feature_1,feature_2,...,feature_n
            Name(or ID),feature_1,feature_2,...,feature_n
            Name(or ID),feature_1,feature_2,...,feature_n
        
    Parameters:
        delimiter: default = "\t", a character to separate columns in file.
        name: type = bool, default = False, If True, then first colmun
            is considered as name of inputs else the first column is a 
            feature column.
        label: type = bool, default = False, If True, then last colmun
            is considered as label of inputs else the last column is a 
            feature column. 
        
    '''
    
    def __init__(self, delimiter = '\t', name = False, label = False):
        
        super().__init__(delimiter = delimiter, name = name, label = label)

