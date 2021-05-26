# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 20:04:57 2021

@author: maily
"""
import numpy as np
import pandas as pd

class vector_projection:
        def __init__(self, dim, priority_levels=None, elements_in_levels=None, alpha =0.01, ratio=0.5):
            self.dim = dim
            self.equal_imp_unit_vec = None
            self.priority_unit_vec = None
            self.priority_levels = priority_levels
            self.elements_in_levels =elements_in_levels
            self.alpha = alpha
            self.depreciation_ratio = ratio
            return
        
        def generate_equal_imp_unit_vector(self):
            self.equal_imp_unit_vec = 1/np.sqrt(self.dim)*np.ones(self.dim)
            return
        
        def generate_priority_unit_vector(self):
            if any(y is None for y in [self.priority_levels, self.elements_in_levels]):
                print("required attributes:\n 1. priority levels \n 2. elements_in_levels")
                return
            equal_imp_vec = np.ones(self.dim)
            if self.priority_levels > self.dim or np.sum(self.elements_in_levels) >self.dim:
                print('''Incorrect arguments,
                      number of levels or sum of elements in all levels
                      must not exceed dimension of vector''')
                return
            if self.priority_levels != len(self.elements_in_levels):
                print('''Length of elements in levels must be equal to 
                      number of priority levels''')
                return
            init_idx = 0
            for idx, val in enumerate(self.elements_in_levels):
                equal_imp_vec[init_idx:init_idx+val] += self.alpha*(self.depreciation_ratio**idx)
                init_idx+=val
            reduction = min(self.alpha*(self.depreciation_ratio**(idx)),1)
            equal_imp_vec[init_idx:] -= reduction
            equal_imp_vec /= np.linalg.norm(equal_imp_vec)
            self.priority_unit_vec = equal_imp_vec
            return
                
        def compute_angles_with_basis_vectors(self,vec):
            vec /= np.linalg.norm(vec)
            return np.degrees(np.arccos(vec))
        
        def compute_vector_projection(self,  df, cols, priority=False):
            if priority:
                if self.priority_unit_vec is None:
                    try:
                        self.generate_priority_unit_vector()
                        return np.dot(df[cols].values, np.array(self.priority_unit_vec))
                    except:
                        print('priority vector not defined and unable to create')
                        return
                else:
                    return np.dot(df[cols].values, np.array(self.priority_unit_vec))
            else:
                if self.equal_imp_unit_vec is None:
                    self.generate_equal_imp_unit_vector()
                return np.dot(df[cols].values, np.array(self.equal_imp_unit_vec))
                    