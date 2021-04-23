# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 16:39:37 2021

@author: maily
"""
import numpy as np
import pandas as pd

N = 2
Levels =2
elements_in_levels = [1,1]

alpha = 0.1
ratio = 0.5

def generate_equal_imp_unit_vector(dim):
    return 1/np.sqrt(dim)*np.ones(dim)

def generate_priority_unit_vector(dim, priority_levels, elements_in_levels, alpha, ratio):
    equal_imp_vec = np.ones(dim)
    if priority_levels > dim or np.sum(elements_in_levels) >dim:
        print('''Incorrect arguments,
              number of levels or sum of elements in all levels
              must not exceed dimension of vector''')
        return
    if priority_levels != len(elements_in_levels):
        print('''Length of elements in levels must be equal to 
              number of priority levels''')
        return
    init_idx = 0
    for idx, val in enumerate(elements_in_levels):
        equal_imp_vec[init_idx:init_idx+val] += alpha*(ratio**idx)
        init_idx+=val
    reduction = min(alpha*(ratio**(idx)),1)
    equal_imp_vec[init_idx:] -= reduction
    equal_imp_vec /= np.linalg.norm(equal_imp_vec)
    return equal_imp_vec
        
def compute_angles_with_basis_vectors(vec):
    vec /= np.linalg.norm(vec)
    return np.degrees(np.arccos(vec))

def count_pareto_dominantion(df, obj_cols, dominated = False):
    #df.sort_values(by=rank_cols[0], inplace = True)
    aa = df[obj_cols].values
    aa= aa.astype(int)
    bb = np.dstack([aa]*aa.shape[0])
    cc = np.swapaxes(bb,0,2)
    if dominated:
        dd = np.less(bb,cc)
        i=0
    else:
        dd = np.greater_equal(bb,cc)
        i=1
    ee = np.all(dd,axis=1)
    return np.sum(ee, axis=0)-i

def compute_vector_projection(vec, df, cols):
    return np.dot(df[cols].values, np.array(vec))
df = pd.DataFrame({'a':[6,8,3,2,10,5,7,1,5,9],'b':[8,7,6,1,9,5,10,3,4,2]})

cols = ['a', 'b']
tups = df[cols].sort_values(cols, ascending=[False,False]).apply(tuple, 1)
f, i = pd.factorize(tups)
factorized = pd.Series(f + 1, tups.index)

df.assign(Rank=factorized)

a = generate_equal_imp_unit_vector(dim=2)

print(compute_angles_with_basis_vectors(a))
b = generate_priority_unit_vector(dim=N, priority_levels=1,
                                  elements_in_levels=[1], alpha=0.01, ratio=0.8)
print(compute_angles_with_basis_vectors(b))

