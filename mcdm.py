# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 07:36:44 2021

@author: thejas_bhat
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class mcdm_solver:
    def __init__(self,df, criterias):
        self.list_of_available_methods = ['uniform_pareto','priority_pareto']
        self.df = df
        self.criterias = criterias
        self.method = None
        self.solver = None
        
    def set_method(self,method_name,*args):
        if method_name in self.list_of_available_methods:
            self.method = method_name
            if method_name == 'uniform_pareto':
                self.solver = self.uniform_pareto(self.df.copy(), self.criterias, args)
            elif method_name == 'priority_pareto':
                self.solver = self.priority_pareto(self.df.copy(), self.criterias, args)
            else:
                print("No solver set")
        else:
            print("Incorrect Solver name, choose from \n{}".format("\n".join(self.list_of_available_methods)))
        return
    
    def solve(self):
        if self.solver is None:
            print("method not selected, use 'set_method' to choose solver")
        else:
            self.df["decision"] = self.solver._solve_()
        return
    def select_top_n_entities(self, n, cols):
        if "decision" not in self.df.columns:
            print("solver hasn't been run, use 'solve' method first")
        else:
            return self.df.loc[self.df.nlargest(min(n,self.df.shape[0]),
                                                "decision").index,cols+['decision']]
        
    class uniform_pareto:
        def __init__(self,df, criterias, list_args):
            #arg_order = [df, criterias,
            #           list_of_boolean -->indicating crierias are ranked in ascending order or not,
            #           ]
            self.df = df
            self.criterias = criterias
            if type(list_args[0]) == bool:self.ascending = [list_args[0] for x in range(len(criterias))]
            else:self.ascending = list_args[0]

        def count_pareto_dominantion(self,df, obj_cols):
            #df.sort_values(by=rank_cols[0], inplace = True)
            aa = df[obj_cols].values
            aa= aa.astype(int)
            bb = np.dstack([aa]*aa.shape[0])
            cc = np.swapaxes(bb,0,2)
            dd = np.greater_equal(bb,cc) 
            ee = np.all(dd,axis=1)
            return np.sum(ee, axis=0)
        def _solve_(self):
            if len(self.criterias) == 1:
                self.df["decision"] = self.df[self.criterias[0]].rank(method = "dense",
                       ascending = self.ascending[0])
                return self.df["decision"]
            else:
                rank_cols = [x+"_rank" for x in self.criterias]
                for idx, ele in enumerate(self.criterias):
                    self.df[ele+"_rank"] = self.df[ele].rank(method = "dense", ascending = self.ascending[idx])
                self.df["decision"] = self.count_pareto_dominantion(self.df,obj_cols= rank_cols)
                return self.df["decision"]
#############################  Execution ######################################
if __name__ == "__main__":  
    df = pd.DataFrame(np.random.rand(500,5))
    df.columns = ['a','b','c','d','e']
    a = mcdm_solver(df, df.columns)
    a.set_method("uniform_pareto",False)    
    a.solve()
    a.select_top_n_entities(5,df.columns.to_list())
#from mcdm import mcdm_solver
#a = mcdm_solver()