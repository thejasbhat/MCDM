# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 07:36:44 2021

@author: thejas_bhat
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from vectors import vector_projection

class mcdm_solver:
    def __init__(self,df, criterias):
        self.list_of_available_methods = ['projection_pareto','efficient_frontier_projection']
        self.df = df
        self.criterias = criterias
        self.method = None
        self.solver = None
        
    def set_method(self,method_name,**kwargs):
        if method_name in self.list_of_available_methods:
            self.method = method_name
            if method_name == 'projection_pareto':
                self.solver = self.projection_pareto(self.df.copy(), self.criterias, kwargs)
            elif method_name == 'efficient_frontier_projection':
                self.solver = self.efficient_frontier_projection(self.df.copy(), self.criterias, kwargs)
            else:
                print("No solver set")
        else:
            print("Incorrect Solver name, choose from \n{}".format("\n".join(self.list_of_available_methods)))
        return
    def get_strength(self):
        if self.solver is None:
            print("method not selected, use 'set_method' to choose solver")
        else:
            return self.solver._get_alpha_()
        return
    def set_strength(self,num):
        if self.solver is None:
            print("method not selected, use 'set_method' to choose solver")
        else:
            self.solver._set_alpha_(num)
        return
    def get_strength_depreciator(self):
        if self.solver is None:
            print("method not selected, use 'set_method' to choose solver")
        else:
            return self.solver._get_ratio_()
        return
    def set_strength_depreciator(self,num):
        if self.solver is None:
            print("method not selected, use 'set_method' to choose solver")
        else:
            self.solver._set_ratio_(num)
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
            return self.df.loc[self.df.nsmallest(min(n,self.df.shape[0]),
                                                "decision").index,cols+['decision']]
        
    
    class projection_pareto:
        def __init__(self,df, criterias, dict_args):
            #arg_order = [df, criterias,
            #           benefit = list_of_boolean -->indicating crierias are ranked in descending order or not,
            #           priority_levels=None, elements_in_levels=None, alpha =0.01, ratio=0.5]
            benefit = False if 'benefit' not in dict_args.keys() else dict_args['benefit']
            levels = []
            col_order = []
            for i in range(len(criterias)):
                if 'P'+str(i+1) in dict_args.keys():
                    dict_val = dict_args['P'+str(i+1)]
                    if type(dict_val) == str:
                        dict_val = [dict_val]
                    elif type(dict_val) == list:
                        pass
                    else:
                        print("Priotity accepts list of column names or just one column name as string")
                        return
                    levels.append(dict_val)
                    col_order+=dict_val
                else:
                    break
            col_order = col_order + [x for x in criterias if x not in col_order]
            self.priority_levels = len(levels)
            self.elements_in_levels = [len(x) for x in levels] if len(levels) !=0 else 0
            self.alpha = 0.01 if 'strength' not in dict_args.keys() else dict_args['alpha']
            self.ratio = 0.5 if 'dep_ratio' not in dict_args.keys() else dict_args['dep_ratio']
            self.df = df[[x for x in df.columns if x not in col_order]+col_order]
            self.criterias = col_order
            self.vector = None
            if type(benefit) == bool:
                self.ascending = [not benefit for x in range(len(criterias))]
            elif type(benefit) == list:
                if len(benefit)==len(criterias):
                    self.ascending = [not x for x in benefit]
                else:
                    print("Length of beneft should be equal to number of criterias")
                    return
            else:
                print("Incorrect benefit data type, expected boolean or list of boolean of criteria length")
                return


        def count_pareto_dominantion(self,df, obj_cols):
            #df.sort_values(by=rank_cols[0], inplace = True)
            aa = df[obj_cols].values
            aa= aa.astype(int)
            bb = np.dstack([aa]*aa.shape[0])
            cc = np.swapaxes(bb,0,2)
            dd = np.greater_equal(bb,cc)
            ee = np.all(dd,axis=1)
            return np.sum(ee, axis=0)-1
        def _get_ratio_(self):
            return self.ratio
        def _set_ratio_(self, num):
            self.ratio = num
            return 
        def _get_alpha_(self):
            return self.alpha
        def _set_alpha_(self,num):
            self.alpha = num
        def _solve_(self):
            if len(self.criterias) == 1:
                self.df["decision"] = self.df[self.criterias[0]].rank(method = "dense",
                       ascending = self.ascending[0])
                return self.df["decision"]
            else:   
                self.vector = vector_projection(dim=len(self.criterias),
                                                priority_levels=self.priority_levels,
                                                elements_in_levels=self.elements_in_levels,
                                                alpha =self.alpha, ratio=self.ratio)
                priority = False
                if self.priority_levels ==0 and self.elements_in_levels ==0:
                    self.vector.generate_equal_imp_unit_vector()
                else:
                    self.vector.generate_priority_unit_vector()
                    priority = True
                rank_cols = [x+"_rank" for x in self.criterias]
                for idx, ele in enumerate(self.criterias):
                    self.df[ele+"_rank"] = self.df[ele].rank(method = "dense", ascending = self.ascending[idx])
                self.df["pareto_domination"] = self.count_pareto_dominantion(self.df,obj_cols= rank_cols)
                self.df["vector_projections"] = self.vector.compute_vector_projection(self.df, cols=rank_cols,
                                                                                      priority=priority)
                sort_cols = ['vector_projections', 'pareto_domination']
                tups = self.df[sort_cols].sort_values(sort_cols, ascending=[True,False]).apply(tuple, 1)
                f, i = pd.factorize(tups)
                factorized = pd.Series(f + 1, tups.index)
                self.df = self.df.assign(Rank=factorized)
                return self.df["Rank"]
        
    class efficient_frontier_projection:
        def __init__(self,df, criterias, dict_args):
            #arg_order = [df, criterias,
            #           benefit = list_of_boolean -->indicating crierias are ranked in descending order or not,
            #           priority_levels=None, elements_in_levels=None, alpha =0.01, ratio=0.5]
            benefit = False if 'benefit' not in dict_args.keys() else dict_args['benefit']
            levels = []
            col_order = []
            for i in range(len(criterias)):
                if 'P'+str(i+1) in dict_args.keys():
                    dict_val = dict_args['P'+str(i+1)]
                    if type(dict_val) == str:
                        dict_val = [dict_val]
                    elif type(dict_val) == list:
                        pass
                    else:
                        print("Priotity accepts list of column names or just one column name as string")
                        return
                    levels.append(dict_val)
                    col_order+=dict_val
                else:
                    break
            col_order = col_order + [x for x in criterias if x not in col_order]
            self.priority_levels = len(levels)
            self.elements_in_levels = [len(x) for x in levels] if len(levels) !=0 else 0
            self.alpha = 0.01 if 'strength' not in dict_args.keys() else dict_args['alpha']
            self.ratio = 0.5 if 'dep_ratio' not in dict_args.keys() else dict_args['dep_ratio']
            self.df = df[[x for x in df.columns if x not in col_order]+col_order]
            self.criterias = col_order
            self.vector = None
            if type(benefit) == bool:
                self.ascending = [not benefit for x in range(len(criterias))]
            elif type(benefit) == list:
                if len(benefit)==len(criterias):
                    self.ascending = [not x for x in benefit]
                else:
                    print("Length of beneft should be equal to number of criterias")
                    return
            else:
                print("Incorrect benefit data type, expected boolean or list of boolean of criteria length")
                return
        def _get_ratio_(self):
            return self.ratio
        def _set_ratio_(self, num):
            self.ratio = num
            return 
        def _get_alpha_(self):
            return self.alpha
        def _set_alpha_(self,num):
            self.alpha = num

        def count_pareto_dominanted(self,df, obj_cols):
            #df.sort_values(by=rank_cols[0], inplace = True)
            aa = df[obj_cols].values
            aa= aa.astype(int)
            bb = np.dstack([aa]*aa.shape[0])
            cc = np.swapaxes(bb,0,2)
            dd = np.less(bb,cc) 
            ee = np.all(dd,axis=1)
            return np.sum(ee, axis=0)
        def _solve_(self):
            if len(self.criterias) == 1:
                self.df["decision"] = self.df[self.criterias[0]].rank(method = "dense",
                       ascending = self.ascending[0])
                return self.df["decision"]
            else:   
                self.vector = vector_projection(dim=len(self.criterias),
                                                priority_levels=self.priority_levels,
                                                elements_in_levels=self.elements_in_levels,
                                                alpha =self.alpha, ratio=self.ratio)
                priority = False
                if self.priority_levels ==0 and self.elements_in_levels ==0:
                    self.vector.generate_equal_imp_unit_vector()
                else:
                    self.vector.generate_priority_unit_vector()
                    priority = True
                rank_cols = [x+"_rank" for x in self.criterias]
                for idx, ele in enumerate(self.criterias):
                    self.df[ele+"_rank"] = self.df[ele].rank(method = "dense", ascending = self.ascending[idx])
                self.df["vector_projections"] = self.vector.compute_vector_projection(self.df, cols=rank_cols,
                                                                                      priority=priority)
                self.df["Rank"] = None
                next_rank = 1
                rank_na_count = np.sum(self.df['Rank'].isna())
                while rank_na_count != 0:
                    perc_na_count = round((self.df.shape[0]-rank_na_count)*100/self.df.shape[0])
                    if perc_na_count%5==0:
                        print("Completed Processing {}% of data".format(perc_na_count))
                    itr_df = self.df[self.df['Rank'].isna()]
                    itr_df["pareto_domination"] = self.count_pareto_dominanted(itr_df,obj_cols= rank_cols)
                    efficient_frontier = itr_df[itr_df["pareto_domination"]==0]
                    mask = efficient_frontier['vector_projections'] == efficient_frontier['vector_projections'].min()
                    rank_indexs = efficient_frontier[mask].index.tolist()
                    self.df.loc[rank_indexs,'Rank'] = next_rank
                    next_rank +=1
                    rank_na_count = np.sum(self.df['Rank'].isna())
                return self.df["Rank"]
#############################  Execution ######################################
if __name__ != "__main__":  
    #df = pd.DataFrame(np.random.rand(500,5))
    #df.columns = ['a','b','c','d','e']
    #df = pd.DataFrame({'a':[6,8,3,2,10,4,7,1,5,9],'b':[10-x+1 for x in [8,7,6,1,9,5,10,3,4,2]]})
    files = ['magical_telescope_dataset','adult_dataset','all_dataset_avg']
    for var in files:
        df = pd.read_csv(var+'.csv',index_col=0)
        a = mcdm_solver(df, [x for x in df.columns if x != "Classifiers"])
        a.set_method("efficient_frontier_projection",benefit=[True, True, False, True, True, True, True, True, False, False])#,P1 = a.criterias[:-2])#,P1 = a.criterias[:-2]    
        a.solve()
        a.solver.df.to_csv(var+'_efficient_frontier_projection.csv')
    method = ['TOPSIS','GRA','VIKOR','PROMETHEE_II','ELECTRE_III']
    for var in files:
        print("Dataset ")
        df = pd.read_csv(var+'_mcdm_results.csv',index_col=0)
        result = pd.read_csv(var+'_efficient_frontier_projection_priority.csv',index_col=0)
        df = df.merge(result[['Classifiers','Rank']],on='Classifiers',how='inner')
        for meth in method:
            print("correlation of _efficient_frontier_projection_priority vs",meth,np.corrcoef(df[meth+'-Rank'].values,df['Rank'].values)[1,0])
    a.get_strength()
    a.set_strength(0.1)
    a.get_strength_depreciator()
    a.set_strength_depreciator(0.1)
    a.select_top_n_entities(5,df.columns.to_list())
#from mcdm import mcdm_solver
#a = mcdm_solver()