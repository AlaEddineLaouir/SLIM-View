import sys
sys.path.append('../../src')
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from pathlib import Path
import pickle
import json
import multiprocessing
import argparse
from dataset import Dataset
from dataset import Dataset
from count_table import CountTable
from workload_generator import *

from algorithms.identity import Identity
from algorithms.dawa import Dawa
from algorithms.noised_data import SynData
from ours import SDPCube
from ours import hdp_view_run
from ours import dawa_run
from ours import privBayes_run
from ours import run_privat_tree
from ours import run_hdmm 

from hdmm import error

parser = argparse.ArgumentParser(description='Evaluate count range queries for different datasets.')
parser.add_argument('--dataset', type=str, help="used dataset [adult, small-adult, nume-adult, trafic, bitcoin, electricity, phoneme, jm, adding-adult] (default: adult)", default="adult")
parser.add_argument('--alg', type=str, help="used algorithm [all, hdpview, dawa, hdmm, identity, privbayes] (default:  all)", default="all")
parser.add_argument('--epsilon', type=float, help="privacy budget (default: 1.0)", default=1.0)
parser.add_argument('--workload', type=str, help="workloads that are used in experiments [small_set, all] (default: all)", default="all")
parser.add_argument('--times', type=int, help="number of synthetic dataset (default: 10)", default=10)
args = parser.parse_args()

for name_dataset_exp in tqdm(["trafic","electricity","nume-adult"],"Dataset loop : ",leave=True):

    root_dir = Path('__file__').resolve().parent.parent.parent
    data_dir = root_dir / "data" / "preprocessed" / name_dataset_exp

    data = Dataset.load(data_dir / 'data.csv', data_dir / 'domain.json')

    epsilon = 0.1

    if __name__ == '__main__':

        direct_workloads = [] #  using direct representation, instead using matrix representation for range query for speed up
        implicit_workloads = []

    
            
        for i in range(1):
            implicit_workloads.append(implicit_random_range_queries(domain=data.domain, size=3000, dim_size=2, seed=i))
            implicit_workloads.append(implicit_random_range_queries(domain=data.domain, size=3000, dim_size=3, seed=i))
            implicit_workloads.append(implicit_random_range_queries(domain=data.domain, size=3000, dim_size=4, seed=i))
            implicit_workloads.append(implicit_random_range_queries(domain=data.domain, size=3000, dim_size=5, seed=i))
            implicit_workloads.append(implicit_random_range_queries(domain=data.domain, size=3000, dim_size=6, seed=i))

        for implicit_workload_name, implicit_workload in implicit_workloads:
            rmse_g,re_g,view_times = run_hdmm.run(data,"dawa",implicit_workload,epsilon,False)
            data_sensi = {
                                    "RMSE" : rmse_g,
                                    "RE" : re_g,
                                    "ET" : view_times
                                }
            df = pd.DataFrame(data_sensi)
            df.to_csv("exp-dawa-"+name_dataset_exp+"-"+implicit_workload_name+".csv")
            
            rmse_g,re_g,view_times = run_hdmm.run(data,"hdmm",implicit_workload,epsilon,False)
            data_sensi = {
                                    "RMSE" : rmse_g,
                                    "RE" : re_g,
                                    "ET" : view_times
                                }
            df = pd.DataFrame(data_sensi)
            df.to_csv("exp-hdmm-"+name_dataset_exp+"-"+implicit_workload_name+".csv")
            
            rmse_g,re_g,view_times = run_hdmm.run(data,"identity",implicit_workload,epsilon,False)
            data_sensi = {
                                    "RMSE" : rmse_g,
                                    "RE" : re_g,
                                    "ET" : view_times
                                }
            df = pd.DataFrame(data_sensi)
            df.to_csv("exp-identity-"+name_dataset_exp+"-"+implicit_workload_name+".csv")



            

        # nbr_dims = 2
        # for i in tqdm(range(len(direct_workloads)),"direct-workload loop : ",leave=False):
    
        #         direct_workload_name, direct_workload = direct_workloads[i]

        # #         rmse_g,re_g,compressions,view_times,workloads_times = run_privat_tree.run(data,epsilon,direct_workload,args.dataset,direct_workload_name,False)
        # #         data_sensi = {
        # #                             "RMSE" : rmse_g,
        # #                             "RE" : re_g,
        # #                             "CR" : compressions,
        # #                             "ET" : view_times,
        # #                         }
        # #         df = pd.DataFrame(data_sensi)
        # #         df.to_csv("exp-PrivTree-"+name_dataset_exp+"-"+direct_workload_name+".csv")

        # #         rmse_g,re_g,compressions,view_times,workloads_times = hdp_view_run.run(data,epsilon,direct_workload,args.dataset,direct_workload_name,False)
        # #         data_sensi = {
        # #                             "RMSE" : rmse_g,
        # #                             "RE" : re_g,
        # #                             "CR" : compressions,
        # #                             "ET" : view_times,
        # #                         }
        # #         df = pd.DataFrame(data_sensi)
        # #         df.to_csv("exp-HDPView-"+name_dataset_exp+"-"+direct_workload_name+".csv")

        #         # rmse_g,re_g,view_times,workloads_times = privBayes_run.run_priv_bayes(data,direct_workload,epsilon,args.dataset,direct_workload_name,False)
        #         # data_sensi = {
        #         #                     "RMSE" : rmse_g,
        #         #                     "RE" : re_g,
        #         #                     "ET" : view_times,
        #         #                 }
        #         # df = pd.DataFrame(data_sensi)
        #         # df.to_csv("exp-PrivBayes-"+name_dataset_exp+"-"+direct_workload_name+".csv")

        #         rmse_g,re_g,compressions,view_times,workloads_times = SDPCube.run(data,0.4,0.9,direct_workload,0.2,epsilon,args.dataset,direct_workload_name,False,80)
        #         data_sensi = {
        #                             "RMSE" : rmse_g,
        #                             "RE" : re_g,
        #                             "CR" : compressions,
        #                             "ET" : view_times,
        #                         }
        #         df = pd.DataFrame(data_sensi)
        #         df.to_csv("exp-DPCEM-"+name_dataset_exp+"-"+direct_workload_name+".csv")



                ###################### SENSITIVITY TEST ######################################################
                # 
                # 
                #  
                # for min_allo in tqdm(np.arange(0.4,0.5,0.1),"min allocation loop : ",leave=False):
                #     for max_allo in tqdm(np.arange(0.9,1.0,0.1),"max-allocation loop : ",leave=False):
                #         for epsilon in tqdm(np.arange(0.1,1.1,0.1),"epsilon loop :",leave=False):
                #             for budget_rate in tqdm(np.arange(0.2,0.8,0.1),"budget-rate loop :",leave=False):
                #                 rmse_g,re_g,compressions,view_times,workloads_times = SDPCube.run(data,min_allo,max_allo,direct_workload,budget_rate,epsilon,args.dataset,direct_workload_name,False,80)
                #                 data_sensi = {
                #                     "NBRDIMS" : nbr_dims,
                #                     "RMSE" : rmse_g,
                #                     "RE" : re_g,
                #                     "CR" : compressions,
                #                     "ET" : view_times,
                #                     "MIN" : min_allo,
                #                     "MAXS" : max_allo,
                #                     "EPS" : epsilon,
                #                     "RATIO" : budget_rate
                #                 }
                #                 df = pd.DataFrame(data_sensi)

                #                 if result_df is None:
                #                     result_df = df
                #                 else:
                #                     result_df = pd.concat([result_df,df])
                                
                #                 result_df.to_csv("sensi-dpc-exp-"+args.dataset+"-dim"+str(nbr_dims)+".csv")
                
                # nbr_dims += 1
                    
