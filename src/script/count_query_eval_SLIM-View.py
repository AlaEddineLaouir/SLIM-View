import sys
sys.path.append('../../src')

import pandas as pd

from tqdm import tqdm
from pathlib import Path

import argparse
from dataset import Dataset
from dataset import Dataset
from workload_generator import *


from ours import SDPCube


if __name__ == "__main__":

        parser = argparse.ArgumentParser(description='Evaluate count range queries for different datasets.')
        parser.add_argument('--dataset', type=str, help="used dataset [adult, small-adult, nume-adult, trafic, bitcoin, electricity, phoneme, jm, adding-adult] (default: adult)", default="adult")
        parser.add_argument('--alg', type=str, help="used algorithm [all, hdpview, dawa, hdmm, identity, privbayes] (default:  all)", default="all")
        parser.add_argument('--epsilon', type=float, help="privacy budget (default: 1.0)", default=1.0)
        parser.add_argument('--workload', type=str, help="workloads that are used in experiments [small_set, all] (default: all)", default="all")
        parser.add_argument('--times', type=int, help="number of synthetic dataset (default: 10)", default=10)
        args = parser.parse_args()


        for name_dataset_exp in tqdm(["bitcoin","traffic","electricity","nume-adult"],"Dataset loop : ",leave=True):

            root_dir = Path('__file__').resolve().parent.parent.parent
            data_dir = root_dir / "data" / "preprocessed" / name_dataset_exp

            data = Dataset.load(data_dir / 'data.csv', data_dir / 'domain.json')

            epsilon = 0.1
            delta = 0.001

        

            direct_workloads = [] #  using direct representation, instead using matrix representation for range query for speed up
            implicit_workloads = []

        
                
            for i in range(1):
                direct_workloads.append(direct_random_range_queries(domain=data.domain, size=3000, dim_size=2, seed=i))
                direct_workloads.append(direct_random_range_queries(domain=data.domain, size=3000, dim_size=3, seed=i))
                direct_workloads.append(direct_random_range_queries(domain=data.domain, size=3000, dim_size=4, seed=i))
                direct_workloads.append(direct_random_range_queries(domain=data.domain, size=3000, dim_size=5, seed=i))
                direct_workloads.append(direct_random_range_queries(domain=data.domain, size=3000, dim_size=6, seed=i))

            nbr_dims = 2
            for i in tqdm(range(len(direct_workloads)),"direct-workload loop : ",leave=False):
        
                    direct_workload_name, direct_workload = direct_workloads[i]

                    rmse_g,re_g,compressions,view_times,workloads_times = SDPCube.run(data,0.4,0.9,direct_workload,0.5,epsilon,delta,0.5,name_dataset_exp,direct_workload_name,False,80,False)
                    data_sensi = {
                                        "RMSE" : rmse_g,
                                        "RE" : re_g,
                                        "CR" : compressions,
                                        "ET" : view_times,
                                    }
                    df = pd.DataFrame(data_sensi)
                    df.to_csv("exp-SLIM-VIEW-"+name_dataset_exp+"-"+direct_workload_name+"-Synth.csv")

                    