import sys
from pathlib import Path
sys.path.append(str(Path('').resolve()))
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from pathlib import Path

import argparse
from src.hdpview import *
from src.hdpview.dataset import Dataset
from src.hdpview.workload_generator import *

from src.competitors_runs import hdp_view_run
from src.competitors_runs import run_privat_tree

if __name__ == '__main__':
    for name_dataset_exp in tqdm(["nume-adult","bitcoin","electricity"],"Dataset loop : ",leave=False):

        root_dir = Path('__file__').resolve().parent
        data_dir = root_dir / "data" / "preprocessed" / name_dataset_exp

        data = Dataset.load(data_dir / 'data.csv', data_dir / 'domain.json')

        epsilon = 0.1

        

        direct_workloads = [] #  using direct representation, instead using matrix representation for range query for speed up
        implicit_workloads = []

        
            
        for i in range(1):
                direct_workloads.append(direct_random_range_queries(domain=data.domain, size=3000, dim_size=2, seed=i))
                direct_workloads.append(direct_random_range_queries(domain=data.domain, size=3000, dim_size=3, seed=i))
                direct_workloads.append(direct_random_range_queries(domain=data.domain, size=3000, dim_size=4, seed=i))
                direct_workloads.append(direct_random_range_queries(domain=data.domain, size=3000, dim_size=5, seed=i))
                direct_workloads.append(direct_random_range_queries(domain=data.domain, size=3000, dim_size=6, seed=i))

        for j in tqdm(range(len(direct_workloads)),"direct-workload loop ",leave=True):
        
                    direct_workload_name, direct_workload = direct_workloads[j]
                    rmse_g,re_g,compressions,view_times,workloads_times = run_privat_tree.run(data,epsilon,direct_workload,True)
                    data_sensi = {
                                    "RMSE" : rmse_g,
                                    "RE" : re_g,
                                    "CR" : compressions,
                                    "ET" : view_times,
                                }
                    df = pd.DataFrame(data_sensi)
                    df.to_csv("exp-PrivTree-"+name_dataset_exp+"-"+direct_workload_name+".csv")

                    rmse_g,re_g,compressions,view_times,workloads_times = hdp_view_run.run(data,epsilon,direct_workload,True)
                    data_sensi = {
                                        "RMSE" : rmse_g,
                                        "RE" : re_g,
                                        "CR" : compressions,
                                        "ET" : view_times,
                                    }
                    df = pd.DataFrame(data_sensi)
                    df.to_csv("exp-HDPView-"+name_dataset_exp+"-"+direct_workload_name+".csv")

                