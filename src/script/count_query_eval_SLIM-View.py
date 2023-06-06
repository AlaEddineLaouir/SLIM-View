import sys
from pathlib import Path
sys.path.append(str(Path('').resolve()))

import pandas as pd

from tqdm import tqdm
from pathlib import Path

import argparse
from src.hdpview.dataset import Dataset
from src.hdpview.workload_generator import *


from src.SLIMView import SLIMView


if __name__ == "__main__":

        for name_dataset_exp in tqdm(["nume-adult","bitcoin","traffic","electricity"],"Dataset loop : ",leave=True):

            root_dir = Path('__file__').resolve().parent
            data_dir = root_dir / "data" / "preprocessed" / name_dataset_exp

            data = Dataset.load(data_dir / 'data.csv', data_dir / 'domain.json')

            epsilon = 0.1

        

            direct_workloads = [] 

        
                
            for i in range(1):
                direct_workloads.append(direct_random_range_queries(domain=data.domain, size=3000, dim_size=2, seed=i))
                direct_workloads.append(direct_random_range_queries(domain=data.domain, size=3000, dim_size=3, seed=i))
                direct_workloads.append(direct_random_range_queries(domain=data.domain, size=3000, dim_size=4, seed=i))
                direct_workloads.append(direct_random_range_queries(domain=data.domain, size=3000, dim_size=5, seed=i))
                direct_workloads.append(direct_random_range_queries(domain=data.domain, size=3000, dim_size=6, seed=i))

            nbr_dims = 2
            for i in tqdm(range(len(direct_workloads)),"direct-workload loop : ",leave=False):
        
                    direct_workload_name, direct_workload = direct_workloads[i]

                    rmse_g,re_g,compressions,view_times,workloads_times = SLIMView.run(data,0.05,direct_workload,0.5,epsilon,0.5,False)
                    data_sensi = {
                                        "RMSE" : rmse_g,
                                        "RE" : re_g,
                                        "CR" : compressions,
                                        "ET" : view_times,
                                    }
                    df = pd.DataFrame(data_sensi)
                    df.to_csv("exp-SLIM-VIEW-"+name_dataset_exp+"-"+direct_workload_name+"-Synth.csv")

                    