import sys
from pathlib import Path
sys.path.append(str(Path('').resolve()))
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from src.hdpview.dataset import Dataset
from src.hdpview.workload_generator import *


from src.competitors_runs import run_hdmm 



for name_dataset_exp in tqdm(["nume-adult","trafic","electricity","bitcoin"],"Dataset loop : ",leave=True):

    root_dir = Path('__file__').resolve().parent
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
                    
