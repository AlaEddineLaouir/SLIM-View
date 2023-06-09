import sys
from pathlib import Path
sys.path.append(str(Path('').resolve()))
import time

import subprocess

from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import math




def run_priv_bayes( dataset_org,workloads,synth=False):
    rmse_g = []
    re_g=[]
    view_times =[]
    workloads_times=[]

    for indice in tqdm(range(len(workloads)),"workload loop : ",leave=True):
        workload  = workloads[indice]
        dataset = dataset_org.df.copy()
        dims = get_attribut_qwl(workload)
        columns_to_drop = [index for index,value in enumerate(dataset.columns) if value not in dims]
        columns_to_not_drop = [index for index,value in enumerate(dataset.columns) if value in dims]
    
        global_dataset =[]
        # (1) Constructing the dataset that will have the same dims as the QWL
        if synth:
            dataset['Mesure'] = np.random.randint(0,10, size=dataset.shape[0])
            global_dataset = dataset.copy()
        else:
            
            dataset['Mesure'] = 1
            dataset = dataset.drop(dataset.columns[columns_to_drop], axis=1) 
            global_dataset = dataset.groupby(dims).sum().reset_index()
            #print(global_dataset.shape)

        
        mins_d =[global_dataset[dim].min() for dim in dims]
        maxs_d =[global_dataset[dim].max() for dim in dims]
        
        mins_d.append(global_dataset['Mesure'].min())
        maxs_d.append(global_dataset['Mesure'].max())

        rmse =0
        re =0
        algo_exe_time = 0
        query_exe_times_true = [0] * len(workload)
        query_exe_times_est = [0] * len(workload)
        for i in tqdm(range(10),"Iteration loop : ",leave=True):
            # All data is saved in file, so PrivBayes C++ can collect them
            root_dir = Path('__file__').resolve().parent
            data_dir = root_dir / "data" / "preprocessed/gendim/raw.csv"
            
            global_dataset.to_csv(data_dir, header=False,index=False)
            metadata_dir = root_dir / "data" / "preprocessed" / "gendim" / "domain_info.txt"
            
            with open(metadata_dir, 'w') as outp:
                for i in range(len(dims)+1):
                    outp.write("C "+ str(float(mins_d[i])) +" "+ str(float(maxs_d[i]))+"\n")
            
            start_time = time.time()

            algo_dir = root_dir / "src" / "competitors_runs" / "lunch_priv_bayes_cpp.sh"
            subprocess.call(algo_dir)
            algo_exe_time += (time.time()-start_time)/10
            result_dir = root_dir / "data" / "preprocessed" / "gendim" / "raw_privbayes.csv"
            perturbed_df = pd.read_csv(result_dir)
            perturbed_df.columns = global_dataset.columns
            
            # (8) Evaluating the algorithm (Using RMSE mesure)
            est =[]
            true =[]

            for index, query in enumerate(workload):
                res_q, time_exe = run_query_on_df(global_dataset,query)
                true.append(res_q)
                query_exe_times_true[index] += time_exe/10
                res_q, time_exe = run_query_on_df(perturbed_df,query)
                est.append(res_q)
                query_exe_times_est[index] += time_exe/10
            
           
        
            est = np.array(est)
            true = np.array(true)
            re += ((np.abs(est - true))/np.array([np.maximum(1,x) for x in true]))/10
            rmse_1 = math.sqrt(np.mean(np.square(est - true)))
            rmse += rmse_1/10
           

            
        rmse_g.append(rmse)
        re_g.append(re)
        view_times.append(algo_exe_time)
        workloads_times.append([query_exe_times_est,query_exe_times_est])
    return rmse_g,re_g,view_times,workloads_times







def run_query_on_df(df,query):
    start_time = time.time()
    boolean_index_org = (df[query.conditions[0].attribute] >= query.conditions[0].start) & (df[query.conditions[0].attribute] <= query.conditions[0].end)
    for cond in query.conditions[1:]:
        boolean_index_org = boolean_index_org & (df[cond.attribute] >= cond.start) & (df[cond.attribute] <= cond.end )           
    res = df[boolean_index_org]['Mesure'].sum()
    end_time = time.time()
    return res, end_time-start_time
def get_attribut_qwl(workload):
    dims = []
    for cond in workload[0].conditions:
        dims.append(cond.attribute)
    print(dims)
    return dims