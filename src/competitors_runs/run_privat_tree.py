import sys
sys.path.append('../../src')
import time
import pickle

import numpy as np

from tqdm import tqdm
import math
from src.hdpview.dataset import Dataset
from src.hdpview.count_table import CountTable
from src.algorithms import privtree
from src.hdpview.domain import Domain

def run(dataset_org,epsilon,workloads,synth):
    rmse_g = []
    re_g=[]
    compressions =[]
    view_times =[]
    workloads_times=[]
    for indice in tqdm(range(len(workloads)),"workload loop : ",leave=True):
        workload  = workloads[indice]
        

        dataset = dataset_org.df.copy()
        #workload  = workload[0]
        dims = get_attribut_qwl(workload)
        columns_to_drop = [index for index,value in enumerate(dataset.columns) if value not in dims]
        columns_to_not_drop = [index for index,value in enumerate(dataset.columns) if value in dims]
    

        global_dataset =[]
        # (1) Constructing the dataset that will have the same dims as the QWL
        if synth:
            dataset['Mesure'] = np.random.randint(0,2000, size=dataset.shape[0])
            dataset = dataset.drop(dataset.columns[columns_to_drop], axis=1)
            global_dataset = dataset.copy()
        else:
            
            dataset['Mesure'] = 1
            dataset = dataset.drop(dataset.columns[columns_to_drop], axis=1) 
            global_dataset = dataset.groupby(dims).sum().reset_index()
        

        global_dataset_mesure = global_dataset.copy()

        theta = 0
        fanout = 2**(len(dims))
        

        algo_execution_time =0
        final_data_ratio =0
        rmse = 0
        re=0
        query_exe_times_true = [0] * len(workload)
        query_exe_times_est = [0] * len(workload)

        for i in tqdm(range(10),"Iteration loop : ",leave=True):

            global_dataset = global_dataset_mesure.copy()
            mesure = global_dataset['Mesure']
            global_dataset.drop('Mesure', axis=1,inplace=True)

            

            counts = [dataset_org.domain.shape[index] for index in columns_to_not_drop]
            
            domain = Domain(global_dataset.columns,counts)

            config = dict(zip(global_dataset.columns,counts))


            global_Dataset = Dataset(global_dataset,domain)


            initial_block = CountTable.from_dataset(global_Dataset)
            initial_block.mesure_table = global_dataset_mesure.copy()
            #initial_block.info()
            start_time_view = time.time()
            p_view, block_result_list = privtree.run(initial_block, np.random.RandomState(0), epsilon=epsilon, fanout=fanout, theta=theta)
            end_time_view = time.time()

            final_data_ratio += (np.abs(len(p_view.blocks) - global_dataset.shape[0])/global_dataset.shape[0])/10

            algo_execution_time += (end_time_view - start_time_view)/10

            est = []
            true=[]
            for index,query in enumerate(workload):
                res,times = run_query_on_df(global_dataset_mesure,query)
                true.append(res)
                query_exe_times_true[index] = times/10

                q_start = time.time()
                res = p_view.run_query(query)
                q_end = time.time()
                est.append(res)
                query_exe_times_est[index] = (q_end-q_start)/10

            

            est = np.array(est)
            true= np.array(true)
            re += (np.abs(est - true)/np.array([np.maximum(1,x) for x in true]))/10
            sq_err = np.square(est - true)
            rmse += math.sqrt(np.mean(sq_err))/10
        
        rmse_g.append(rmse)
        re_g.append(re)
        view_times.append(algo_execution_time)
        compressions.append(final_data_ratio)
        workloads_times.append([query_exe_times_est,query_exe_times_est])
    
    return rmse_g,re_g,compressions,view_times,workloads_times

    
    
    
        

def get_attribut_qwl(workload):
    dims = []
    for cond in workload[0].conditions:
        dims.append(cond.attribute)
    return dims

def run_query_on_df(df,query):
    start_time = time.time()
    boolean_index_org = (df[query.conditions[0].attribute] >= query.conditions[0].start) & (df[query.conditions[0].attribute] <= query.conditions[0].end)
    for cond in query.conditions[1:]:
        boolean_index_org = boolean_index_org & (df[cond.attribute] >= cond.start) & (df[cond.attribute] <= cond.end )           
    res = df[boolean_index_org]['Mesure'].sum()
    end_time = time.time()
    return res, end_time-start_time
