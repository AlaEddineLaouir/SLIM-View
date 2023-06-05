import queue
import sys
sys.path.append('../../src')
import time
import pickle
from functools import reduce
import pandas as pd
import numpy as np
import math
from dataset import Dataset
from count_table import CountTable
from algorithms import privtree
from domain import Domain
from ektelo.algorithm.dawa.partition_engines import l1partition

from hdmm import error
from hdmm import templates
from workload_generator import Pmatrix
from tqdm import tqdm
from hdmm.matrix import EkteloMatrix, VStack, Kronecker, Weighted
from hdmm import workload

def run(dataset_org,method,workloads,epsilon,synth):

    prng = np.random.RandomState(0)

    rmse_g = []
    re_g= []
    view_times =[]
    workloads_times=[]
    for indice in tqdm(range(len(workloads))):
        workload  = workloads[indice]

        dataset = dataset_org.df.copy()
        dataset_shape = dataset_org.domain.shape
        #workload  = workload[0]
        dims = list(workload.attrs)
        columns_to_drop = [index for index,value in enumerate(dataset.columns) if value not in dims]
        columns_to_not_drop = [index for index,value in enumerate(dataset.columns) if value in dims]


        shape =[]
        bins = []
        for i in columns_to_not_drop:
            bins.append(range(dataset_shape[i]+1))
            shape.append(dataset_shape[i])

        dataset_c = dataset.drop(dataset.columns[columns_to_drop], axis=1) 
        ans = np.histogramdd(dataset_c.values, bins)[0]
        x = ans.flatten()
    
        
        rmse = 0
        re = 0
        algo_exe_time = 0

        for i in tqdm(range(0,2)):

            # global_dataset = global_dataset_mesure.copy()
            # mesure = global_dataset['Mesure']
            # global_dataset.drop('Mesure', axis=1,inplace=True)

            W = workload
            x = np.transpose(x)
            x_res= None

            if method == "identity" :
                start = time.time()
                x_res = [elem + np.random.laplace(loc=0,scale=1/epsilon) for elem in x]
                x_res = np.array(x_res)
                algo_exe_time += (time.time()- start)/2
            
            if method == "hdmm":
                start = time.time()
                counts = [dataset_org.domain.shape[index] for index in columns_to_not_drop]
                domain = Domain(dims,counts)

                hdmm_template = templates.KronPIdentity(Pmatrix(domain), domain.shape)
                hdmm_template.optimize(workload)
            
                #rootmse = error.rootmse2(workload, hdmm_template.strategy(),x, eps=epsilon)
                A = hdmm_template.strategy()
                W, A = convert_implicit(W), convert_implicit(A)
                delta = A.sensitivity()
                AtA = A.gram()
                AtA1 = AtA.pinv()
                x_e = AtA * x
                x_ep = [elem + np.random.laplace(loc=0,scale=delta/epsilon) for elem in x_e]

                x_ep = np.array(x_ep)
                x_res = AtA1 * x_ep
                algo_exe_time += (time.time()- start)/2
            if method == "dawa":
                start = time.time()
                x_res, partition_num = dawaPartition(x, epsilon, 0.7, prng)
                algo_exe_time += (time.time()- start)/2

            

            x_res = np.transpose(x_res)

            est = W * x_res
            true = W * x

            est = np.array( est)
            true = np.array(true)

           

            est = np.array(est)
            true = np.array(true)
            re += (np.abs(est - true)/np.array([np.maximum(1,x) for x in true]))/2
            sq_err = np.square(est - true)
            rmse += math.sqrt(np.mean(sq_err))/2
        rmse_g.append(rmse)
        re_g.append(re)
        view_times.append(algo_exe_time)
    
    return rmse_g,re_g,view_times

            

    

def get_attribut_qwl(workload):
    dims = []
    for cond in workload[0].conditions:
        dims.append(cond.attribute)
    return dims
def convert_implicit(A):
    if isinstance(A, EkteloMatrix) or isinstance(A, workload.ExplicitGram):
        return A
    return EkteloMatrix(A)
def run_query_on_df(df,query):
    #start_time = time.time()
    boolean_index_org = (df[query.conditions[0].attribute] >= query.conditions[0].start) & (df[query.conditions[0].attribute] <= query.conditions[0].end)
    for cond in query.conditions[1:]:
        boolean_index_org = boolean_index_org & (df[cond.attribute] >= cond.start) & (df[cond.attribute] <= cond.end )           
    res = df[boolean_index_org]['Mesure'].sum()
    #end_time = time.time()
    return res#, end_time-start_time
def run_mat_q(Q,x):
    return np.sum([Q[i]*x[i] for i in range(len(Q))])


def dawaPartition(count_tensor, epsilon, ratio, prng):
    """Dawa partitioning with `dpcomp_core.algorithm.dawa`
    Args:
        count_tensor (np.array): raw data of count tensor
        epsilon (float): privacy budget
        ratio (float): budget ratio
        seed (int): random seed
    Returns:
        NoisedData
    """
    count_vector = count_tensor.ravel().astype('int')
    pSeed = prng.randint(1000000)
    # partitioning phase
    partition = l1partition.l1partition_approx_engine().Run(count_vector, epsilon, ratio, pSeed)
    partition_num = len(partition)
    # print('[DAWA] number of dawa partition: ', partition_num)
    # perturbation phase not optimized for workload
    noise_vector = prng.laplace(0.0, 1.0 / ((1-ratio) * epsilon), len(count_vector))
    for (start, end) in partition:
        if start != end:
            bucket_size = end+1-start
            noise_vector[start:end+1] = noise_vector[start] / bucket_size
            count_vector[start:end+1] = count_vector[start:end+1].sum() / bucket_size
    
    count_vector = count_vector.astype('float')
    count_vector += noise_vector

    return count_vector, partition_num



def mul_array(ar):
    mul = 1
    for e in ar:
        mul = mul*e
    return mul
