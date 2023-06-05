from copy import deepcopy
import random 
import sys

sys.path.append('../../src')
import time

import pickle

import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import gmpy2
from tqdm import tqdm
import math
import itertools
import matplotlib.pyplot as plt
import multiprocessing as mp
mp.set_start_method('fork')
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.factory import get_termination
from pymoo.optimize import minimize



def run(dataset_org, min_allo,max_allo,workloads, budget_sampling, epsilon,delta,lp_budget,dataset_name,exp_name,synth,lp_iter,random_sampling):
    
   
    rmse_g = []
    re_g=[]
    compressions =[]
    view_times =[]
    workloads_times=[]

    ep_lp = epsilon * lp_budget
    ep_sn = epsilon * (1-lp_budget)

    ep_s = budget_sampling * ep_sn
    ep_n = (1 - budget_sampling) * ep_sn

    for indice in tqdm(np.arange(1), " Workload loop : "):
        dataset = dataset_org.df.copy()
        workload  = workloads[indice]
        dims = get_attribut_qwl(workload)
        columns_to_drop = [index for index,value in enumerate(dataset.columns) if value not in dims]
    
        global_dataset =[]
        # (1) Constructing the dataset that will have the same dims as the QWL
        if synth:
            dataset['Mesure'] = np.random.randint(0,10, size=dataset.shape[0])
            dataset = dataset.drop(dataset.columns[columns_to_drop], axis=1)
            global_dataset = dataset.copy()
        else:
            
            dataset['Mesure'] = 1
            dataset = dataset.drop(dataset.columns[columns_to_drop], axis=1) 
            global_dataset = dataset.groupby(dims).sum().reset_index()
            #print(global_dataset.shape)
    
        # (2) Getting the region targeted by the workload
        qwl_df = None
        for query in workload:
                boolean_index = (global_dataset[query.conditions[0].attribute] >= query.conditions[0].start) & (global_dataset[query.conditions[0].attribute] <= query.conditions[0].end)
                for cond in query.conditions[1:]:
                    boolean_index = boolean_index & (global_dataset[cond.attribute] >= cond.start) & (global_dataset[cond.attribute] <= cond.end )
                region_df = global_dataset[boolean_index]
                
                if region_df.shape[0] > 1:
                    query.size = region_df.shape[0]
                    if query.size > 1 :
                        if qwl_df is None:
                            qwl_df = region_df
                        else:
                            qwl_df = pd.concat([qwl_df,region_df]).drop_duplicates()
        
        if qwl_df is None:
            continue
        
        global_dataset = qwl_df.copy()
        intial_data_size = sys.getsizeof(qwl_df)

        workload.sort(key=lambda query:query.size)
        workload_regions = [query for query in workload if query.size > 1]

        # To get the avg of th 10 expies
        algo_execution_time =0
        final_data_ratio =0
        rmse = 0
        re=0
        query_exe_times_true =[0] * len(workload)
        query_exe_times_est =[0] * len(workload)



        for i in tqdm(np.arange(10), " Iterations loop : "):
            qwl_df = global_dataset.copy()    
            # (3) Creating a df for each query, with delete from qwl_df to remove overlapping
            
            
            qwl_total_size = qwl_df.shape[0]
            qwl_df_divided = []
            qwl_regions =[]
            for region in workload_regions:
                if qwl_df.shape[0] > 1:
                    boolean_index = (qwl_df[region.conditions[0].attribute] >= region.conditions[0].start) & (qwl_df[region.conditions[0].attribute] <= region.conditions[0].end)
                    for cond in region.conditions[1:]:
                        boolean_index = boolean_index & (qwl_df[cond.attribute] >= cond.start) & (qwl_df[cond.attribute] <= cond.end )
                    region_df = qwl_df[boolean_index]
                    if region_df.shape[0] > 1:
                        qwl_df_divided.append(region_df)
                        qwl_regions.append(region)
                        qwl_df = qwl_df[np.logical_not(boolean_index)]

            start_time_view = time.time()        
            # (4) Optimisation phase
            

            allo_regions = get_allocation_for_regions(qwl_df_divided,ep_lp,ep_n,random_sampling)


            sampled_dfs=[]
            asso_data_s =[]

            for index,region in enumerate(qwl_regions):
                region_df = qwl_df_divided[index]
                allocation = allo_regions[index]
                sampled_df,asso_data = create_dp_view_asso_simple(region,region_df,allocation,ep_s,ep_n,delta,random_sampling,rate=min_allo)
                sampled_dfs.append(sampled_df)
                asso_data_s = asso_data + asso_data_s
            

            
            # # (5) Sampling phase and perturbation
            
            end_view_time = time.time()
            algo_execution_time += (end_view_time - start_time_view)/10

            # result_data_size = sys.getsizeof(perturbed_global_df)
            perturbed_global_df = pd.concat(sampled_dfs)
            association_table_df =pd.DataFrame(asso_data_s)
            cols =[[dim+'_s',dim+'_e'] for dim in dims] + [['noise']]
            cols = [item for sublist in cols for item in sublist]
            association_table_df.columns = cols
            
            original_size = global_dataset.shape[0] + len(qwl_regions)
            result_size = perturbed_global_df.shape[0] + association_table_df.shape[0]
            data_ratio = np.abs(original_size-result_size)
            data_ratio = data_ratio/original_size
            final_data_ratio += data_ratio/10
            
            _eval = time.time()
            # (8) Evaluating the algorithm (Using RMSE mesure)
            est =[]
            true =[]
            for index, query in enumerate(workload):
                res_q, time_exe = run_query_on_df(global_dataset,query)
                true.append(res_q)
                query_exe_times_true[index] += time_exe/10
                res_q= run_query_with_association(perturbed_global_df,association_table_df,query)
                est.append(res_q)
                query_exe_times_est[index] += time_exe/10
            
            _eval = time.time() - _eval
           
        
            est = np.array(est)
            true = np.array(true)
            re += np.mean(np.abs(est - true)/np.array([np.maximum(1,x) for x in true]))/10
            rmse_1 = math.sqrt(np.mean(np.square(est - true)))
            rmse += rmse_1/10

            
        rmse_g.append(rmse)
        re_g.append(re)
        view_times.append(algo_execution_time)
        workloads_times.append([query_exe_times_est,query_exe_times_est])
        compressions.append(final_data_ratio)
    
   
    return rmse_g,re_g,compressions,view_times,workloads_times

def get_attribut_qwl(workload):
    dims = []
    for cond in workload[0].conditions:
        dims.append(cond.attribute)
    return dims

def add_laplace_noise(x,epsilon):
    val = x + np.random.laplace(loc=0, scale=1/epsilon)
    return val


def get_association_table_empty(qwl_sub_empty,ep_n,delta=0):
    association_table=[]
    for sub in qwl_sub_empty:
        coord = dump_region_coordinates(sub)
        coord.append(add_laplace_noise(0,ep_n)/get_region_vol(sub))
        association_table.append(coord)
    
    return association_table





def run_query_on_df(df,query):
    start_time = time.time()
    boolean_index_org = (df[query.conditions[0].attribute] >= query.conditions[0].start) & (df[query.conditions[0].attribute] <= query.conditions[0].end)
    for cond in query.conditions[1:]:
        boolean_index_org = boolean_index_org & (df[cond.attribute] >= cond.start) & (df[cond.attribute] <= cond.end )           
    res = df[boolean_index_org]['Mesure'].sum()
    end_time = time.time()
    return res, end_time-start_time
def run_query_with_association(df,association_table,query):
    
    boolean_index_org = (df[query.conditions[0].attribute] >= query.conditions[0].start) & (df[query.conditions[0].attribute] <= query.conditions[0].end)
    for cond in query.conditions[1:]:
        boolean_index_org = boolean_index_org & (df[cond.attribute] >= cond.start) & (df[cond.attribute] <= cond.end )           
    res = df[boolean_index_org]['Mesure'].sum()
    
    noises =[]
    for _,row in association_table.iterrows():
        noises.append(get_intersection_noise(row,query))

    return res+np.sum(noises)

def get_intersection_noise(row,query):
    volume = 1
    for cond in query.conditions:
        table_s = row[cond.attribute+'_s']
        table_e = row[cond.attribute+'_e']
        if (cond.start <= table_s and cond.end >= table_s) or (cond.start >= table_s and cond.start <= table_e):
            volume = volume * np.abs((np.max([table_s,cond.start]) - np.min([table_e,cond.end])))
        else:
            return 0
    return volume * row['noise']


def dump_region_coordinates(region):
    coordinates= []
    for condition in region.conditions:
        coordinates = coordinates + [condition.start,condition.end]
    return coordinates



def expo_mech_sampling(ep_s,region_df,allo):
    
        ## Shuffle
        region_df = region_df.sample(frac=1)
        region_df.reset_index(drop=True, inplace=True)
        
        res_query = region_df['Mesure'].sum()
        
        possible_output_indexs = [ i for i in range(region_df.shape[0]-allo+1)]
        scores = [ gmpy2.mpz(score_starting_index_2dp(i,i+allo,region_df.copy())) for i in possible_output_indexs]
        

        ## Expo Mecha
        probabilities = []
        for score in scores:
            temp = ep_s * score
            temp = temp / (2 * 1)
            temp = gmpy2.exp(temp)
            probabilities.append(temp)
        p_S =[]
        for pro in probabilities:
            p_S.append(float(pro/np.linalg.norm(probabilities, ord=1)))

        p_S /= np.sum(p_S)
        res_index =  np.random.choice(possible_output_indexs, 1, p=p_S)[0]

        resulted_region_df = region_df.iloc[np.r_[res_index:res_index+allo]]
       
        
        return resulted_region_df

def score_starting_index_2dp(index_start,index_end,region_df):
    score = region_df.iloc[np.r_[index_start:index_end]]['Mesure'].sum()
    if score ==0:
        score = 0.000001
    return score



### For each region in para
def create_dp_view_asso_simple(region, region_df,allocation,ep_s,ep_n,delta,random_sampling,rate):
    allocation = np.min([region_df.shape[0]-1,allocation])
    nb_cut_points = np.sum([cond.end - cond.start + 1 for cond in region.conditions])
    nb = int(np.min([nb_cut_points,math.ceil((region.size-allocation)*rate)]))
    if nb>0:
        nb_sub_regions = random.choice(range(0,nb))
    else:
        nb_sub_regions = 0
    
    sub_regions= [region]
    if nb_sub_regions > 0:
        sub_regions =  create_sub_region_2(region,nb_sub_regions)
    
    sampled_df = expo_mech_sampling(ep_s,region_df,allocation)
    association_table_data = get_association_table_empty(sub_regions,ep_n,0)
    
    return[sampled_df,association_table_data]

def create_sub_region_2(regions_range,nb_sub):
    sub_regions =[regions_range]

    while(nb_sub > 0):
        random_cuts =[]
        for i_r, region in enumerate(sub_regions):
            cuts =[]
            for cond in region.conditions:
                if cond.end - cond.start > 1:
                    cuts.append([cond.attribute,random.choice(range(cond.start,cond.end))])
            if len(cuts) > 0:
                random_cuts.append([i_r,random.choice(cuts)])
        if len(random_cuts) == 0:
            return sub_regions
        else:
            random_cut = random.choice(random_cuts)
            reg = sub_regions[random_cut[0]]
            del sub_regions[random_cut[0]]
            sub_left,sub_right = split_sub_region(random_cut[1][1],random_cut[1][0],reg)
            
            sub_regions.append(sub_left)
            sub_regions.append(sub_right)
            
            nb_sub = nb_sub - 1
    return sub_regions




def create_sub_regions(region_range,region_data,nb_sub,dims):
    sub_regions =[]
   
    sub_regions.append(region_range)
    while(nb_sub > 0):
        good_range = False
        while(not good_range):
            random_index = random.choice(range(0,len(sub_regions)))
            to_be_split = sub_regions[random_index]
            split_dim = random.choice(dims)
            range_split = [range_s for range_s in to_be_split.conditions if range_s.attribute == split_dim][0]
            good_range = range_split.start != range_split.end
        del sub_regions[random_index]
        split_point = random.choice(range(range_split.start,range_split.end))

        sub_left,sub_right = split_sub_region(split_point,split_dim,to_be_split)
        
        sub_regions.append(sub_left)
        sub_regions.append(sub_right)
        
        nb_sub = nb_sub - 1
    
    return sub_regions


def split_sub_region(split_point,split_dim,sub_region):
    sub_region_left = deepcopy(sub_region)
    sub_region_right = deepcopy(sub_region)
    
    for i in range(len(sub_region.conditions)):
        if(sub_region.conditions[i].attribute == split_dim):
            sub_region_left.conditions[i].end = split_point
            sub_region_right.conditions[i].start = split_point + 1
            break
    return sub_region_left,sub_region_right

def get_region_data(region_range,data):
    boolean_index = (data[region_range.conditions[0].attribute] >= region_range.conditions[0].start) & (data[region_range.conditions[0].attribute] <= region_range.conditions[0].end)
    for cond in region_range.conditions[1:]:
        boolean_index = boolean_index & (data[cond.attribute] >= cond.start) & (data[cond.attribute] <= cond.end )
    region_df = data[boolean_index]

    return region_df

def get_region_vol(region):
    volume = 1
    for cond in region.conditions:
        volume = volume * (cond.end - cond.start + 1)
    return volume


def separate_empty_regions(regions,data):
    empty=[]
    non_empty=[]
    non_empty_data=[]
    for i,df in enumerate(data):
        if df.shape[0] > 0:
            non_empty.append(regions[i])
            non_empty_data.append(df)
        else:
            empty.append(regions[i])
    return non_empty,non_empty_data,empty


def get_allocation_for_regions(regions,ep_lp,ep_n,random_sampling):
            avgs = []
            sums = []
            maxs = [] # sizes
            mins = []
            counts=[]
            for region_df in regions:
                sum_r = region_df['Mesure'].sum() + np.random.laplace(loc=0, scale=1/(ep_lp/2))
                count_r = np.max([2,region_df.shape[0] +np.random.laplace(loc=0, scale=1/(ep_lp/2))]) #+ np.random.laplace(loc=0, scale=1/ep_c)
                mean_r = (sum_r/count_r)  
                avgs.append(mean_r)
                sums.append(sum_r)
                maxs.append(count_r)
                if random_sampling:
                    mins.append(1)
                else:
                    mins.append(1)
                counts.append(count_r)

            algorithm = NSGA2(
            pop_size=40,
            n_offsprings=10,
            sampling=get_sampling("real_random"),
            crossover=get_crossover("real_sbx", prob=0.9, eta=15),
            mutation=get_mutation("real_pm", eta=20),
            eliminate_duplicates=True,
            )
            termination = get_termination("n_gen", 80)


            problem = AllocationLP(maxs,mins,sums,avgs,len(avgs),ep_n)
            res = minimize(problem,
                     algorithm,
                     termination,
                     seed=1,
                     save_history=True,
                     verbose=False)

            X = res.X
            F = res.F

            ### Getting the exp with minimum MSE expected
            mses = [exp[1] for exp in F]

            index = np.argmin(mses)
          

            allo_regions = X[index]
            allo_regions = [int(math.ceil(allo)) for allo in allo_regions]
            
            return allo_regions

class AllocationLP(ElementwiseProblem):
    def __init__(self ,maxs,mins,sums,avgs,num,epsilon):
        super().__init__(n_var=num,
                         n_obj=2,
                         n_constr=0,
                         xl=np.array(mins),
                         xu=np.array(maxs))
        
        self.maxs = maxs
        self.sums = sums
        self.mins = mins
        self.avgs = avgs
        self.num = num
        self.epsilon = epsilon

    def _evaluate(self, x, out, *args, **kwargs):

        f1 = np.sum(self.maxs)+ np.sum(x) 
        f2 = -(np.sum([((x[i]*(self.avgs[i]))) for i in range(len(x))]))
        

        out["F"] = [f1, f2]



    








    