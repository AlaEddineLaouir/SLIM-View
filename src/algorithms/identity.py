from algorithms.noised_data import NoisedData
import numpy as np
from functools import reduce
import pandas as pd
import math

def Identity(data, prng, epsilon, workload_optimized=False, workloads=[]):
    if workload_optimized:
        projected_data = {}
        datavector = data.datavector(flatten=False) 
        workload_num = len(workloads)
        for proj, _ in workloads:
            aggregate_cols = filter(lambda x: x not in proj, data.domain.attrs)
            indices = tuple([data.domain.attrs.index(col) for col in aggregate_cols])
            x = datavector.sum(axis=indices)
            y = x + prng.laplace(0.0, 1.0*workload_num/epsilon, data.project(proj).domain.shape)
            projected_data[proj] = y
        return NoisedData(data.domain.attrs, projected_data, workload_optimized)
    else:
        rmse_s = []
        for workload in workloads:
            dataset = data.df.copy()
            dataset_domain = data.domain.shape
            #workload  = workload[0]
            dims = get_attribut_qwl(workload)
            columns_to_drop = [index for index,value in enumerate(dataset.columns) if value not in dims]
            columns_to_not_drop = [index for index,value in enumerate(dataset.columns) if value in dims]

            global_dataset = dataset.copy()
            global_dataset['Mesure'] = 1
            global_dataset = global_dataset.drop(global_dataset.columns[columns_to_drop], axis=1) 
            global_dataset = global_dataset.groupby(dims).sum().reset_index()
            rmse = 0.0
            for i in range(10):
                histo, shape = create_dd_histo_dawa(dataset,data.domain.shape,columns_to_drop,columns_to_not_drop)
                domainS = reduce(lambda x, y: x*y, shape)
                res = histo + prng.laplace(0.0, 1.0/epsilon, domainS)
                
                perturbed_df = pd.DataFrame()
                domain_size = reduce(lambda x, y: x*y, shape)
                for i in range(len(dims)):
                    dim_members = []
                    if i+ 1 == len(dims):
                        dim_reps = int(domain_size)
                        dim_members =  list(range(shape[i])) * dim_reps
                    else:
                        dim_reps = mul_array([shape[k] for k in range(i+1,len(shape))])
                        for member in range(shape[i]):
                            dim_members = dim_members + [member] * dim_reps
                        
                    perturbed_df[dims[i]] = dim_members
                    domain_size = domain_size / dim_reps
                    
                perturbed_df['Mesure'] = res
                
                # (8) Evaluating the algorithm (Using RMSE mesure)
                est =[]
                true =[]
                for index, query in enumerate(workload):
                    res_q = run_query_on_df(global_dataset,query)
                    true.append(res_q)
                    #query_exe_times_true[index] += time_exe/10
                    res_q = run_query_on_df(perturbed_df,query)
                    est.append(res_q)
                    #query_exe_times_est[index] += time_exe/10
                    
                
                
                est = np.array(est)
                true = np.array(true)
                    #re += np.mean(np.abs(est - true)/true)/10
                rmse_1 = math.sqrt(np.mean(np.square(est - true)))
                rmse += rmse_1/10
            rmse_s.append(rmse)
        
        res = np.mean(rmse_s)
        print("")
            


        #return NoisedData(data.domain.attrs, perturbed_df, workload_optimized)

def get_attribut_qwl(workload):
    dims = []
    for cond in workload[0].conditions:
        dims.append(cond.attribute)
    print(dims)
    return dims

def run_query_on_df(df,query):
    #start_time = time.time()
    boolean_index_org = (df[query.conditions[0].attribute] >= query.conditions[0].start) & (df[query.conditions[0].attribute] <= query.conditions[0].end)
    for cond in query.conditions[1:]:
        boolean_index_org = boolean_index_org & (df[cond.attribute] >= cond.start) & (df[cond.attribute] <= cond.end )           
    res = df[boolean_index_org]['Mesure'].sum()
    #end_time = time.time()
    return res#, end_time-start_time

def create_dd_histo_dawa(dataset,dataset_shape,columns_to_drop,columns_not_drop, flatten=True):
        """ return the database in vector-of-counts form """
        shape =[]
        bins = []
        for i in columns_not_drop:
            bins.append(range(dataset_shape[i]+1))
            shape.append(dataset_shape[i])

        dataset_c = dataset.drop(dataset.columns[columns_to_drop], axis=1) 
        ans = np.histogramdd(dataset_c.values, bins)[0]
        return (ans.flatten(),shape )if flatten else (ans,shape)
def mul_array(ar):
    mul = 1
    for e in ar:
        mul = mul*e
    return mul
