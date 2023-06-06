# SLIM-View (SampLing dIfferentially Materialized View)

  

This code is associated to the work we submitted to **EDBT 2024**

To run the experiments you need to :
- Install all the rependencies 
> virtualenv venv
> source venv/bin/activate
> pip3 install -r requirements.txt

- Download the datasets
> ./download_dataset.sh
- Preprocess the data
> python3 src/script/preprocess.py
- Run experiments :
	- SLIM-View
	 > python3 src/script/count_query_eval_SLIM-View.py
	- Partitioning HDPView and PrivTree
		> python3 src/script/count_query_eval_HDP-PrivTree.py
	- Workload-dependent approaches and Identity
		> python3 src/script/count_query_eval_HDP-PrivTree.py
	- Generative-model PrivBayes
		> python3 src/script/count_query_eval_PrivBayes.py