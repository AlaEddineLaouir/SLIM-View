#include <iostream>
#include <fstream>
#include <tgmath.h>
#include <cmath>
#include <cstdio>
#include <sys/stat.h>
#include <sys/types.h>

using namespace std;

#include "methods.h"

double exp_marginal(table& tbl, base& base, int k) {
	const vector<vector<int>> mrgs = tools::kway(k, tools::vectorize(tbl.dim));
	double err = 0.0;
	for (const vector<int>& mrg : mrgs) err += tools::TVD(tbl.getCounts(mrg), base.getCounts(mrg));
	return err / mrgs.size();
}


// bin adult ml 10 0.25 0.5 1.0 2.0 4.0

int main(int argc, char *argv[]) {
		// arguments
		string dataset = "gendim";//argv[1];
		
		// mode: "ml" or "cq"
		string mode = "cq";// argv[2];

		bool is_train;
		if (mode == "ml") {
			is_train = true;
		} else if (mode =="cq"){
			is_train = false;
		} else {
			cout << "Error: mode is wrong " << mode << endl;
			exit(1);
		}

		string data_dir = "synthetic_data/"+mode+"/"+dataset;
		mkdir("synthetic_data", 0755);
		mkdir(("synthetic_data/"+mode).c_str(), 0755);
		mkdir(data_dir.c_str(), 0755);

		// Just to set the output path
		data_dir = "/Users/alaeddinelaouir/PhDProjects/HDPView/data/preprocessed/gendim/";

		int rep = 1; // stoi(argv[3]);
		
		vector<double> epsilons;
		epsilons.push_back(0.1);
		// for (int i = 4; i < argc; i++) {
		// 	epsilons.push_back(stod(argv[i]));
		// }
		random_device rd;
		engine eng(rd());
		
		/* in case of jupyter-base docker image */
		table tbl("/Users/alaeddinelaouir/PhDProjects/HDPView/data/preprocessed/" + dataset + "/", is_train, true);
		for (double theta : {4.0}) {
		cout << "theta: " << theta << endl;
		for (double epsilon : epsilons) {
			for (int i = 0; i < rep; i++) {
				cout << "epsilon: " << epsilon << " rep: " << i << endl;
				bayesian bayesian(eng, tbl, epsilon, theta);
				
				ofstream sample(data_dir+"raw_privbayes.csv");
				table syn = bayesian.syn;
				for (int i = 0; i < syn.data.size();i ++){
					for (int j = 0; j < syn.data[i].size();j ++){
						int d = syn.data[i][j];
						sample << syn.translators[j]->int2str(d);
						if (j != syn.data[i].size()-1)
							sample << ",";
					}
					sample << endl;
				}			
				sample.close();
			}
		}
	epsilons.pop_back();
	}
	cout << "done." << endl;
	return 0;
}
