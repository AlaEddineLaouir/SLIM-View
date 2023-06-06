#!/bin/bash
mkdir -p ./data/raw ./data/preprocessed ./data/synthetic

# bitcoin
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00526/data.zip -P ./data/raw/bitcoin
unzip ./data/raw/bitcoin/data.zip -d ./data/raw/bitcoin/

# adult
wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data -P ./data/raw/adult/
wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test -P ./data/raw/adult/
sed -i -e '1d' ./data/raw/adult/adult.test

# electricity
wget https://www.openml.org/data/get_csv/2419/electricity-normalized.csv -P ./data/raw/electricity/


# trafic
wget http://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz -P ./data/raw/trafic
gzip -d ./data/raw/trafic/Metro_Interstate_Traffic_Volume.csv.gz 


