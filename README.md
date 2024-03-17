# FedHydro

## 00. Data preparation
### Download the dataset
- Download the CAMELS dataset from https://ral.ucar.edu/solutions/products/camels
- Place related files into the directories ./dataset/series_data/discharge_data and ./dataset/series_data/forcing_data
### Merge the dataset
- Execute ./dataset/series_data/utils/generate_data.py to generate the merged dataset
- Place the merged dataset into ./dataset/series_data/
- Create a class for loading the merged dataset in ./dataset/

## 01. Execute
- Set up a distributed environment(docker recommended)
- Execute ./worker.py on each worker node
- Run the corresponding submitter program *_submit_*.py
- Extract the trained model "*.model" from the result file named "Node-7-Retrieve"
  *specific instructions can be found at https://github.com/EngineerDDP/Parallel-SGD*

## 02. Test
- Generate controlled experiment models using the code provided in ./dataset/code_test/fedsrp/chapter4_exp2.py and ./dataset/code_test/fedsrp/FedAvg
- Test related parameters using the code provided in ./dataset/code_test/fedsrp/chapter4_exp2_updated.py
