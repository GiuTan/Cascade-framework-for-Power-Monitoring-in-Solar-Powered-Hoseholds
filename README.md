# A Deep Cascade Framework for Non-Intrusive Power Disaggregation in Solar-Powered Households
This repository contains Python code to implement the approach proposed in "A Deep Cascade Framework for Non-Intrusive Power Disaggregation in Solar-Powered Households" by Tanoni et al. submitted to ISCAS 25. The repository contains codes to reproduce the cascade approach as well as five NILM benchmark architectures, implemented in Pytorch. 

Each folder refers to a different experiment:
- 'models' refers to experiments on solar-powered households data with one architecture for all the appliances and solar power 
- 'models_cascade' refers to experiments on solar-powered households data within the cascade framework for the CRNN (proposed method)
- 'models_NILM' refers to experiments on non solar-powered households data with one architecture for all the appliances

An external folder called 'datasets' should be created to contain files .csv with signals from the various houses of UK-DALE and REFIT. The files should be internally organized as follows:
columns= 'Time','active', 'solar', 'kettle', 'microwave','fridge','washing machine', 'dishwasher','net' 
- 'net' column refers to the active power minus solar power production
- 'active' is the aggregate active power consumption of the building
- 'Time' is the timestamps columns

The file run.sh allows to run 6 experiments changing the seed and the results.py script will compute the average metrics among the 6 trials.
Data are available under request. 


