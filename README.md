# Efficient and Light-Weight Federated Learning via Asynchronous Distributed Dropout

This is the official code base for "Efficient and Light-Weight Federated Learning via Asynchronous Distributed Dropout". 

The experiments are performed on 4 P100 GPUs. (Note due to the indeterministic nature of asynchronous algorithms and HodWild! style lock-free training method, it is required to run the exact hardware setting to replicate the reported results.)

The command line for running Hetero AsyncDrop for CIFAR100 dataset is: 

python3 ./asyncdrop_main.py

The command line for running Async FedAvg baseline for CIFAR100 dataset is: 

python3 ./asyncdrop_main.py --baseline
