# GCRN
An implementation of GCRN backbone.

#### Requirements
* Python 3.8.8 -> Anaconda Distribution
* pytorch 1.9.1 -> py3.8_cuda11.1_cudnn8.0.5_0
* pandas 1.2.4 
* numpy 1.20.1
* torch-summary 1.4.5 -> pip install torch-summary https://pypi.org/project/torch-summary/ (must necessary)
* jpholiday -> pip install jpholiday (not must, but if you want onehottime)

#### Preparation
* For PEMSBAY dataset, please first upzip ./PEMSBAY/pems-bay.zip to get ./PEMSBAY/pems-bay.h5 file.
* For traintest_GCRN.py, please first run: python generate_training_data.py --dataset=DATA
* GCRN.py only supports adaptive graph, contructed by single embedding vector, defaulted as (N, 8).
** Trainer traintestgts_GCRN.py is inherited from [GTS](https://github.com/chaoshangcs/GTS).
** Trainer traintest+_GCRN.py is inherited from [DL-Traff](https://github.com/deepkashiwa20/DL-Traff-Graph/blob/main/workMETRLA/pred_DCRNN.py).
** Trainer traintest_GCRN.py fix the bug in traintestgts.py as reported [here](https://github.com/deepkashiwa20/MegaCRN/issues/1#issuecomment-1445274957).
* MCRN.py supports multi graphs (adjacency graph and/or adaptive graph).
** Trainer traintest_MCRN.py follows traintest_GCRN.py.

#### Running
* cd model
* python traintest_GCRN.py --dataset=DATA --gpu=GPU_DEVICE_ID 
* DATA = {METRLA, PEMSBAY}
