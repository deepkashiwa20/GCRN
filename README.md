# GCRN
An implementation of GCRN backbone.

#### Requirements
* Python 3.8.8 -> Anaconda Distribution
* pytorch 1.9.1 -> py3.8_cuda11.1_cudnn8.0.5_0
* pandas 1.2.4 
* numpy 1.20.1
* torch-summary 1.4.5 -> pip install torch-summary https://pypi.org/project/torch-summary/ (must necessary)
* jpholiday -> pip install jpholiday (not must, but if you want onehottime)

#### Instructions
* For PEMSBAY dataset, please first upzip ./PEMSBAY/pems-bay.zip to get ./PEMSBAY/pems-bay.h5 file.
* Two trainers, one is traintest_GCRN.py inherited from [GTS](https://github.com/chaoshangcs/GTS), another is traintest+_GCRN.py.
* For traintest_GCRN.py, please first run: python generate_training_data.py --dataset=DATA
* cd model
* python traintest_GCRN.py or traintest+_GCRN.py --dataset=DATA --gpu=GPU_DEVICE_ID 
* DATA = {METRLA, PEMSBAY}
