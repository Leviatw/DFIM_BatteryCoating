# DFIM_BatteryCoating
The project provides the codes to regenerate the simulated data and the simulation results in
T. Gong, D. Liu, H. Kim, S.-H. Kim, T. Kim, D. Lee, Y. Xie. "Distribution-free Image Monitoring with Application to Battery Coating Process". IISE Transactions, 2024.

## Data
`DFIM_BatteryCoating.ipynb` provides data generation for Table 1, 2, 3, and S.2, covering both Type 1 and Type 2 data under both in-control and out-of-control conditions. Please note that users need to create the local storage paths `"data/in-control/"` and `"data/out-of-control/"` due to GitHub's storage limitations. It is advised not to store the data in full scale locally due to its substantial memory requirements.

**Note:** Users are responsible for creating the specified local storage paths.

## DFIM
`DFIM_BatteryCoating.ipynb` includes the implementation of the proposed method. 

## MEWMA
`MEWMA.ipynb` includes the implementation of the baseline MEWMA proposed by Wang and Lai (2019). 

## MGLR
`DFIM_BatteryCoating.ipynb` includes the implementation of the baseline MGLR proposed by He et al. (2016). For more details, one can also refer to GitHub repository at https://github.com/fmegahed/image-mglr. 

## ST-SSD
The folder `"ST-SSD"` includes the implementation of the baseline ST-SSD proposed by Yan et al. (2018). For more details, one can also refer to GitHub repository at https://github.com/hyan46/STSSD. The results are produced in GaTech ISyE HTCondor Cluster. The software environment is Python 3.8.13 running in a virtual environment on Linux servers. For the background on HTCondor Cluster, one can refer to  https://github.com/BillHuang01/condor_tutorial. 

**Note 1:**
To achieve stable statistical performance, ensure that the arguments, such as the training sequence length and the number of sequences for monitoring, match those in the paper.

**Note 2:**
Since simulations contain a group of settings, the code specify the setting by a series of arguments, including `ControlType`, `DataType`, `DistType`, `phi`, `rho`, `CovType`, `DeltaNorm` and `ShiftType`.

**Note 3:**
Given the significant memory cost of the data, it is recommended that users customize the code to generate data in an online manner for monitoring, especially for high-dimensional Type 2 data. Additionally, consider monitoring sequences in parallel for efficiency. 

