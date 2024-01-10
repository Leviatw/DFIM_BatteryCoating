# DFIM_BatteryCoating
The code to regenerate the data and the results in
T. Gong, D. Liu, H. Kim, S.-H. Kim, T. Kim, D. Lee, Y. Xie. "Distribution-free Image Monitoring with Application to Battery Coating Process". IISE Transactions, 2024.

## Data
DFIM_BatteryCoating.ipynb provides data generation for Table 1, 2, 3 and S.2, which include both Type 1 and Type 2 data in both in-control and out-of-control conditions. Note that the local storage paths "data/in-control/" and "data/out-of-control/" should be created by users due to the storage limit in Github. And it is not recommended to generate and store locally the data in full scale due to their huge storage size. 

## Implementation
DFIM_BatteryCoating.ipynb provides the implementation of the proposed method, which can regenerate the results in Table 1, 2, 3 and S.2 for DFIM. Currently, However, DFIM_BatteryCoating.ipynb provides a demo. The arguments including the length of the sequences and the number of sequences should be made full scale as in the paper to achieve stable statistical performance. The memory cost of the data is considerable, thus it is recommended that users customize the code to generate the data for monitoring in an online manner and monitor the sequences parallelly, especially for high-dimensional Type 2 data.
