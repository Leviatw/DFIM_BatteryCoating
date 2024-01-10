# DFIM_BatteryCoating
The code to regenerate the data and the results in
T. Gong, D. Liu, H. Kim, S.-H. Kim, T. Kim, D. Lee, Y. Xie. "Distribution-free Image Monitoring with Application to Battery Coating Process". IISE Transactions, 2024.

## Data
`DFIM_BatteryCoating.ipynb` provides data generation for Table 1, 2, 3, and S.2, covering both Type 1 and Type 2 data under both in-control and out-of-control conditions. Please note that users need to create the local storage paths `"data/in-control/"` and `"data/out-of-control/"` due to GitHub's storage limitations. It is advised not to store the data in full scale locally due to its substantial memory requirements.

**Note:** Users are responsible for creating the specified local storage paths.

## Implementation
`DFIM_BatteryCoating.ipynb` includes the implementation of the proposed method, enabling the regeneration of results in Table 1, 2, 3, and S.2 for DFIM. To achieve stable statistical performance, ensure that the arguments, such as sequence length and the number of sequences, match those in the paper.

Given the significant memory cost of the data, it is recommended that users customize the code to generate data in an online manner for monitoring, especially for high-dimensional Type 2 data. Additionally, consider monitoring sequences in parallel for efficiency.

