# Introduction

- This is the code reference for the main process of **GAVE** accepted by SIGIR'25.

- The code provided is for reference only. 

- Because the dataset is too large to upload, you need to add the dataset before it can run

- GAVE sturcture: code/bidding_train_env/baseline/dt/dt.py -> class GAVE

**Implementation Details**:  

- **Codebase**: Lightweight reference implementation using a variant of the [AuctionNet](https://github.com/alimama-tech/AuctionNet) benchmark, i.e.,  [AIGB](https://github.com/alimama-tech/NeurIPS_Auto_Bidding_AIGB_Track_Baseline)
- **Data**: A **sample dataset** for training is available at `Data/trajectory/trajectory_data.csv`. Test data was **omitted** due to size constraints.  

**Execution**:  

1. Add full datasets with structure:  
   ```plaintext
   GAVE folder/
   --data/
   ----traffic/
   ------period-x.csv  # Test data
   ----trajectory/
   ------trajectory_data.csv # Train data

- Run:  

  ```python code/main/main_train_test.py```

# Citation
If you feel our work is insightful and want to use the code or cite our paper, please add the following citation to your paper references.

```
@article{gao2025generative,
  title={Generative Auto-Bidding with Value-Guided Explorations},
  author={Gao, Jingtong and Li, Yewen and Mao, Shuai and Jiang, Peng and Jiang, Nan and Wang, Yejing and Cai, Qingpeng and Pan, Fei and Gai, Kun and An, Bo and others},
  journal={arXiv preprint arXiv:2504.14587},
  year={2025}
}
```
