# Note ! ! !

- A rough implementation code for GAVE.

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
@inproceedings{gao2025samplellm,
  title={SampleLLM: Optimizing Tabular Data Synthesis in Recommendations},
  author={Gao, Jingtong and Du, Zhaocheng and Li, Xiaopeng and Wang, Yichao and Li, Xiangyang and Guo, Huifeng and Tang, Ruiming and Zhao, Xiangyu},
  booktitle={Companion Proceedings of the ACM on Web Conference 2025},
  pages={211--220},
  year={2025}
}
```
