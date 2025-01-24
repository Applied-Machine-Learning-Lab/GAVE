**Note ! ! !**

- A rough implementation code for GAVE.

- The code provided is for reference only. 

- Because the dataset is too large to upload, you need to add the dataset before it can run

- GAVE sturcture: code/bidding_train_env/baseline/dt/dt.py

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