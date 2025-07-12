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

# Note
The loss function in the original paper is highly sensitive to the hyperparameters, so it requires careful adjustment to achieve good results. This might be because the accuracy of the estimation of the value function significantly affects the model performance.

In addition, we have provided a more stable loss function without learnable value functions. It may not perform as well as the original loss function with careful adjustment, but it is more stable and only requires a slight adjustment of the hyperparameters to achieve a comparable effect as described in the paper.

You can adjust the loss in code/bidding_train_env/baseline/dt/dt.py

```
# Loss function without learnable value function. It's more stable but may perform worse than a finetuned loss with learnable value function.
# In this case, we simply boost exploration by maxmaizing curr_score_preds_1 in wo.
# wo = torch.sigmoid(1 * (curr_score_preds_1-curr_score_preds.clone().detach()))
# wo_frozen = wo.clone().detach()
# loss1 = torch.mean((1-wo_frozen)*((action_preds - action_target) ** 2) + wo_frozen*((action_preds - action_1_frozen) ** 2))
# loss2 = torch.mean((curr_score_preds - curr_score_target) ** 2)*200
# loss3 = torch.mean(1-wo)*100
# loss = loss1+loss2+loss3

# The loss in the paper. It's param sensitive and need careful param selection.
wo = torch.sigmoid(100 * (curr_score_preds_1 - curr_score_preds))
wo_frozen = wo.clone().detach()
diff = curr_score_target - value_preds
weight = torch.where(diff > 0, self.expectile, (1 - self.expectile))
loss1 = torch.mean((1 - wo_frozen) * ((action_preds - action_target) ** 2) +
                  wo_frozen * ((action_preds - action_1_frozen) ** 2))
loss2 = torch.mean((curr_score_preds - curr_score_target) ** 2) * 200
loss3 = torch.mean(weight * (diff ** 2)) * 100
loss4 = torch.mean((curr_score_preds_1 - value_preds_frozen) ** 2) * 100
loss = loss1 + loss2 + loss3 + loss4
```

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
