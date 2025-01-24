import time
import gin
import numpy as np
import os
import psutil
# from saved_model.DTtest.dt import DecisionTransformer
from bidding_train_env.baseline.dt.dt import GAVE
from bidding_train_env.strategy.base_bidding_strategy import BaseBiddingStrategy
import torch
import pickle


class DtBiddingStrategy(BaseBiddingStrategy):

    def __init__(self, budget=100, name="Decision-Transformer-PlayerStrategy", cpa=2, category=1,
                 model_name="160000.pt", model_param = {
    "step_num": 300000,
    "save_step": 20000,
    "dir": "../data/trajectory/trajectory_data_all.csv",
    "hidden_size": 512,
    "learning_rate": 0.0001,
    "time_dim": 8,
    "batch_size": 128,
    "device": "cpu",
    "block_config": {
        "n_ctx": 1024,
        "n_embd": 512,
        "n_layer": 8,
        "n_head": 16,
        "n_inner": 1024,
        "activation_function": "relu",
        "n_position": 1024,
        "resid_pdrop": 0.1,
        "attn_pdrop": 0.1,
    }
}):
        super().__init__(budget, name, cpa, category)

        file_name = os.path.dirname(os.path.realpath(__file__))
        dir_name = os.path.dirname(file_name)
        dir_name = os.path.dirname(dir_name)
        model_path = os.path.join(model_param["save_dir"], model_name)
        picklePath = os.path.join(model_param["save_dir"], "normalize_dict.pkl")

        with open(picklePath, 'rb') as f:
            normalize_dict = pickle.load(f)
        self.model = GAVE(state_dim=16, act_dim=1,
                                hidden_size=model_param['hidden_size'], state_mean=normalize_dict["state_mean"],
                                state_std=normalize_dict["state_std"], device=model_param['device'],
                                learning_rate=model_param["learning_rate"], time_dim=model_param['time_dim'],
                                block_config=model_param['block_config'], expectile=model_param['expectile']
                                )
        self.model.load_net(model_path)

    def reset(self):
        self.remaining_budget = self.budget

    def bidding(self, timeStepIndex, pValues, pValueSigmas, historyPValueInfo, historyBid,
                historyAuctionResult, historyImpressionResult, historyLeastWinningCost):
        time_left = (48 - timeStepIndex) / 48
        budget_left = self.remaining_budget / self.budget if self.budget > 0 else 0
        history_xi = [result[:, 0] for result in historyAuctionResult]
        history_pValue = [result[:, 0] for result in historyPValueInfo]
        history_conversion = [result[:, 1] for result in historyImpressionResult]

        historical_xi_mean = np.mean([np.mean(xi) for xi in history_xi]) if history_xi else 0

        historical_conversion_mean = np.mean(
            [np.mean(reward) for reward in history_conversion]) if history_conversion else 0

        historical_LeastWinningCost_mean = np.mean(
            [np.mean(price) for price in historyLeastWinningCost]) if historyLeastWinningCost else 0

        historical_pValues_mean = np.mean([np.mean(value) for value in history_pValue]) if history_pValue else 0

        historical_bid_mean = np.mean([np.mean(bid) for bid in historyBid]) if historyBid else 0

        def mean_of_last_n_elements(history, n):
            l = len(history)
            last_n_data = history[max(0, l - n):l]
            if len(last_n_data) == 0:
                return 0
            else:
                return np.mean([np.mean(data) for data in last_n_data])

        last_three_xi_mean = mean_of_last_n_elements(history_xi, 3)
        last_three_conversion_mean = mean_of_last_n_elements(history_conversion, 3)
        last_three_LeastWinningCost_mean = mean_of_last_n_elements(historyLeastWinningCost, 3)
        last_three_pValues_mean = mean_of_last_n_elements(history_pValue, 3)
        last_three_bid_mean = mean_of_last_n_elements(historyBid, 3)

        current_pValues_mean = np.mean(pValues)
        current_pv_num = len(pValues)

        historical_pv_num_total = sum(len(bids) for bids in historyBid) if historyBid else 0
        last_three_ticks = slice(max(0, timeStepIndex - 3), timeStepIndex)
        last_three_pv_num_total = sum(
            [len(historyBid[i]) for i in range(max(0, timeStepIndex - 3), timeStepIndex)]) if historyBid else 0

        test_state = np.array([
            time_left, budget_left, historical_bid_mean, last_three_bid_mean,
            historical_LeastWinningCost_mean, historical_pValues_mean, historical_conversion_mean,
            historical_xi_mean, last_three_LeastWinningCost_mean, last_three_pValues_mean,
            last_three_conversion_mean, last_three_xi_mean,
            current_pValues_mean, current_pv_num, last_three_pv_num_total,
            historical_pv_num_total
        ])

        if timeStepIndex == 0:
            self.model.init_eval()

        alpha = self.model.take_actions(test_state, budget=self.budget, cpa=self.cpa,
                                        pre_reward=sum(history_conversion[-1]) if len(history_conversion) != 0 else None)
        bids = alpha * pValues
        return bids


