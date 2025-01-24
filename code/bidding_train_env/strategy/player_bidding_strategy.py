import time
import numpy as np
import os
import psutil

from bidding_train_env.strategy.base_bidding_strategy import BaseBiddingStrategy


class PlayerBiddingStrategy(BaseBiddingStrategy):
    def __init__(self, budget=100, name="PlayerStrategy", cpa=40, category=1):
        super().__init__(budget, name, cpa, category)

    def reset(self):
        self.remaining_budget = self.budget

    def bidding(self, timeStepIndex, pValues, pValueSigmas, historyPValueInfo, historyBid,
                historyAuctionResult, historyImpressionResult, historyLeastWinningCost):

        return self.cpa * pValues
