from abc import ABC, abstractmethod


class BaseBiddingStrategy(ABC):

    def __init__(self, budget=100, name="BaseStrategy", cpa=2, category=1):
        self.budget = budget
        self.remaining_budget = budget
        self.name = name
        self.cpa = cpa
        self.category = category

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def bidding(self, timeStepIndex, pValues, pValueSigmas, historyPValueInfo, historyBid,
                historyAuctionResult, historyImpressionResult, historyLeastWinningCost):
        pass
