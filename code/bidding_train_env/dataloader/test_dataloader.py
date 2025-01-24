import os
import numpy as np
import pandas as pd
import pickle
import warnings

warnings.filterwarnings('ignore')


class TestDataLoader:
    def __init__(self, file_path="./data/log.csv"):
        self.file_path = file_path
        self.raw_data_path = os.path.join(os.path.dirname(file_path), "raw_data.pickle")
        self.raw_data = self._get_raw_data()
        self.keys, self.test_dict = self._get_test_data_dict()

    def _get_raw_data(self):
        if os.path.exists(self.raw_data_path):
            with open(self.raw_data_path, 'rb') as file:
                return pickle.load(file)
        else:
            tem = pd.read_csv(self.file_path)
            with open(self.raw_data_path, 'wb') as file:
                pickle.dump(tem, file)
            return tem

    def _get_test_data_dict(self):
        grouped_data = self.raw_data.sort_values('timeStepIndex').groupby(['deliveryPeriodIndex', 'advertiserNumber'])
        data_dict = {key: group for key, group in grouped_data}
        return list(data_dict.keys()), data_dict

    def mock_data(self, key):
        data = self.test_dict[key]
        pValues = data.groupby('timeStepIndex')['pValue'].apply(list).apply(np.array).tolist()
        pValueSigmas = data.groupby('timeStepIndex')['pValueSigma'].apply(list).apply(np.array).tolist()
        leastWinningCosts = data.groupby('timeStepIndex')['leastWinningCost'].apply(list).apply(np.array).tolist()
        num_timeStepIndex = len(pValues)
        budget = data['budget'].iloc[0]
        cpa = data['CPAConstraint'].iloc[0]
        category = data['advertiserCategoryIndex'].iloc[0]
        return num_timeStepIndex, pValues, pValueSigmas, leastWinningCosts, budget, cpa, category


if __name__ == '__main__':
    pass
