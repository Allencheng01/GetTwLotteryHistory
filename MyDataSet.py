import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset

import MyParam

class MyLotteryDataSet(Dataset):
    def __init__(self, LotteryHistoryLists):
        super().__init__()
        self.LotteryHistoryLists = LotteryHistoryLists
    def __len__(self):
        return len(self.LotteryHistoryLists) - MyParam.LOTTERY_HEIGHT
    def __getitem__(self, Index):
        TargetIndex = Index + MyParam.LOTTERY_HEIGHT

        # Prepare train data
        MyDecodeMap = np.zeros((MyParam.LOTTERY_HEIGHT, MyParam.LOTTERY_NUM), dtype=np.long)
        # MyDecodeMap = np.zeros((MyParam.LOTTERY_HEIGHT, MyParam.LOTTERY_HEIGHT), dtype=np.long)
        for row in range(Index, Index + MyParam.LOTTERY_HEIGHT):
            for col in range(MyParam.LOTTERY_NUM):
                MyDecodeMap[row - Index][col] = self.LotteryHistoryLists[row]['Numbers'][col]

        return torch.tensor(self.LotteryHistoryLists[TargetIndex]['Numbers'][MyParam.TRAIN_NUM_INDEX], dtype=torch.long), torch.tensor(MyDecodeMap, dtype=torch.long)
