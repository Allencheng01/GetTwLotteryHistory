import os
import sys
import pickle
import numpy as np
import torch
from MyModel import MobileNet_Lottery

from GetLotteryHistory import GetLotteryHistory
import MyParam

def predict():
    _args_ = MyParam.ARGS()
    if _args_.ReFetchLog or not os.path.isfile(MyParam.SAVE_LIST_FILENAME):
        LotteryHistoryLists = GetLotteryHistory()
        LotteryHistoryLists = sorted(LotteryHistoryLists, key=lambda k: k['Index'])
        with open(MyParam.SAVE_LIST_FILENAME, 'wb') as wfp:
            pickle.dump(LotteryHistoryLists, wfp)
    else:
        with open(MyParam.SAVE_LIST_FILENAME, 'rb') as rfp:
            LotteryHistoryLists = pickle.load(rfp)

    TargetIndex = len(LotteryHistoryLists)
    Index = TargetIndex - MyParam.LOTTERY_HEIGHT
    MyDecodeMap = np.zeros((1, MyParam.LOTTERY_HEIGHT, MyParam.LOTTERY_NUM), dtype=np.long)
    for row in range(Index, Index + MyParam.LOTTERY_HEIGHT):
        for col in range(MyParam.LOTTERY_NUM):
            MyDecodeMap[0][row - Index][col] = LotteryHistoryLists[row]['Numbers'][col]

    # print('Load', MyParam.CHECKPOINT_FILENAME)
    checkpoint = torch.load(MyParam.CHECKPOINT_FILENAME)
    net = checkpoint['net'].to('cuda')

    net.eval()
    with torch.no_grad():
        MyDecodeMap = torch.tensor(MyDecodeMap, dtype=torch.long).to('cuda')
        Predict_Class = net(MyDecodeMap)

        # print(torch.argmax(Predict_Class).item() + 1)
        values, indices = torch.max(Predict_Class, 1)
        print('[{0}]: {1}'.format(indices.item() + 1, values.item()))
        # print(values)
        # print(indices)
    

def main():
    for i in range(0, 7):
        MyParam.TRAIN_NUM_INDEX = i
        MyParam.CHECKPOINT_FILENAME = 'checkpoint{0}.ckpt'.format(MyParam.TRAIN_NUM_INDEX)
        predict()

if __name__ == "__main__":
    sys.exit(main())