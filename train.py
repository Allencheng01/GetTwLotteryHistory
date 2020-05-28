import os
import sys
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

import MyParam
from GetLotteryHistory import GetLotteryHistory
from MyModel import MobileNet_Lottery
from MyDataSet import MyLotteryDataSet

def train():
    _args_ = MyParam.ARGS()
    if _args_.ReFetchLog or not os.path.isfile(MyParam.SAVE_LIST_FILENAME):
        LotteryHistoryLists = GetLotteryHistory()
        LotteryHistoryLists = sorted(LotteryHistoryLists, key=lambda k: k['Index'])
        with open(MyParam.SAVE_LIST_FILENAME, 'wb') as wfp:
            pickle.dump(LotteryHistoryLists, wfp)
    else:
        with open(MyParam.SAVE_LIST_FILENAME, 'rb') as rfp:
            LotteryHistoryLists = pickle.load(rfp)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print(device)
    MyDataSet = MyLotteryDataSet(LotteryHistoryLists)
    dataloader = DataLoader(MyDataSet, shuffle=False, num_workers=0, batch_size=2)

    if os.path.isfile(MyParam.CHECKPOINT_FILENAME):
        checkpoint = torch.load(MyParam.CHECKPOINT_FILENAME)
        net = checkpoint['net']
        _optimizer_ = checkpoint['optimizer']
        _epoch_ = checkpoint['epoch']
    else:
        net = MobileNet_Lottery(MyParam.LOTTERY_NUM, MyParam.EMBEDDED_CH, num_classes=49)
        _optimizer_ = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
        _epoch_ = 0

    # _optimizer_ = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    net = net.to(device)
    _loss_fun_ = nn.CrossEntropyLoss()

    # scheduler = lr_scheduler.ReduceLROnPlateau(_optimizer_)
    scheduler = lr_scheduler.StepLR(_optimizer_, step_size=50)

    _min_total_loss_ = sys.maxsize
    while True:
        if _epoch_ > 200:
            break
        _epoch_ += 1
        print('_epoch_ :', _epoch_)
        for param_group in _optimizer_.param_groups:
            print('learning rate = ', param_group['lr'])
        _total_loss_ = 0
        for batch_idx, (Target, DecodeMap) in enumerate(dataloader):
            Target = Target.to(device)
            DecodeMap = DecodeMap.to(device)

            net.zero_grad()
            Predict_Target = net(DecodeMap)
            _loss_ = _loss_fun_(Predict_Target, Target)
            _loss_.backward()
            _optimizer_.step()
            _total_loss_ += _loss_

        print(_total_loss_.item())
        # scheduler.step(_total_loss_.item())
        scheduler.step()

        with open('train_log.txt', 'a') as wfd:
            wfd.write('[{0}] _epoch_ = {1} _total_loss_ = {2}\n'.format(MyParam.TRAIN_NUM_INDEX, _epoch_, _total_loss_))

        if _total_loss_ < _min_total_loss_:
            _min_total_loss_ = _total_loss_
            state = {'epoch': _epoch_,
                     'net': net,
                     'optimizer': _optimizer_,
                    'total_loss': _total_loss_}
            torch.save(state, MyParam.CHECKPOINT_FILENAME)

def main():
    with open('train_log.txt', 'w') as wfd:
        wfd.write('')
    for i in range(0, 7):
        MyParam.TRAIN_NUM_INDEX = i
        MyParam.CHECKPOINT_FILENAME = 'checkpoint{0}.ckpt'.format(MyParam.TRAIN_NUM_INDEX)
        train()

if __name__ == "__main__":
    sys.exit(main())