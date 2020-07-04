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
from GetLotteryHistory_2 import GetLotteryHistory_2
# import MyModel
from MyShuffleNet import My_shufflenet_v2_x2_0
import torchvision.models as models
from MyDataSet import MyLotteryDataSet

import time
import datetime

def train():
    _args_ = MyParam.ARGS()
    if _args_.ReFetchLog or not os.path.isfile(MyParam.SAVE_LIST_FILENAME_2):
        LotteryHistoryLists = GetLotteryHistory_2()
        LotteryHistoryLists = sorted(LotteryHistoryLists, key=lambda k: k['Index'])
        with open(MyParam.SAVE_LIST_FILENAME_2, 'wb') as wfp:
            pickle.dump(LotteryHistoryLists, wfp)
    else:
        with open(MyParam.SAVE_LIST_FILENAME_2, 'rb') as rfp:
            LotteryHistoryLists = pickle.load(rfp)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print(device)
    MyDataSet = MyLotteryDataSet(LotteryHistoryLists)
    dataloader = DataLoader(MyDataSet, shuffle=True, num_workers=8, batch_size=8)

    if os.path.isfile(MyParam.CHECKPOINT_FILENAME2):
        checkpoint = torch.load(MyParam.CHECKPOINT_FILENAME2)
        net = checkpoint['net']
        _optimizer_ = checkpoint['optimizer']
        _epoch_ = checkpoint['epoch']
        for param_group in _optimizer_.param_groups:
            _min_total_loss_ = param_group['lr']
    else:
        net = My_shufflenet_v2_x2_0(VocabSize=MyParam.LOTTERY_2_MAX_NUM+1, EmbeddingSize=MyParam.EMBEDDED_CH, num_classes=MyParam.LOTTERY_2_MAX_NUM+1)
        # _optimizer_ = optim.Adam(net.parameters())
        _optimizer_ = optim.SGD(net.parameters(), lr=1e-3,momentum=0.8, weight_decay=1e-4)
        _epoch_ = 0
        _min_total_loss_ = sys.maxsize

    net = net.to(device)
    _loss_fun_ = nn.CrossEntropyLoss()

    # scheduler = lr_scheduler.ReduceLROnPlateau(_optimizer_, factor=0.8)
    scheduler = lr_scheduler.StepLR(_optimizer_, step_size=50, gamma=0.5)

    # for param_group in _optimizer_.param_groups:
    #     param_group['lr'] = 3e-4
    net.train()
    while True:
        if _epoch_ > 200:
            break
        _epoch_ += 1
        print('_epoch_ :', _epoch_)
        for param_group in _optimizer_.param_groups:
            print('learning rate = ', param_group['lr'])
        _total_loss_ = 0
        _loss_list_ = list()
        for batch_idx, (Target, DecodeMap) in enumerate(dataloader):
            Target = Target.to(device)
            DecodeMap = DecodeMap.to(device)

            net.zero_grad()
            Predict_Target = net(DecodeMap)
            _loss_ = _loss_fun_(Predict_Target, Target)
            _loss_.backward()
            _optimizer_.step()
            _total_loss_ += abs(_loss_.item())
            _loss_list_.append(abs(_loss_.item()))
            # print(batch_idx, _loss_.item())

        print('mean:', np.mean(_loss_list_))
        print(_total_loss_)
        # scheduler.step(_total_loss_)
        scheduler.step()

        with open('train_log_2.txt', 'a') as wfd:
            wfd.write('[{0}] _epoch_ = {1} _total_loss_ = {2} mean = {3}\n'.format(MyParam.TRAIN_NUM_INDEX, _epoch_, _total_loss_, np.mean(_loss_list_)))

        if _total_loss_ < _min_total_loss_:
            _min_total_loss_ = _total_loss_
            state = {'epoch': _epoch_,
                     'net':net,
                     'optimizer': _optimizer_,
                     'total_loss': _total_loss_}
            torch.save(state, MyParam.CHECKPOINT_FILENAME2)

def main():
    _t1_ = time.time()
    # with open('train_log_2.txt', 'w') as wfd:
    #     wfd.write('')
    for i in range(0, MyParam.LOTTERY_NUM):
        MyParam.TRAIN_NUM_INDEX = i
        MyParam.CHECKPOINT_FILENAME2 = 'checkpoint_2_{0}.ckpt'.format(MyParam.TRAIN_NUM_INDEX)
        train()
    print(datetime.datetime.now())
    print(time.time() - _t1_)

if __name__ == "__main__":
    sys.exit(main())