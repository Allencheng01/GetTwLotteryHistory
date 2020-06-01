import argparse as argp

SAVE_LIST_FILENAME = 'Lottery649_History.bin'
SAVE_LIST_FILENAME_2 = 'SuperLotto638_History.bin'
LOTTERY_NUM = 7
LOTTERY_HEIGHT = 128
EMBEDDED_CH = 64
LOTTERY_1_MAX_NUM = 49
LOTTERY_2_MAX_NUM = 38
# 0 ~ 6
TRAIN_NUM_INDEX = 0
CHECKPOINT_FILENAME = 'checkpoint{0}.ckpt'.format(TRAIN_NUM_INDEX)
CHECKPOINT_FILENAME2 = 'checkpoint_2_{0}.ckpt'.format(TRAIN_NUM_INDEX)

def ARGS():
    _parser_ = argp.ArgumentParser()
    _parser_.add_argument('--ReFetchLog', dest="ReFetchLog", default=False, type=bool, help='Refetch Log from website')

    return _parser_.parse_args()