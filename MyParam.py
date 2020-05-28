import argparse as argp

SAVE_LIST_FILENAME = 'Lottery649_History.bin'
LOTTERY_NUM = 7
LOTTERY_HEIGHT = 30
EMBEDDED_CH = 50
# 0 ~ 6
TRAIN_NUM_INDEX = 2
CHECKPOINT_FILENAME = 'checkpoint{0}.ckpt'.format(TRAIN_NUM_INDEX)

def ARGS():
    _parser_ = argp.ArgumentParser()
    _parser_.add_argument('--ReFetchLog', dest="ReFetchLog", default=False, type=bool, help='Refetch Log from website')

    return _parser_.parse_args()