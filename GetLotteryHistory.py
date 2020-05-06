# BSD 2-Clause License

# Copyright (c) 2020, Allen Cheng
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os, sys
from bs4 import BeautifulSoup
import requests
import re
import datetime
import pickle
import time

Payload = {
'__EVENTTARGET' : '',
'__EVENTARGUMENT' : '',
'__LASTFOCUS' : '',
'__VIEWSTATE' : '',
'__VIEWSTATEGENERATOR' : '',
'__EVENTVALIDATION': '',
'Lotto649Control_history$DropDownList1': '2',
'Lotto649Control_history$chk': 'radYM',
'Lotto649Control_history$dropYear': '103',
'Lotto649Control_history$dropMonth': '4',
'Lotto649Control_history$btnSubmit': '查詢',
}

RequestUrl = "https://www.taiwanlottery.com.tw/Lotto/Lotto649/history.aspx"

def ParsingLotteryByDate(Year, Month):
    print("Parsing {0}/{1} data".format(Year, Month))
    Sess = requests.session()
    Sess.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.129 Safari/537.36'

    Result0 = Sess.get(RequestUrl)
    soup0 = BeautifulSoup(Result0.text, features='lxml')

    for key in Payload.keys():
        if key.startswith('_'):
            tag = soup0.find('input', {'id':key})
            if tag != None:
                Payload[key] = tag['value']
    Payload['Lotto649Control_history$dropYear'] = Year
    Payload['Lotto649Control_history$dropMonth'] = Month

    Result1 = Sess.post(RequestUrl, data=Payload)
    soup1 = BeautifulSoup(Result1.text, features='lxml')

    # LotteryNumbers = soup1.findAll('span', id = re.compile(r'Lotto649Control_history_dlQuery_SNo[\w]+'))
    LotteryNumbers = soup1.findAll(id=re.compile(r'Lotto649Control_history_dlQuery_SNo[\w]+'))
    LotteryDays    = soup1.findAll(id=re.compile(r'Lotto649Control_history_dlQuery_L649_DDate_[\w]+'))
    LotteryIndex    = soup1.findAll(id=re.compile(r'Lotto649Control_history_dlQuery_L649_DrawTerm_[\w]+'))
    ResultList = list()

    if (len(LotteryDays) * 7) != len(LotteryNumbers):
        raise Exception('Could not match the number of LotterDays/LotteryNumbers')
    if len(LotteryDays) != len(LotteryIndex):
        raise Exception('Could not match the number of LotterDays/LotteryIndex')
    for Day_Idx, Days in enumerate(LotteryDays):
        SingleDayDict = dict()
        SingleDayDict['Day'] = Days.text
        SingleDayDict['Index'] = LotteryIndex[Day_Idx].text
        SingleDayDict['Numbers'] = list()
        for Number_Idx in range(7):
            SingleDayDict['Numbers'].append(int(LotteryNumbers[(Day_Idx*7) + Number_Idx].text))
        ResultList.append(SingleDayDict)
    return ResultList

def main():
    StartYear = 103
    EndYear = datetime.datetime.today().year - 1911
    EndMonth = datetime.datetime.today().month

    FinalResultArray = list()

    for TargetYear in range(StartYear, EndYear + 1):
        for TargetMonth in range(1, 13):
            if (TargetYear >= EndYear) and (TargetMonth > EndMonth):
                break
            FinalResultArray += ParsingLotteryByDate('{0:03d}'.format(TargetYear), '{0:01d}'.format(TargetMonth))
            time.sleep(0.5)

    FinalResultArray = sorted(FinalResultArray, key=lambda k: k['Index'])
    with open('FinalResultList.bin', 'wb') as fp:
        pickle.dump(FinalResultArray, fp)
    for item in FinalResultArray:
        print(item)

if __name__ == "__main__":
    sys.exit(main())