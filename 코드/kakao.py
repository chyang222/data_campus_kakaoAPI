#%%
'''
import requests

url = 'https://kauth.kakao.com/oauth/token'
rest_api_key = '929545ff15e313c0f586aca6be8f2d36'
redirect_uri = 'https://naver.com'
authorize_code = 'zfFenvaGP00PWwibcNR-w6vBVAqEemxSMu1_4j9law_S5TAwxW4p-niDgnusg6P3siJxRAo9dRsAAAGC6In0nA'

data = {
    'grant_type':'authorization_code',
    'client_id':rest_api_key,
    'redirect_uri':redirect_uri,
    'code': authorize_code,
    }

response = requests.post(url, data=data)
tokens = response.json()
print(tokens)

# json 저장
import json

with open("kakao_code.json","w") as fp:
    json.dump(tokens, fp)

#P8AmG3_9Y6uB_WaMjDXSNuEs9h-3MuDu1xphtk1TL_jt5YV1HaSUdj4M_fPyGnxPaTxYLAopb7gAAAGCxTDD7Q

'''
#----------------------------------------kakao token---------------------------------------------------------------------------------

import pandas as pd
import xgboost as xgb


df = pd.read_csv("/Users/unixking/Desktop/경기대청년/에이치투오/34/data/MINTdata.csv")
df["GTU"] = df.TPS / 10

df['Date'] = df.TR_YMD_HMS.apply(lambda x: str(x))
df['year'] = df.TR_YMD_HMS.apply(lambda x: str(x)[:4])
df['month'] = df.TR_YMD_HMS.apply(lambda x: str(x)[4:6])
df['day'] = df.TR_YMD_HMS.apply(lambda x: str(x)[6:8])
df['hour'] = df.TR_YMD_HMS.apply(lambda x: str(x)[8:10])
df['mintue'] = df.TR_YMD_HMS.apply(lambda x: str(x)[10:12])

df.Date = pd.to_datetime(df.Date, format="%Y%m%d%H%M%S")

df["dayofweek"] = df.Date.dt.dayofweek

#----------------------------------------GTU----------------------------------------------------------------------------------

def GTU(tr_1, new_df, vali, test):
    x = tr_1[["day", "hour", "mintue", "dayofweek"]]            #train데이터 사용
    y = tr_1[["GTU"]]

    xvali = vali[["day", "hour", "mintue", "dayofweek"]]        #validation 사용
    yvali = vali[["GTU"]]

    eval_set = [(xvali, yvali)]

    new_df["Date"] = test["index"]

    test = test[["day", "hour", "mintue", "dayofweek"]]            #예측데이터

    new_df['day'] = test['day']
    new_df['hour'] = test['hour']
    new_df['mintue'] = test['mintue']
    new_df['dayofweek'] = test['dayofweek']
    new_df.reset_index()

    xgb_model = xgb.XGBRegressor(n_estimators= 250, learning_rate=0.07, subsample=0.7,
                        colsample_bytree=0.7, max_depth=6, silent = 1, nthread = 4, min_child_weight = 4)

    
    xgb_model.fit(x, y, eval_set=eval_set, eval_metric="rmse", early_stopping_rounds=15)
    pred = xgb_model.predict(test)

    new_df['GTU'] = pred

    return new_df

#---------------------------------ELPS_MSEC_SUM----------------------------------------------------------------------------------
def ELPS_SUM(tr_1, new_df, vali):
    x = tr_1[["day", "hour", "mintue", "dayofweek", "GTU"]]
    y = tr_1[["ELPS_AVG"]]

    xvali = vali[["day", "hour", "mintue", "dayofweek", "GTU"]]        #validation 사용
    yvali = vali[["ELPS_AVG"]]

    eval_set = [(xvali, yvali)]

    test = new_df[["day", "hour", "mintue", "dayofweek", "GTU"]]            #예측데이터

    xgb_model = xgb.XGBRegressor(n_estimators=250, learning_rate=0.07, subsample=0.7,
                        colsample_bytree=0.7, max_depth=6, silent = 1, nthread = 4, min_child_weight = 4)

    
    xgb_model.fit(x, y, eval_set= eval_set, eval_metric="rmse", early_stopping_rounds=15)
    pred = xgb_model.predict(test)

    new_df['ELPS_AVG'] = pred

    return new_df

#-----------------------------------------------------ELPS_MSEC_MIN-------------------------------------------------------------------------

def ELPS_MIN(tr_1, new_df, vali):

    x = tr_1[["day", "hour", "mintue", "dayofweek", "GTU"]]
    y = tr_1[["ELPS_MIN"]]

    xvali = vali[["day", "hour", "mintue", "dayofweek", "GTU"]]        #validation 사용
    yvali = vali[["ELPS_MIN"]]

    eval_set = [(xvali, yvali)]

    test = new_df[["day", "hour", "mintue", "dayofweek", "GTU"]]            #예측데이터

    xgb_model = xgb.XGBRegressor(n_estimators=250, learning_rate=0.07, subsample=0.7,
                        colsample_bytree=0.7, max_depth=6, silent = 1, nthread = 4, min_child_weight = 4)


    xgb_model.fit(x, y, eval_set= eval_set, eval_metric="rmse", early_stopping_rounds=15)
    pred = xgb_model.predict(test)

    new_df['ELPS_MIN'] = pred

    return new_df

#-----------------------------------------------------ELPS_MSEC_MAX-------------------------------------------------------------------------

def ELPS_MAX(tr_1, new_df, vali):
    x = tr_1[["day", "hour", "mintue","dayofweek", 'GTU']]
    y = tr_1[["ELPS_MAX"]]

    xvali = vali[["day", "hour", "mintue", "dayofweek", "GTU"]]        #validation 사용
    yvali = vali[["ELPS_MAX"]]

    eval_set = [(xvali, yvali)]

    test = new_df[["day", "hour", "mintue", "dayofweek", "GTU"]]            #예측데이터

    xgb_model = xgb.XGBRegressor(n_estimators=250, learning_rate=0.07, subsample=0.7,
                        colsample_bytree=0.7, max_depth=6, silent = 1, nthread = 4, min_child_weight = 4)


    xgb_model.fit(x, y, eval_set= eval_set, eval_metric="rmse", early_stopping_rounds=15)
    pred = xgb_model.predict(test)

    new_df['ELPS_MAX'] = pred

    return new_df


#--------------------------------------------------------------TPS--------------------------------------------------------------------------------

df.day = df.day.astype('int')
df.hour = df.hour.astype('int')
df.mintue = df.mintue.astype('int')
new_df = pd.DataFrame()


#### tr_1은 7월 31일부터 8월 6일까지 7일치 데이터
tr_1 = df[(df['TR_CODE'] == 'NRSTNCS12000') & ((df['day'] == 31) | (df['day'] <= 6))]
tr_1 = tr_1[["TR_CODE", 'TPS', 'GTU', 'ELPS_AVG', 'ELPS_MIN', 'ELPS_MAX', 'Date', 'year', 'month', 'day', 'hour', 'mintue', 'dayofweek']]


#### comp은 8월 7일부터 8월 11일까지 5일치 데이터
comp = df[(df['TR_CODE'] == 'NRSTNCS12000') & ((df['day'] != 31) & (df['day'] >= 7))]
comp = comp[["TR_CODE", 'TPS', 'GTU', 'ELPS_AVG', 'ELPS_MIN', 'ELPS_MAX', 'Date', 'year', 'month', 'day', 'hour', 'mintue', 'dayofweek']]

#----------------------------------------test dataset----------------------------------------------------------------------------------
testdataset = pd.date_range("2022/08/11 19:10", "2022/08/13 19:10", freq="10T")

testdata = pd.DataFrame(range(len(testdataset)), index=testdataset)

testdata["year"] = testdata.index.year # 연도 정보
testdata["month"] = testdata.index.month # 월 정보
testdata["day"] = testdata.index.day # 일 정보
testdata["hour"] = testdata.index.hour # 시간 정보
testdata["mintue"] = testdata.index.minute # 분 정보
testdata["dayofweek"] = testdata.index.dayofweek

testdata = testdata.reset_index()
testdata = testdata.drop(0, axis=1)

#----------------------------------------main----------------------------------------------------------------------------------

if __name__ == '__main__':
    new_df = GTU(tr_1, new_df, comp, testdata)
    new_df = ELPS_SUM(tr_1, new_df, comp)
    new_df = ELPS_MIN(tr_1, new_df, comp)
    new_df = ELPS_MAX(tr_1, new_df, comp)

    x = tr_1[["day", "hour", "mintue","dayofweek", 'GTU', "ELPS_AVG", "ELPS_MIN", "ELPS_MAX"]]
    y = tr_1[["TPS"]]

    xvali = comp[["day", "hour", "mintue","dayofweek", 'GTU', "ELPS_AVG", "ELPS_MIN", "ELPS_MAX"]]
    yvali = comp[["TPS"]]

    eval_set = [(xvali, yvali)]

    test = new_df[["day", "hour", "mintue", "dayofweek", "GTU", "ELPS_AVG", "ELPS_MIN", "ELPS_MAX"]]          

    xgb_model = xgb.XGBRegressor(n_estimators=950, learning_rate=0.07, subsample=0.7,
                        colsample_bytree=0.7, max_depth=6, silent = 1, nthread = 4, min_child_weight = 4)


    xgb_model.fit(x, y, eval_set= eval_set, eval_metric="rmse", early_stopping_rounds=15)

    pred = xgb_model.predict(test)

    new_df["TPS"] = pred

    new_df.Date = new_df.Date.astype('str')
    new_df['hi'] = new_df.Date.apply(lambda x: x[:4] + x[5:7] + x[8:10] + x[11:13] + "00")

#----------------------------------------------------------visualization 01------------------------------------------------------------------

import matplotlib.pyplot as plt
from matplotlib import dates
from matplotlib import rc
rc('font', family='AppleGothic')

tr_1["TPS_MIN"] = tr_1["TPS"] * 0.33
tr_1["TPS_MAX"] = tr_1["TPS"] * 1.66

comp["TPS_MIN"] = comp["TPS"] * 0.33
comp["TPS_MAX"] = comp["TPS"] * 1.66

new_df["TPS_MIN"] = new_df["TPS"] * 0.33
new_df["TPS_MAX"] = new_df["TPS"] * 1.66


fig = plt.figure(figsize=(18,18)) ## 캔버스 생성
fig.set_facecolor('white') ## 캔버스 색상 설정
ax = fig.add_subplot() ## 그림 뼈대(프레임) 생성

plt.xlabel("date_time") 
plt.ylabel('TPS')
plt.xticks(rotation=45)
ax.xaxis.set_major_locator(dates.MintueLocator(interval = 10))

ax.plot(tr_1['Date'], tr_1["TPS"], marker='',label= '7/31 - 8/6 (train)', color = "g", linewidth = 2.5)
ax.plot(tr_1['Date'], tr_1["TPS_MIN"], color = "seagreen", linewidth = 0.4)
ax.plot(tr_1['Date'], tr_1["TPS_MAX"], color = "seagreen", linewidth = 0.4)
plt.fill_between(tr_1['Date'], tr_1["TPS_MIN"], tr_1["TPS_MAX"], color = "seagreen", alpha=0.15)


ax.plot(comp['Date'], comp["TPS"], marker='', label= "8/7 - 8/11 (validation)", color = "blue",linewidth = 2.5) 
ax.plot(comp['Date'], comp["TPS_MIN"], color = "cornflowerblue", linewidth = 0.4)
ax.plot(comp['Date'], comp["TPS_MAX"], color = "cornflowerblue", linewidth = 0.4)
plt.fill_between(comp['Date'], comp["TPS_MIN"], comp["TPS_MAX"], color = "cornflowerblue", alpha=0.15)


ax.plot(new_df['Date'], new_df["TPS"], marker='', label= "8/11 19시 - 8/13 19시", color="red", linewidth = 2.5) 
ax.plot(new_df['Date'], new_df["TPS_MIN"], color = "lightcoral", linewidth = 0.4)
ax.plot(new_df['Date'], new_df["TPS_MAX"], color = "lightcoral", linewidth = 0.4)
plt.fill_between(new_df['Date'], new_df["TPS_MIN"], new_df["TPS_MAX"], color = "lightcoral", alpha=0.15)

plt.legend()
plt.show()

#----------------------------------------------------------visualization 02------------------------------------------------------------------

tr_1["TPS_MIN"] = tr_1["TPS"] * 0.33
tr_1["TPS_MAX"] = tr_1["TPS"] * 1.66

comp["TPS_MIN"] = comp["TPS"] * 0.33
comp["TPS_MAX"] = comp["TPS"] * 1.66

new_df["TPS_MIN"] = new_df["TPS"] * 0.33
new_df["TPS_MAX"] = new_df["TPS"] * 1.66

tr_1 = tr_1.reset_index()
comp = comp.reset_index()
new_df = new_df.reset_index()

fig = plt.figure(figsize=(18,18)) ## 캔버스 생성
fig.set_facecolor('white') ## 캔버스 색상 설정
ax = fig.add_subplot() ## 그림 뼈대(프레임) 생성

plt.xlabel("date_time") 
plt.ylabel('TPS')
plt.xticks(rotation=45)

ax.plot(tr_1.index, tr_1["TPS"], marker='',label= '7/31 - 8/6 (train)', color = "g", linewidth = 2.5)
ax.plot(tr_1.index, tr_1["TPS_MIN"], color = "seagreen", linewidth = 0.4)
ax.plot(tr_1.index, tr_1["TPS_MAX"], color = "seagreen", linewidth = 0.4)
plt.fill_between(tr_1.index, tr_1["TPS_MIN"], tr_1["TPS_MAX"], color = "seagreen", alpha=0.15)


ax.plot(comp.index, comp["TPS"], marker='', label= "8/7 - 8/11 (validation)", color = "blue",linewidth = 2.5) 
ax.plot(comp.index, comp["TPS_MIN"], color = "cornflowerblue", linewidth = 0.4)
ax.plot(comp.index, comp["TPS_MAX"], color = "cornflowerblue", linewidth = 0.4)
plt.fill_between(comp.index, comp["TPS_MIN"], comp["TPS_MAX"], color = "cornflowerblue", alpha=0.15)


ax.plot(new_df.index, new_df["TPS"], marker='', label= "8/11 19시 - 8/13 19시", color="red", linewidth = 2.5) 
ax.plot(new_df.index, new_df["TPS_MIN"], color = "lightcoral", linewidth = 0.4)
ax.plot(new_df.index, new_df["TPS_MAX"], color = "lightcoral", linewidth = 0.4)
plt.fill_between(new_df.index, new_df["TPS_MIN"], new_df["TPS_MAX"], color = "lightcoral", alpha=0.15)

plt.legend()
plt.show()

#----------------------------------------------------------Outlier detection------------------------------------------------------------------

#### for문을 통해 총 10개의 subplot을 만들고 tr_1.index만큼 난수생성후 그 인덱스에 해당하는 random을 집어 넣는다.

import random

out_index = []
out_value = []

for i in range(1,10):
    out_index.append(random.randrange(0, len(new_df)-1))
    out_value.append(random.uniform(-10, 10))

out_df = pd.DataFrame(out_value, index=out_index, columns = ['TPS'])
out_df = out_df.sort_index(ascending=True)

err = []

for i in out_df.index:
    if new_df.TPS_MIN[i] < out_df.TPS[i] < new_df.TPS_MAX[i]:
        err.append(0)
    else:
        err.append(1)

out_df["error"] = err

fal = out_df[out_df.error == 1]
nor = out_df[out_df.error == 0]

fig = plt.figure(figsize=(18,18)) ## 캔버스 생성
fig.set_facecolor('white') ## 캔버스 색상 설정
ax = fig.add_subplot() ## 그림 뼈대(프레임) 생성

plt.xlabel("date_time") 
plt.ylabel('TPS')
plt.xticks(rotation=45)

ax.plot(tr_1.index, tr_1["TPS"], marker='',label= '7/31 - 8/6 (train)', color = "g", linewidth = 2.5)
ax.plot(tr_1.index, tr_1["TPS_MIN"], color = "seagreen", linewidth = 0.4)
ax.plot(tr_1.index, tr_1["TPS_MAX"], color = "seagreen", linewidth = 0.4)
plt.fill_between(tr_1.index, tr_1["TPS_MIN"], tr_1["TPS_MAX"], color = "seagreen", alpha=0.15)


ax.plot(comp.index, comp["TPS"], marker='', label= "8/7 - 8/11 (validation)", color = "blue",linewidth = 2.5) 
ax.plot(comp.index, comp["TPS_MIN"], color = "cornflowerblue", linewidth = 0.4)
ax.plot(comp.index, comp["TPS_MAX"], color = "cornflowerblue", linewidth = 0.4)
plt.fill_between(comp.index, comp["TPS_MIN"], comp["TPS_MAX"], color = "cornflowerblue", alpha=0.15)


ax.plot(new_df.index, new_df["TPS"], marker='', label= "8/11 19시 - 8/13 19시", color="red", linewidth = 2.5) 
ax.plot(new_df.index, new_df["TPS_MIN"], color = "lightcoral", linewidth = 0.4)
ax.plot(new_df.index, new_df["TPS_MAX"], color = "lightcoral", linewidth = 0.4)
plt.fill_between(new_df.index, new_df["TPS_MIN"], new_df["TPS_MAX"], color = "lightcoral", alpha=0.15)


ax.scatter(nor.index, nor.TPS,  marker = 'o', color = "yellow", s = 50)
ax.scatter(fal.index, fal.TPS,  marker = '*', color = "black", s = 50)
plt.legend()
plt.show()


#-----------------------------------------------kakao api----------------------------------------------------------------------------------------------------------------------------------------

'''

import requests
import json

if len(err) >= 1:    
    with open("kakao_code.json","r") as fp:
        tokens = json.load(fp)

    url="https://kapi.kakao.com/v2/api/talk/memo/default/send"

    # kapi.kakao.com/v2/api/talk/memo/default/send 

    headers={
        "Authorization" : "Bearer " + tokens["access_token"]
    }

    data={
        "template_object": json.dumps({
            "object_type": "text",
            "text": "현재 TPS가 예상 TPS를 벗어났습니다.",
            "link": {
                "web_url" : "text, link, button_title, buttons",
                "mobile_web_url" : "text"
            },
            "button_title" : "404"   
        })
    }

    response = requests.post(url, headers=headers, data=data)
    response.status_code
'''



# %%
