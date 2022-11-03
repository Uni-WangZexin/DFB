import os
import json
import pandas as pd
import time
import datapreprocess
import numpy as np
path = 'NAB/realTweets'

with open('../NAB/labels/combined_windows.json', encoding='utf-8') as a:
    label_json = json.load(a)
print(label_json)
i = 0
for file_name in os.listdir(path):
    print(i)
    if not (os.path.exists('./NAB/realTweets-standard/{}'.format(str(i)))):
        os.mkdir('./NAB/realTweets-standard/{}'.format(str(i)))
    now_anomaly = label_json[os.path.join(path,file_name)[4:]]
    df = pd.read_csv(os.path.join(path,file_name))
    #print(df)
    df['timestamp'] = df['timestamp'].apply(lambda x:int(time.mktime(time.strptime(x,"%Y-%m-%d %H:%M:%S"))))
    #time.mktime(time.strptime(df['timestamp']))
    print(len(df['timestamp']))
    df['label']=0
    interval = df.loc[1,'timestamp'] - df.loc[0,'timestamp']
    now_anomaly_new = []
    for item in now_anomaly:
        start = int(((time.mktime(time.strptime(item[0][:-7],"%Y-%m-%d %H:%M:%S")))- df.loc[0,'timestamp'])/interval)
        end = int(((time.mktime(time.strptime(item[1][:-7],"%Y-%m-%d %H:%M:%S")))- df.loc[0,'timestamp'])/interval)
        print(start,end)
        df.loc[start:end+1,'label'] = 1
    n1 = int(len(df)*0.3)
    n2 = int(len(df)*0.5)
    df['missing']=0
    df['value'],*_ = datapreprocess.standardize_kpi(df['value'])
    df_1 = df[:n1]
    df_2 = df[n1:n2]
    df_3 = df[n2:]
    df_1.to_csv('./NAB/realTweets-standard/{}/{}_train.csv'.format(str(i),str(i)),index=False,mode='w')
    df_2.to_csv('./NAB/realTweets-standard/{}/{}_validation.csv'.format(str(i),str(i)),index=False,mode = 'w')
    df_3.to_csv('./NAB/realTweets-standard/{}/{}_test.csv'.format(str(i),str(i)),index=False,mode = 'w')
    i+=1

    #Transfomer Anomaly data
    value = np.asarray(df['value']).reshape(-1,1) 
    label = np.asarray(df['label']).reshape(-1,1)
    n = int(len(value)*0.5)
    os.mkdir('../Anomaly-Transformer-new/dataset/NAB/{}'.format(str(i)))
    np.save('../Anomaly-Transformer-new/dataset/NAB/{}/{}_train.npy'.format(str(i),str(i)),value[:n])
    np.save('../Anomaly-Transformer-new/dataset/NAB/{}/{}_test.npy'.format(str(i),str(i)),value[n:])
    np.save('../Anomaly-Transformer-new/dataset/NAB/{}/{}_test_label.npy'.format(str(i),str(i)),label[n:])
 