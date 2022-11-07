import pandas as pd
import os
import datapreprocess
import numpy as np
train = np.array([]).reshape(-1,1)
test = np.array([]).reshape(-1,1)
test_label = np.array([]).reshape(-1,1)
if __name__ == '__main__':
    for i in range(0,210):#38
        #os.mkdir('./AnoTransfer-data/real-world-data-standard/{}'.format(str(i)))
        file = './AnoTransfer-data/real-world/{}.csv'.format(str(i))
        df = pd.read_csv(file)

        timestamp, missing, (value, label) = datapreprocess.complete_timestamp(df['timestamp'],(df['value'],df['label']))
        value =value.astype(float)

        missing2 = np.isnan(value)
        missing = np.logical_or(missing,missing2).astype(int)

        #print(np.where(missing==1)[0])
        label = label.astype(float)
        label[np.where(missing==1)[0]]=np.nan
        value[np.where(missing==1)[0]]=np.nan

        df2 = pd.DataFrame()
        df2['timestamp'] = timestamp
        df2['value'] = value
        df2['label'] = label
        df2['missing'] = missing.astype(int)
        #parse label
        df2 = df2.fillna(method = 'bfill')
        df2 = df2.fillna(0)
        #parse value
        df2['value'] = value
        df2 = df2.interpolate(method='linear',limit_direction='forward')
        df2 = df2.interpolate(method='linear',limit_direction='backward')

        df2['label'] = df2['label'].astype(int)
        df2['value'],*_ = datapreprocess.standardize_kpi(df2['value'])
        n1 = int(len(df)*0.3)
        n2 = int(len(df)*0.5)
        df_1 = df2[:n1]
        df_2 = df2[n1:n2]
        df_3 = df2[n2:]
        df_1.to_csv('./AnoTransfer-data/real-world-data-standard/{}/{}_train.csv'.format(str(i),str(i)),index=False,mode='w')
        df_2.to_csv('./AnoTransfer-data/real-world-data-standard/{}/{}_validation.csv'.format(str(i),str(i)),index=False,mode = 'w')
        df_3.to_csv('./AnoTransfer-data/real-world-data-standard/{}/{}_test.csv'.format(str(i),str(i)),index=False,mode = 'w')


        #Transformer Anomaly data
        """ value = np.asarray(df2['value']).reshape(-1,1) 
        label = np.asarray(df2['label']).reshape(-1,1) 
        n = int(len(value)*0.5)
        if not os.path.exists('../Anomaly-Transformer-new/dataset/WSD/{}'.format(str(i))):
            os.mkdir('../Anomaly-Transformer-new/dataset/WSD/{}'.format(str(i)))
        np.save('../Anomaly-Transformer-new/dataset/WSD/{}/{}_train.npy'.format(str(i),str(i)),value[:n])
        np.save('../Anomaly-Transformer-new/dataset/WSD/{}/{}_test.npy'.format(str(i),str(i)),value[n:])
        np.save('../Anomaly-Transformer-new/dataset/WSD/{}/{}_test_label.npy'.format(str(i),str(i)),label[n:]) """
        
"""         value = np.asarray(df2['value']).reshape(-1,1) 
        label = np.asarray(df2['label']).reshape(-1,1)
        n = int(len(value)*0.5) 
        train = np.append(train, value[:n],axis = 0)
        test = np.append(test, value[n:],axis = 0)
        test_label = np.append(test_label, label[n:],axis = 0)
    if not os.path.exists('../Anomaly-Transformer-new/dataset/WSD'):
        os.mkdir('../Anomaly-Transformer-new/dataset/WSD')
    np.save('../Anomaly-Transformer-new/dataset/WSD/WSD_train.npy',train)
    np.save('../Anomaly-Transformer-new/dataset/WSD/WSD_test.npy',test)
    np.save('../Anomaly-Transformer-new/dataset/WSD/WSD_test_label.npy',test_label) """