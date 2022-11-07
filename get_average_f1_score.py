import numpy as np
import os
f = open('./30train20valid50test.txt')
score=np.array([])
name=np.array([])
tag = 0
filelist = os.listdir('../AnoTranfer-data/real-world/test')
print(filelist)
for line in f:
    l = line.strip().split(' ')
    name = np.append(name, l[1])
    score = np.append(score, float(l[6]))
print(score)
print(np.mean(score))
print(np.sort(name))