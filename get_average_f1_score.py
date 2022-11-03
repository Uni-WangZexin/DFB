import numpy as np
f = open('./result.txt')
score=np.array([])
tag = 0
for line in f:
    if tag==0:
        if line.strip() =='NAB':
            tag=1
        continue
    if line.strip()=='':
        tag=0
        continue
    print(line)
    l = line.strip().split(' ')
    score = np.append(score, float(l[6]))
print(score)
print(np.mean(score))
    