
f = open('./resultpot2.txt','r')
sum_f1=0.0
sum_trainloss=0.0
sum_testloss=0.0
cnt=0.1
flag=0
""" for line in f:
    #print(line[:4])
    if(line[:4]=='mode'):
        if(flag==0):
            flag+=1
        else:
            print(cnt)
            print(sum/cnt)
        sum=0
        cnt=0
        continue
    line = line.strip().split(' ')
    sum=sum+float(line[6])
    cnt+=1
print(cnt)
print(sum/cnt) """
#filelist = [5,14,16,17,19,27,29,32,33,35,36,37,46,47,50,53,59,62,65,67,68,70,71,72,73,74,76,79,80,83,85]
flag=0
for line in f:
    if(line[:4]=='mode'):
        print(sum_f1/cnt)
        sum_f1=0.0
        print(sum_trainloss/cnt)
        sum_trainloss=0.0
        print(sum_testloss/cnt)
        sum_testloss=0.0
        cnt=0
        continue
    l = line.strip().split(' ')
    sum_f1=sum_f1+float(l[6])
    sum_trainloss = sum_trainloss+float(l[13])
    sum_testloss = sum_trainloss+float(l[16])
    cnt+=1

