
f = open('./resultxrsy3.txt','r')
sum=0.0
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
filelist = [5,14,16,17,19,27,29,32,33,35,36,37,46,47,50,53,59,62,65,67,68,70,71,72,73,74,76,79,80,83,85]
flag=0
for line in f:
    l = line.strip().split(' ')
    if(int(l[1]) not in filelist):
        continue
    if(l[1]=='5'):
        print(sum/cnt)
        sum=0.0
        cnt=0
    sum=sum+float(l[6])
    cnt+=1

