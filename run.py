import os

""" for i in range(0,10):
    #os.system("python single_cpu_trainer.py --data_name {} > {}.out".format(str(i),str(i)))
    os.system("python single_cpu_trainer.py --data_name {} --data_dir './NAB/realTweets-standard'".format(str(i)))
    os.system("rm  ./ckpt/{}.ckpt".format(str(i)))
    #print("python single_cpu_trainer.py --data_ """
#filelist = [5,14,16,17,19,27,29,32,33,35,36,37,46,47,50,53,59,62,65,67,68,70,71,72,73,74,76,79,80,83,85]
filelist = [i for i in range(100,200)]
print(filelist)
""" with open('./resultxrsy4.txt','a') as f:
    f.write('mode:0\n')
for i in range(0,210):
    #os.system("python single_cpu_trainer.py --data_name {} > {}.out".format(str(i),str(i)))
    if i in filelist:
        os.system("python single_cpu_trainer.py --data_name {} --data_dir './AnoTransfer-data/real-world'  --exp_mode 0".format(str(i)))
        os.system("rm  ./ckpt/{}.ckpt".format(str(i)))
    #print("python single_cpu_trainer.py --data_ 

with open('./resultxrsy4.txt','a') as f:
    f.write('mode:1\n')
for i in range(20,210):
    #os.system("python single_cpu_trainer.py --data_name {} > {}.out".format(str(i),str(i)))
    if i in filelist:
        os.system("python single_cpu_trainer.py --data_name {} --data_dir './AnoTransfer-data/real-world' --n_positive 0 --exp_mode 1".format(str(i)))
        os.system("rm  ./ckpt/{}.ckpt".format(str(i))) """
with open('./resultxrsy4.txt','a') as f:
    f.write('mode:2\n')
for i in range(0,210):
    #os.system("python single_cpu_trainer.py --data_name {} > {}.out".format(str(i),str(i)))
    if i in filelist:
        os.system("python single_cpu_trainer.py --data_name {} --data_dir './AnoTransfer-data/real-world' --n_negative 0 --exp_mode 2".format(str(i)))
        os.system("rm  ./ckpt/{}.ckpt".format(str(i)))
""" with open('./resultxrsy4.txt','a') as f:
    f.write('mode:3\n')
for i in range(0,210):
    #os.system("python single_cpu_trainer.py --data_name {} > {}.out".format(str(i),str(i)))
    if i in filelist:
        os.system("python single_cpu_trainer.py --data_name {} --data_dir './AnoTransfer-data/real-world'  --n_random 0 --exp_mode 3".format(str(i)))
        os.system("rm  ./ckpt/{}.ckpt".format(str(i)))
with open('./resultxrsy4.txt','a') as f:
    f.write('mode:4\n')
for i in range(0,210):
    #os.system("python single_cpu_trainer.py --data_name {} > {}.out".format(str(i),str(i)))
    if i in filelist:
        os.system("python single_cpu_trainer.py --data_name {} --data_dir './AnoTransfer-data/real-world'  --exp_mode 4".format(str(i)))
        os.system("rm  ./ckpt/{}.ckpt".format(str(i)))
with open('./resultxrsy4.txt','a') as f:
    f.write('mode:5\n')
for i in range(0,210):
    #os.system("python single_cpu_trainer.py --data_name {} > {}.out".format(str(i),str(i)))
    if i in filelist:
        os.system("python single_cpu_trainer.py --data_name {} --data_dir './AnoTransfer-data/real-world'  --exp_mode 5".format(str(i)))
        os.system("rm  ./ckpt/{}.ckpt".format(str(i)))
with open('./resultxrsy4.txt','a') as f:
    f.write('mode:6\n')
for i in range(0,210):
    #os.system("python single_cpu_trainer.py --data_name {} > {}.out".format(str(i),str(i)))
    if i in filelist:
        os.system("python single_cpu_trainer.py --data_name {} --data_dir './AnoTransfer-data/real-world'  --n_positive 0 --n_negative 0 --exp_mode 6".format(str(i)))
        os.system("rm  ./ckpt/{}.ckpt".format(str(i)))
with open('./resultxrsy4.txt','a') as f:
    f.write('mode:7\n')
for i in range(0,210):
    #os.system("python single_cpu_trainer.py --data_name {} > {}.out".format(str(i),str(i)))
    if i in filelist:
        os.system("python single_cpu_trainer.py --data_name {} --data_dir './AnoTransfer-data/real-world'  --n_positive 0 --n_random 0 --exp_mode 7".format(str(i)))
        os.system("rm  ./ckpt/{}.ckpt".format(str(i))) """
with open('./resultxrsy4.txt','a') as f:
    f.write('mode:8\n')
for i in range(0,210):
    #os.system("python single_cpu_trainer.py --data_name {} > {}.out".format(str(i),str(i)))
    if i in filelist:
        os.system("python single_cpu_trainer.py --data_name {} --data_dir './AnoTransfer-data/real-world'  --n_random 0 --n_negative 0 --exp_mode 8".format(str(i)))
        os.system("rm  ./ckpt/{}.ckpt".format(str(i)))
""" with open('./resultxrsy4.txt','a') as f:
    f.write('mode:9\n')
for i in range(0,210):
    #os.system("python single_cpu_trainer.py --data_name {} > {}.out".format(str(i),str(i)))
    if i in filelist:
        os.system("python single_cpu_trainer.py --data_name {} --data_dir './AnoTransfer-data/real-world'   --exp_mode 9".format(str(i)))
        os.system("rm  ./ckpt/{}.ckpt".format(str(i))) """
with open('./resultxrsy4.txt','a') as f:
    f.write('mode:10\n')
for i in range(0,210):
    #os.system("python single_cpu_trainer.py --data_name {} > {}.out".format(str(i),str(i)))
    if i in filelist:
        os.system("python single_cpu_trainer.py --data_name {} --data_dir './AnoTransfer-data/real-world'  --n_positive 0 --n_negative 0 --n_random 0 --exp_mode 10".format(str(i)))
        os.system("rm  ./ckpt/{}.ckpt".format(str(i)))