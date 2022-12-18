import os

""" for i in range(0,10):
    #os.system("python single_cpu_trainer.py --data_name {} > {}.out".format(str(i),str(i)))
    os.system("python single_cpu_trainer.py --data_name {} --data_dir './NAB/realTweets-standard'".format(str(i)))
    os.system("rm  ./ckpt/{}.ckpt".format(str(i)))
    #print("python single_cpu_trainer.py --data_ """
filelist = os.listdir('./data/AIOPS2018')
print(filelist)

""" #8
for i in filelist:
    #os.system("python single_cpu_trainer.py --data_name {} > {}.out".format(str(i),str(i)))
    os.system("python single_cpu_trainer.py --data_name {} --data_dir './data/AIOPS2018'  --n_random 0 --n_negative 0 --exp_mode 8 --ar_mode 0".format(str(i[:-4])))
    os.system("rm  -rf ./ckpt/*")

#10
for i in filelist:
    #os.system("python single_cpu_trainer.py --data_name {} > {}.out".format(str(i),str(i)))

    os.system("python single_cpu_trainer.py --data_name {} --data_dir './data/AIOPS2018'  --n_positive 0 --n_negative 0 --n_random 0 --exp_mode 10 --ar_mode 0".format(str(i[:-4])))
    os.system("rm  -rf ./ckpt/*")

#8
for i in filelist:
    #os.system("python single_cpu_trainer.py --data_name {} > {}.out".format(str(i),str(i)))
    os.system("python single_cpu_trainer.py --data_name {} --data_dir './data/AIOPS2018'  --n_random 0 --n_negative 0 --exp_mode 8 --ar_mode 1".format(str(i[:-4])))
    os.system("rm  -rf ./ckpt/*")

#10
for i in filelist:
    #os.system("python single_cpu_trainer.py --data_name {} > {}.out".format(str(i),str(i)))

    os.system("python single_cpu_trainer.py --data_name {} --data_dir './data/AIOPS2018'  --n_positive 0 --n_negative 0 --n_random 0 --exp_mode 10 --ar_mode 1".format(str(i[:-4])))
    os.system("rm  -rf ./ckpt/*") """
#11
""" for i in range(0,210):
    #os.system("python single_cpu_trainer.py --data_name {} > {}.out".format(str(i),str(i)))
    if i in filelist:
        os.system("python single_cpu_trainer.py --data_name {} --data_dir './AnoTransfer-data/real-world'  --n_negative 0 --n_random 0 --exp_mode 11".format(str(i)))
        os.system("rm  ./ckpt/{}.ckpt".format(str(i)))
#12
for i in range(0,210):
    #os.system("python single_cpu_trainer.py --data_name {} > {}.out".format(str(i),str(i)))
    if i in filelist:
        os.system("python single_cpu_trainer.py --data_name {} --data_dir './AnoTransfer-data/real-world'  --n_negative 0 --exp_mode 12".format(str(i)))
        os.system("rm  ./ckpt/{}.ckpt".format(str(i)))
#13
for i in range(0,210):
    #os.system("python single_cpu_trainer.py --data_name {} > {}.out".format(str(i),str(i)))
    if i in filelist:
        os.system("python single_cpu_trainer.py --data_name {} --data_dir './AnoTransfer-data/real-world'  --n_negative 0 --exp_mode 13".format(str(i)))
        os.system("rm  ./ckpt/{}.ckpt".format(str(i))) """


for i in filelist:
    #os.system("python single_cpu_trainer.py --data_name {} > {}.out".format(str(i),str(i)))

    os.system("python single_cpu_trainer.py --data_name {} --data_dir './data/AIOPS2018'  --n_positive 0 --n_negative 0 --n_random 50 --exp_mode 6 --ar_mode 0".format(str(i[:-4])))
    os.system("rm  -rf ./ckpt/*")