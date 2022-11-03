import os

""" for i in range(0,10):
    #os.system("python single_cpu_trainer.py --data_name {} > {}.out".format(str(i),str(i)))
    os.system("python single_cpu_trainer.py --data_name {} --data_dir './NAB/realTweets-standard'".format(str(i)))
    os.system("rm  ./ckpt/{}.ckpt".format(str(i)))
    #print("python single_cpu_trainer.py --data_ """

for i in range(187,210):
    #os.system("python single_cpu_trainer.py --data_name {} > {}.out".format(str(i),str(i)))
    os.system("python single_cpu_trainer.py --data_name {} --data_dir './AnoTransfer-data/real-world-data-standard'".format(str(i)))
    os.system("rm  ./ckpt/{}.ckpt".format(str(i)))
    #print("python single_cpu_trainer.py --data_ 