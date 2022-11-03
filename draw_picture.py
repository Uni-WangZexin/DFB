import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

file = './AnoTransfer-data/real-world/0.csv'

df = pd.read_csv(file)
time = df['timestamp']
value = df['value']
plt.plot(time,value)
plt.show()