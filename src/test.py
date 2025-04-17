import numpy as np
import pandas as pd
import os

fileName = "test.MOV.npy"
filePath = os.path.join( os.path.dirname( __file__) , "output/", fileName)
print('>> ', filePath)
data = np.load( filePath , allow_pickle=True)
df = pd.DataFrame(data.tolist())
df.to_csv("ball_positions.csv", index=False)
