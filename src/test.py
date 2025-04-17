import numpy as np
import pandas as pd
import os

fileName = "test3.MP4.npy"
folderPath = os.path.join( os.path.dirname( __file__) , "output/")
filePath = os.path.join( folderPath, fileName)
print('>> ', filePath)
data = np.load( filePath , allow_pickle=True)
df = pd.DataFrame(data.tolist())

outputPath = os.path.join( folderPath, fileName+".csv")
df.to_csv(outputPath, index=False)
