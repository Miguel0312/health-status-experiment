import pandas as pd
import logging

#TODO: improve configuration
NUMBER_OF_SAMPLES = 24
DATA_FILE = "data/mini-baidu-dataset.csv"
# DATA_FILE = "data/baidu-dataset.csv"

# Keeps only the last N samples of each disk on the dataframe
def getLastSamples(df, N):
	serialNumbers = df["serial-number"].unique()
	toKeep = []
	for serialNumber in serialNumbers:
		indices = df.index[-N:]
		for index in indices:
			toKeep.append(index)

	return df.iloc[toKeep]

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,format='%(message)s')

logger.info("Started")

logger.info("Reading data file")
data = pd.read_csv("data/mini-baidu-dataset.csv")

good_hard_drives = data[data["Drive Status"] == 1]
bad_hard_drives = data[data["Drive Status"] == -1]

#print(good_hard_drives)
#print(bad_hard_drives)

bad_hard_drives = getLastSamples(bad_hard_drives, 8)
print(bad_hard_drives)

# TODO: add the rate of change to the table (read how the current reearch does it: hour by hour or compares to another basis)

# TODO: creating training and test samples 


logger.warning("Finished")