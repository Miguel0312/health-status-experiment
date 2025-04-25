import pandas as pd
import logging

#TODO: improve configuration
NUMBER_OF_SAMPLES = 8
DATA_FILE = "data/mini-baidu-dataset.csv"
# DATA_FILE = "data/baidu-dataset.csv"
CHANGE_RATE_INTERVAL = 6

def computeChangeRates(df, interval):
	# Remove the serial number
	ratesColumns = list(df.columns)[1:]
	titles = [column + " Change Rate" for column in ratesColumns]
	for column in ratesColumns:
		df = df.assign(column = [None for i in range(len(df))])

	for i in range(interval, len(df)):
		if df.iloc[i]["serial-number"] != df.iloc[i-interval]["serial-number"]:
			continue
		for idx, column in enumerate(ratesColumns):
			df.at[i, titles[idx]] = (df.iloc[i][column] - df.iloc[i-interval][column])/interval

	return df

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

# TODO: check how the normalization is done
good_hard_drives = computeChangeRates(good_hard_drives, CHANGE_RATE_INTERVAL)
bad_hard_drives = computeChangeRates(bad_hard_drives, CHANGE_RATE_INTERVAL)

bad_hard_drives = getLastSamples(bad_hard_drives, NUMBER_OF_SAMPLES)

#TODO: filter columns (z-scores, maybes)

print(list(bad_hard_drives.columns))
print(bad_hard_drives)


# TODO: creating training and test samples 


logger.warning("Finished")