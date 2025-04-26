import pandas as pd
import numpy as np
import logging
import math
from enum import Enum
from tqdm import tqdm

pd.options.mode.copy_on_write = True

#TODO: improve configuration
NUMBER_OF_SAMPLES = 8
DATA_FILE = "data/mini-baidu-dataset.csv"
# DATA_FILE = "data/baidu-dataset.csv"
CHANGE_RATE_INTERVAL = 6
FEATURE_COUNT = 16

class HealthStatusAlgorithm(Enum):
	LINEAR = 1

class FeatureSelectionAlgorithm(Enum):
	Z_SCORE = 1

#TODO: save the result with the change rates on a separate file so you only need to execute it once
def computeChangeRates(df, interval):
	# Remove the serial number
	ratesColumns = list(df.columns)[1:]
	titles = [column + " Change Rate" for column in ratesColumns]
	for title in titles:
		df[title] = [None for i in range(len(df))]

	serial_numbers = df["serial-number"].unique()
	toKeep = []
	for serial_number in tqdm(serial_numbers):
		toKeep = toKeep + list((df[df["serial-number"] == serial_number]).iloc[interval:].index)
	
	values = [[None for i in range(interval)] for i in range(len(ratesColumns))]

	for i in tqdm(toKeep):
		for idx, column in enumerate(ratesColumns):
			values[idx].append((df.iloc[i][column] - df.iloc[i-interval][column])/interval)

	for idx, title in enumerate(titles):
		df[title] = values[idx]


	return df.iloc[toKeep]

def z_score(goodSamples, badSamples):
	nf = len(badSamples)
	ng = len(goodSamples)
	mf = np.average(badSamples)
	mg = np.average(goodSamples)
	vf = np.var(badSamples)
	vg = np.var(goodSamples)

	if vf == 0 and vg == 0:
		return 0

	return math.fabs(mf-mg)/math.sqrt(vf/nf + vg/ng)

def featureSelection(df, algorithm, toKeepCount):
	func = None
	match algorithm:
		case FeatureSelectionAlgorithm.Z_SCORE:
			func = z_score

	# Remove the serial and status
	columns = list(df.columns)[2:]
	good_hard_drives = df[df["Drive Status"] == 1]
	bad_hard_drives = df[df["Drive Status"] == -1]

	results = []

	for col in columns:
		goodSamples = list(good_hard_drives[col])
		badSamples = list(bad_hard_drives[col])

		results.append((func(goodSamples, badSamples), col))

	results.sort()
	toKeep = list(df.columns)[0:2] + [result[1] for result in results][:toKeepCount]
	
	return df[toKeep]

# Keeps only the last N samples of each disk on the dataframe
def getLastSamples(df, N):
	serialNumbers = df["serial-number"].unique()
	toKeep = []
	for serialNumber in serialNumbers:
		indices = df[df["serial-number"] == serialNumber].index[-N:]
		for index in indices:
			toKeep.append(index)

	return df.loc[toKeep]

# Linearly map the values [0,n-1] to [maxi, mini]
# TODO: add an option to decide if the bigger or smaller intervals get less elements
# For now, there are more elements with bigger values
def LinearAlgorithm(mini, maxi, i, n):
	return maxi - math.floor((maxi-(mini-1))*i/n)

# A column with a score in [1,maxLevel] is given to each sample
# If good is set, it is always equal to maxLevel
# Else the algorithm is used to give a score in [1,maxLevel-1]
def addHealthStatus(df, good, algorithm, maxLevel):
	if good:
		df["Health Status"] = [maxLevel for i in range(len(df))]
		return df
	
	func = None

	match algorithm:
		case HealthStatusAlgorithm.LINEAR:
			func = LinearAlgorithm

	serialNumbers = df["serial-number"].unique()
	healthStatusValues = []
	for serialNumber in serialNumbers:
		cnt = len(df[df["serial-number"] == serialNumber])
		newValues = [func(1, maxLevel-1, i, cnt) for i in range(cnt)]
		healthStatusValues = healthStatusValues + newValues

	df["Health Status"] = healthStatusValues
	return df


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,format='%(message)s')

logger.info("Started")

logger.info("Reading data file")
data = pd.read_csv(DATA_FILE)

logger.info("Computing change rates")
data = computeChangeRates(data, CHANGE_RATE_INTERVAL)

good_hard_drives = data[data["Drive Status"] == 1]
bad_hard_drives = getLastSamples(data[data["Drive Status"] == -1], NUMBER_OF_SAMPLES)

data = pd.concat([bad_hard_drives, good_hard_drives])

# TODO: this probably should be done only once on the big dataset in a separate script and then exported here
# This shouldn't be a problem on a Jupyter Notebook, in which results are cached
# TODO: check if the features change a lot when CHANGE_RATE_INTERVAL and NUMBER_OF_SAMPLES change
logger.info("Selecting %d features using the %s algorithm", FEATURE_COUNT, FeatureSelectionAlgorithm.Z_SCORE.name)
data = featureSelection(data, FeatureSelectionAlgorithm.Z_SCORE, FEATURE_COUNT)
logger.info("Features kept: %s", str(list(data.columns)))

good_hard_drives = data[data["Drive Status"] == 1]
bad_hard_drives = getLastSamples(data[data["Drive Status"] == -1], NUMBER_OF_SAMPLES)

bad_hard_drives = addHealthStatus(bad_hard_drives, False, HealthStatusAlgorithm.LINEAR, 4)
good_hard_drives = addHealthStatus(good_hard_drives, True, HealthStatusAlgorithm.LINEAR, 4)

print(len(good_hard_drives))
print(list(good_hard_drives["Health Status"]))

# TODO: creating training and test samples 

logger.warning("Finished")