import numpy as np
import math
from enum import Enum

def computeChangeRates(df, interval):
	"""
	For each attribute, adds a column to df that computes df[id][t] - df[id][t-interval]
  Deletes the first interval samples of each id, since the change rate can't be computed for them
	"""
	# Remove the serial number
	ratesColumns = list(df.columns)[2:]
	titles = [column + " Change Rate" for column in ratesColumns]
	for title in titles:
		df[title] = [None for i in range(len(df))]

	for idx, column in enumerate(ratesColumns):
		tmpValues = list(df[column])
		tmpValues = [np.nan] * interval + tmpValues[:-interval]
		dif = np.subtract(df[column], tmpValues)
		df[titles[idx]] = dif

	tmpValues = list(df["serial-number"])
	tmpValues = [np.nan]*interval + tmpValues[:-interval]
	df["dif-serial-number"] = tmpValues
	df = df.loc[df["serial-number"] == df["dif-serial-number"]]
	df.drop(columns=["dif-serial-number"])

	return df

def getLastSamples(df, N):
	"""
	Returns a dataframe with only the N last samples with each serial-number
	"""
	serialNumbers = df["serial-number"].unique()
	toKeep = []
	for serialNumber in serialNumbers:
		indices = df[df["serial-number"] == serialNumber].index[-N:]
		for index in indices:
			toKeep.append(index)

	return df.loc[toKeep]

class HealthStatusAlgorithm(Enum):
	LINEAR = 1

def LinearAlgorithm(mini, maxi, i, n):
	"""
	Linearly map the values [0,n-1] to [maxi, mini]
	"""
	return maxi - math.floor((maxi-(mini-1))*i/n)

def addHealthStatus(df, good, algorithm, maxLevel):
	"""
	A column with a score in [0,maxLevel] is given to each sample
  If good is set, it is always equal to maxLevel
  Else the algorithm is used to give a score in [0,maxLevel-1]
	"""
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
		newValues = [func(0, maxLevel-1, i, cnt) for i in range(cnt)]
		healthStatusValues = healthStatusValues + newValues

	df["Health Status"] = healthStatusValues
	return df