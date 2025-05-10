from enum import Enum
import numpy as np
import pandas as pd
import math

class FeatureSelectionAlgorithm(Enum):
	Z_SCORE = 1
	
def z_score(goodSamples, badSamples):
	"""
	Input: two lists with the good and bad samples
  Output: the z-score that measure how different the distribution of the values are between the good and bad samples
	"""
	nf = len(badSamples)
	ng = len(goodSamples)
	mf = np.average(badSamples)
	mg = np.average(goodSamples)
	vf = np.var(badSamples)
	vg = np.var(goodSamples)

	if vf == 0 and vg == 0:
		return 0

	return math.fabs(mf-mg)/math.sqrt(vf/nf + vg/ng)

def selectFeatures(df, algorithm, toKeepCount):
	"""
	Returns df with only toKeepCount features corresponding to the ones that score the highest according to algorithm
	"""
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

	results.sort(reverse=True)
	toKeep = list(df.columns)[0:2] + [result[1] for result in results][:toKeepCount]
	
	return df[toKeep]