

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import seaborn as sns
sns.set_style("darkgrid", {"axes.facecolor": ".9"})


def main() :
	# https://www.census.gov/quickfacts/table/PST045215/00
	prior_racial_distribution = {"White": 61.6, "Hispanic": 17.6, "Black": 13.3, "Asian": 5.6, "Native": 1.2, "Other": 0.7}

	# Load
	data = pd.read_csv('data-police-shootings/fatal-police-shootings-data-modified.csv')

	labels = []
	x = []
	y = []
	labels.append('White')
	x.append(prior_racial_distribution['White'])
	y.append(data['race'].value_counts()['W'])
	labels.append('Hispanic')
	x.append(prior_racial_distribution['Hispanic'])
	y.append(data['race'].value_counts()['H'])
	labels.append('Black')
	x.append(prior_racial_distribution['Black'])
	y.append(data['race'].value_counts()['B'])
	labels.append('Asian')
	x.append(prior_racial_distribution['Asian'])
	y.append(data['race'].value_counts()['A'])
	labels.append('Native')
	x.append(prior_racial_distribution['Native'])
	y.append(data['race'].value_counts()['N'])
	labels.append('Other')
	x.append(prior_racial_distribution['Other'])
	y.append(data['race'].value_counts()['O'])

	x = normalise(x)
	y = normalise(y)

	z = []

	for pair in zip(x, y) :
		z.append(pair[1]/pair[0])


	print(x)
	print(y)
	print(z)

	ax = sns.barplot(labels, z, alpha=0.75)
	plt.show()

def normalise(unf_data) :
	return [float(element)/sum(unf_data) for element in unf_data]

if __name__ == "__main__" :
	main()

'''
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
sns.set(style="darkgrid", color_codes=True)


def main():

	## Flush
	sns.plt.clf()
	sns.plt.cla()
	sns.plt.close()

	# Load
	#data = pd.read_csv('data-police-shootings/fatal-police-shootings-data-modified.csv')


	print prior_racial_distribution.values()

	plt.figure(figsize=(18.2, 10))
	ax = sns.distplot(prior_racial_distribution.values(), kde=False, label=prior_racial_distribution.keys())




	plt.show()

'''