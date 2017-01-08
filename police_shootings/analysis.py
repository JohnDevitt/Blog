
import pandas as pd
from sklearn import preprocessing
import copy
from sklearn.cluster import KMeans
from sklearn import preprocessing
from scipy.stats import pearsonr
import networkx as nx
import numpy as np

def main():


	data = pd.read_csv('data-police-shootings/fatal-police-shootings-data-modified.csv')

	# Drop the location data as it's not longer important
	del data['Latitude']
	del data['Longitude']
	del data['city']
	del data['state']

	# Encode and scale the data
	encoded_data = pd.get_dummies(data)
	scaler = preprocessing.StandardScaler()
	scaled_data = preprocessing.normalize(encoded_data)
	scaled_encoded_dataframe = pd.DataFrame(data=scaled_data, columns=encoded_data.columns.values)

	pearson_scoring(scaled_encoded_dataframe)
	intra_class_distance(scaled_encoded_dataframe)
	spectral_score(scaled_encoded_dataframe)

def intra_class_distance(data) :

	scores = {}

	for column in data.columns.values :
		current_score = 0
		clone = copy.deepcopy(data)
		del clone[column]
		kmeans = KMeans(n_clusters=1, random_state=0).fit(clone)
		for index, row in clone.iterrows():
			current_score = kmeans.transform([row])

		current_score = current_score[0][0]/len(data)
		scores[column] = current_score

	print '----- Intra-class Distance -----'
	print 'Race white: ' + str(scores['race_W'])
	print 'Race black: ' + str(scores['race_B'])
	print 'Race hispanic: ' + str(scores['race_H'])
	print 'Population Density: ' + str(scores['Density'])
	print 'Population Total: ' + str(scores['Population'])
	print 'Wage estimate: ' + str(scores['AverageWages'])
	print 'Wealth: ' + str(scores['Wealthy'])


def pearson_scoring(data) :

	scores = {}

	for column in data.columns.values :
		current_score = 0
		clone = copy.deepcopy(data)
		del clone[column]
		for clone_column in clone.columns.values :
			correlation = pearsonr(data[column], clone[clone_column])
			current_score = current_score + correlation[0]

		scores[column] = abs(current_score)/len(data.columns.values)

	print '----- Perason corrleation -----'
	print 'Race white: ' + str(scores['race_W'])
	print 'Race black: ' + str(scores['race_B'])
	print 'Race hispanic: ' + str(scores['race_H'])
	print 'Population Density: ' + str(scores['Density'])
	print 'Population Total: ' + str(scores['Population'])
	print 'Wage estimate: ' + str(scores['AverageWages'])
	print 'Wealth: ' + str(scores['Wealthy'])

def spectral_score(data) :

	graph = nx.Graph()

	for index, row in data.iterrows():
		graph.add_node(index)

	for index_outer, row_outer in data.iterrows():
		for index_inner, row_inner in data.iterrows() :
			if index_outer is not index_inner:

				sigma = 1
				weight = ( pow(np.linalg.norm(index_outer - index_inner), 2) / 2 * pow(sigma, 2) )
				graph.add_edge(index_outer, index_inner, weight=1)

	print('done')


	
if __name__ == "__main__" :
	main()