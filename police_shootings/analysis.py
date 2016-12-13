
import pandas as pd
from sklearn import preprocessing
import copy
from sklearn.cluster import KMeans
from sklearn import preprocessing
from scipy.stats import pearsonr

def main():

	data = pd.read_csv('data-police-shootings/fatal-police-shootings-data-modified.csv')

	del data['Latitude']
	del data['Longitude']
	del data['city']
	del data['state']


	le = preprocessing.LabelEncoder()
	min_max_scaler = preprocessing.MinMaxScaler()
	numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
	for column in data.columns.values :
		if data[column].dtypes not in numerics :
			le.fit(data[column])
			data[column] = le.transform(data[column])
		x_scaled = min_max_scaler.fit_transform(data[column])
		data[column] = x_scaled	


	intra_class_distance(data)

def intra_class_distance(data) :


	'''

	scores = {}

	for column in data.columns.values :
		current_score = 0
		clone = copy.deepcopy(data)
		del clone[column]
		kmeans = KMeans(n_clusters=1, random_state=0).fit(clone)
		for index, row in clone.iterrows():
			current_score = kmeans.transform(row)

		current_score = current_score[0][0]/len(data)
		scores[column] = current_score

	print scores

	'''

def pearson_scoring(data) :

	scores = {}

	for column in data.columns.values :
		current_score = 0
		clone = copy.deepcopy(data)
		del clone[column]
		for clone_column in clone.columns.values :
			correlation = pearsonr(data[column], clone[clone_column])
			current_score = current_score + correlation[0]

		scores[column] = abs(current_score)

	print scores

#def spectral_score(data) :


	
if __name__ == "__main__" :
	main()