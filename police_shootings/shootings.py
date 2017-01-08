import pandas as pd
from pandas import DataFrame
import datetime
from uszipcode import ZipcodeSearchEngine
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#from mpl_toolkits.basemap import Basemap

def main():

	data = pd.read_csv("https://raw.githubusercontent.com/washingtonpost/data-police-shootings/master/fatal-police-shootings-data.csv")

	# Change dates to days
	data['day'] = data.apply(lambda row: date_to_day(row['date']), axis = 1)
	data = data.drop('date', axis=1)

	# Add additional useful features
	features = ['Population', 'Density', 'TotalWages', 'Wealthy', 'Latitude', 'Longitude']
	for feature in features:
		data[feature] = data.apply(lambda row: 
			city_to_feature(row['city'], row['state'], feature),
			axis = 1)

	# Drop rows with nonsensical values for features
	data = data[data.Population != 0]

	# Average wages is more informative than total wages
	data['AverageWages'] = data.apply(lambda row: row['TotalWages']/row['Population'], axis = 1)
	data = data.drop('TotalWages', axis = 1)

	# Drop any rows with missing data
	data = data.dropna()

	# Also names and ids are not useful in this case
	data = data.drop('name', axis = 1)
	data = data.drop('id', axis = 1)

	data.to_csv('data-police-shootings/fatal-police-shootings-data-modified.csv', index=False)


def date_to_day(date_string) :

	day_list = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
	date = datetime.datetime.strptime(date_string, "%Y-%m-%d").date()
	return day_list[date.weekday()]

def city_to_feature(city, state, feature) :

	zipcodes = city_to_zipcodes(city, state)
	for zipcode in zipcodes :
		if zipcode[feature] is None :
			return float('NaN')
	else :
		values = [zipcode[feature] for zipcode in zipcodes]
		return np.mean(values)

def city_to_zipcodes(city, state) :

	zipcodeEngineSearch = ZipcodeSearchEngine()
	try:
		zipcodes = zipcodeEngineSearch.by_city_and_state(city, state)
		if not zipcodes :
			# A zipcode that is known to return null
			return [zipcodeEngineSearch.by_zipcode(97003)]
		return zipcodes
	except ValueError :
		# A zipcode that is known to return null
		return [zipcodeEngineSearch.by_zipcode(97003)]

if __name__ == "__main__" :
	main()