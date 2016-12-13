
import pandas as pd
import pylab as plt
from mpl_toolkits.basemap import Basemap
import seaborn as sns
sns.set_context("paper", font_scale=2)
sns.set_style("white")

def main() :

	plt.close('all')

	point_size = 8
	alpha = 0.25

	# Load
	data = pd.read_csv('data-police-shootings/fatal-police-shootings-data-modified.csv')

	# create the map
	basemap = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
			projection='lcc',lat_1=33,lat_2=45,lon_0=-95, resolution='c')


	# load the shapefile, use the name 'states'
	basemap.readshapefile('/home/john/Downloads/basemap-1.0.7rel/examples/st99_d00', name='states', drawbounds=True)

	latitudes = data['Latitude'].tolist()
	longitudes = data['Longitude'].tolist()

	coordinates = zip(longitudes, latitudes)
	kill_count = {}

	for coordinate in coordinates :
		if coordinate in kill_count:
			kill_count[coordinate] = kill_count[coordinate] + point_size
		else:
			kill_count[coordinate] = point_size

	colour = 0
	step = 1.0/len(kill_count)

	for coordinate in kill_count :
		x, y = basemap(coordinate[0], coordinate[1])
		basemap.scatter(x, y, kill_count[coordinate], marker='o', color=[colour, alpha, alpha])
		colour = colour + step
		print colour


	# Show map
	plt.show()

if __name__ == "__main__" :
	main()