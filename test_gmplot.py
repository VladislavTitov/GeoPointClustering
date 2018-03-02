import gmplot
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans

gmap1 = gmplot.GoogleMapPlotter(0, 0, 0)

crashes = pd.read_csv("all_crashes_2012.csv", sep=",")

crashes['LAT'].astype('float')
crashes['LNG'].astype('float')

latitudes = crashes['LAT'].values
longitudes = crashes['LNG'].values

print("Latitude type = ", type(latitudes[0]))

# gmap1.scatter(latitudes, longitudes, 'red', marker=True)
# gmap1.draw("mymap.html")


gmap2 = gmplot.GoogleMapPlotter(0, 0, 0)

latlng = np.vstack((latitudes, longitudes)).T
print(latlng)
db = KMeans(n_clusters=3).fit(latlng)

temp = np.hsplit(db.cluster_centers_, 2)
cntr_lat = temp[0].reshape(1, -1)[0]
cntr_lng = temp[1].reshape(1, -1)[0]

gmap2.scatter(cntr_lat, cntr_lng, 'red', marker=True)
gmap2.draw("mymap_cntr.html")
