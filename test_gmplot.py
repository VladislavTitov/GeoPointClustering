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


gmap2 = gmplot.GoogleMapPlotter(40.001831, -75.078804, 10)

latlng = np.vstack((latitudes, longitudes)).T
print(latlng)
#db = KMeans(n_clusters=3).fit(latlng)
db = DBSCAN(eps=0.001, min_samples=4).fit(latlng)

# temp = np.hsplit(db.cluster_centers_, 2)
# cntr_lat = temp[0].reshape(1, -1)[0]
# cntr_lng = temp[1].reshape(1, -1)[0]
#
# gmap2.scatter(cntr_lat, cntr_lng, 'black', marker=True)

labels = db.labels_
unique_labels = set(labels)
n_clusters_ = len(unique_labels) - (1 if -1 in labels else 0)

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -1:
        continue
    labeled_points = latlng[labels == k]
    split_array = np.hsplit(labeled_points, 2)
    gmap2.scatter(split_array[0].reshape(1, -1)[0], split_array[1].reshape(1, -1)[0], color=to_hex(col), size=70, marker=False)

gmap2.draw("mymap_cntr.html")
