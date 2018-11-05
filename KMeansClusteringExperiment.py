
from DataUtils import DataUtils
from sklearn.cluster import KMeans
import sklearn.preprocessing as preprocessing
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import numpy

class KMeansClusteringExperiment(object):

    def __init__(self, data_set_file_name):

        self.data_set_file_name = data_set_file_name
        self.data_set = DataUtils.load_data_to_nd_array(data_set_file_name)
        self.data_set_feature_labels = DataUtils.load_feature_labels_from_file(data_set_file_name)

    def run_experiment(self):

        scaler = preprocessing.Normalizer()
        normalized_x = scaler.fit_transform(self.data_set[:, 0:len(self.data_set[0]) - 1])

        # Elbow method code taken from https://pythonprogramminglanguage.com/kmeans-elbow-method/

        distortions = []
        K = range(1, len(normalized_x[0]))
        for k in K:
            kmeanModel = KMeans(n_clusters=k).fit(normalized_x)
            distortions.append(sum(numpy.min(cdist(normalized_x, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / normalized_x.shape[0])

        plt.plot(K, distortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title("elbow method for k-means clustering for file: " + self.data_set_file_name)
        plt.show()

        # These were selected by manual inspection of the elbow method graph.
        if self.data_set_file_name == "winequalitydataset":
            k = 4
        elif self.data_set_file_name == "adultdataset":
            k = 4
        else:
            k = 3

        clusterer = KMeans(n_clusters=k).fit(normalized_x)
        labels = clusterer.predict(normalized_x)

        for i in range(len(normalized_x[0])):
            plt.scatter(normalized_x[:, i], self.data_set[:, len(self.data_set[0]) - 1], c=labels, s=50, cmap='viridis')
            min_val = numpy.min(normalized_x[:, i])
            max_val = numpy.max(normalized_x[:, i])
            x_axis_margin = abs(max_val - min_val) * 0.01
            plt.xlim([min_val - x_axis_margin, max_val + x_axis_margin])
            plt.title(str(k) + "-clusters for file: " + self.data_set_file_name + ", feature: " + self.data_set_feature_labels[i])
            plt.show()

        print("K Means Clustering Experiment has been run for file " + self.data_set_file_name + ".csv")