
from DataUtils import DataUtils
from sklearn import decomposition
import sklearn.preprocessing as preprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ICAExperiment(object):

    def __init__(self, data_set_file_name):

        self.data_set_file_name = data_set_file_name
        self.data_set = DataUtils.load_data_to_nd_array(data_set_file_name)

    def visualize_dimensionality_reduced_data(self):

        dim_reduced_x = self.reduce_data(self.data_set[:, 0:len(self.data_set[0]) - 1])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = dim_reduced_x[:, 0]
        y = dim_reduced_x[:, 1]
        z = dim_reduced_x[:, 2]
        ax.scatter(x, y, z, c='r', marker='o')
        ax.set_xlabel('Reduced Feature 1')
        ax.set_ylabel('Reduced Feature 2')
        ax.set_zlabel('Reduced Feature 3')
        plt.title("ICA redux to 3 dimensions for file: " + self.data_set_file_name)
        plt.show()

    def reduce_data(self, data):

        scaler = preprocessing.Normalizer()
        normalized_x = scaler.fit_transform(data)

        ica = decomposition.FastICA(n_components=3)
        dim_reduced_x = ica.fit_transform(normalized_x)

        return dim_reduced_x