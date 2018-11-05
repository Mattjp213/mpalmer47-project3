
from DataUtils import DataUtils
import sklearn.preprocessing as preprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE

class TSNEExperiment(object):

    def __init__(self, data_set_file_name):

        self.data_set_file_name = data_set_file_name
        self.data_set = DataUtils.load_data_to_nd_array(data_set_file_name)

    def visualize_dimensionality_reduced_data(self):

        scaler = preprocessing.Normalizer()
        normalized_x = scaler.fit_transform(self.data_set[:, 0:len(self.data_set[0]) - 1])

        tsne = TSNE(n_components=3, perplexity=40, n_iter=300)
        dim_reduced_x = tsne.fit_transform(normalized_x)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = dim_reduced_x[:, 0]
        y = dim_reduced_x[:, 1]
        z = dim_reduced_x[:, 2]
        ax.scatter(x, y, z, c='r', marker='o')
        ax.set_xlabel('Reduced Feature 1')
        ax.set_ylabel('Reduced Feature 2')
        ax.set_zlabel('Reduced Feature 3')
        plt.title("TSNE redux to 3 dimensions for file: " + self.data_set_file_name)
        plt.show()