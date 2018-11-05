
# Matt Palmer, mpalmer47@gatech.edu, GTID: 903336804
# CS 7641 Fall 2018 Project 3

from KMeansClusteringExperiment import KMeansClusteringExperiment
from ExpectationMaximizationClusteringExperiment import ExpectationMaximizationClusteringExperiment

WINE_DATA_SET_FILE_NAME = "winequalitydataset"
ADULT_DATA_SET_FILE_NAME = "adultdataset"

def run_k_means_clustering_experiment(data_set_file_name):
    k_means_clustering_experiment = KMeansClusteringExperiment(data_set_file_name)
    k_means_clustering_experiment.run_experiment()

def run_expectation_maximization_clustering_experiment(data_set_file_name):
    expectation_maximization_clustering_experiment = ExpectationMaximizationClusteringExperiment(data_set_file_name)
    expectation_maximization_clustering_experiment.run_experiment()

if __name__ == "__main__":

    print("Application running...\n")

    ###############################################################################################
    # Part 1 - Run the clustering algorithms on the data sets. ####################################
    ###############################################################################################
    # - Run the k-means clustering experiment on the wine data set.
    #run_k_means_clustering_experiment(WINE_DATA_SET_FILE_NAME)
    ###############################################################################################
    # - Run the k-means clustering experiment on the adult data set.
    #run_k_means_clustering_experiment(ADULT_DATA_SET_FILE_NAME)
    ###############################################################################################
    # - Run the expectation maximization clustering experiment on the wine data set.
    #run_expectation_maximization_clustering_experiment(WINE_DATA_SET_FILE_NAME)
    ###############################################################################################
    # - Run the expectation maximization clustering experiment on the adult data set.
    #run_expectation_maximization_clustering_experiment(ADULT_DATA_SET_FILE_NAME)
    ###############################################################################################

    ###############################################################################################
    # Part 2 - Apply the dimensionality reduction algorithms to the data sets. ####################
    ###############################################################################################
    # TODO -- Need to implement...
    # - Run the PCA dimensionality reduction experiment on the wine data set.
    # - Run the PCA dimensionality reduction experiment on the adult data set.
    # - Run the ICA dimensionality reduction experiment on the wine data set.
    # - Run the ICA dimensionality reduction experiment on the adult data set.
    # - Run the Randomized Projections dimensionality reduction experiment on the wine data set.
    # - Run the Randomized Projections dimensionality reduction experiment on the adult data set.
    # - Run the TSNE dimensionality reduction experiment on the wine data set.
    # - Run the TSNE dimensionality reduction experiment on the adult data set.

    ###############################################################################################
    # Part 3 - Rerun clustering algorithms after reducing dimensionality of data sets. ############
    ###############################################################################################
    # TODO -- Need to implement...

    ###############################################################################################
    # Part 4 - Rerun neural network algorithm on 1 data set with reduced dimensionality. ##########
    ###############################################################################################
    # TODO -- Need to implement...

    ###############################################################################################
    # Part 5 - Rerun neural network algorithm on 1 data set with dim redux and reclustering. ######
    ###############################################################################################
    # TODO -- Need to implement...

    print("\nAll done.\n")