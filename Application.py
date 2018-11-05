
# Matt Palmer, mpalmer47@gatech.edu, GTID: 903336804
# CS 7641 Fall 2018 Project 3

from KMeansClusteringExperiment import KMeansClusteringExperiment
from ExpectationMaximizationClusteringExperiment import ExpectationMaximizationClusteringExperiment
from PCAExperiment import PCAExperiment
from ICAExperiment import ICAExperiment
from RandomizedProjectionsExperiment import RandomizedProjectionsExperiment
from TSNEExperiment import TSNEExperiment

WINE_DATA_SET_FILE_NAME = "winequalitydataset"
ADULT_DATA_SET_FILE_NAME = "adultdataset"

def run_k_means_clustering_experiment(data_set_file_name):
    k_means_clustering_experiment = KMeansClusteringExperiment(data_set_file_name)
    k_means_clustering_experiment.run_experiment()

def run_expectation_maximization_clustering_experiment(data_set_file_name):
    expectation_maximization_clustering_experiment = ExpectationMaximizationClusteringExperiment(data_set_file_name)
    expectation_maximization_clustering_experiment.run_experiment()

def run_pca_experiment(data_set_file_name):
    pca_experiment = PCAExperiment(data_set_file_name)
    pca_experiment.visualize_dimensionality_reduced_data()

def run_ica_experiment(data_set_file_name):
    ica_experiment = ICAExperiment(data_set_file_name)
    ica_experiment.visualize_dimensionality_reduced_data()

def run_randomized_projections_experiment(data_set_file_name):
    randomized_projections_experiment = RandomizedProjectionsExperiment(data_set_file_name)
    randomized_projections_experiment.visualize_dimensionality_reduced_data()

def run_tsne_experiment(data_set_file_name):
    tsne_experiment = TSNEExperiment(data_set_file_name)
    tsne_experiment.visualize_dimensionality_reduced_data()

def run_k_means_clustering_experiment_with_pca(data_set_file_name):
    k_means_clustering_experiment = KMeansClusteringExperiment(data_set_file_name)
    k_means_clustering_experiment.run_experiment_with_pca()

def run_k_means_clustering_experiment_with_ica(data_set_file_name):
    k_means_clustering_experiment = KMeansClusteringExperiment(data_set_file_name)
    k_means_clustering_experiment.run_experiment_with_ica()

def run_k_means_clustering_experiment_with_randomized_projections(data_set_file_name):
    k_means_clustering_experiment = KMeansClusteringExperiment(data_set_file_name)
    k_means_clustering_experiment.run_experiment_with_randomized_projections()

def run_expectation_maximization_clustering_experiment_with_pca(data_set_file_name):
    expectation_maximization_clustering_experiment = ExpectationMaximizationClusteringExperiment(data_set_file_name)
    expectation_maximization_clustering_experiment.run_experiment_with_pca()

def run_expectation_maximization_clustering_experiment_with_ica(data_set_file_name):
    expectation_maximization_clustering_experiment = ExpectationMaximizationClusteringExperiment(data_set_file_name)
    expectation_maximization_clustering_experiment.run_experiment_with_ica()

def run_expectation_maximization_clustering_experiment_with_randomized_projections(data_set_file_name):
    expectation_maximization_clustering_experiment = ExpectationMaximizationClusteringExperiment(data_set_file_name)
    expectation_maximization_clustering_experiment.run_experiment_with_randomized_projections()

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
    # - Run the PCA dimensionality reduction experiment on the wine data set.
    #run_pca_experiment(WINE_DATA_SET_FILE_NAME)
    # - Run the PCA dimensionality reduction experiment on the adult data set.
    #run_pca_experiment(ADULT_DATA_SET_FILE_NAME)
    # - Run the ICA dimensionality reduction experiment on the wine data set.
    #run_ica_experiment(WINE_DATA_SET_FILE_NAME)
    # - Run the ICA dimensionality reduction experiment on the adult data set.
    #run_ica_experiment(ADULT_DATA_SET_FILE_NAME)
    # - Run the Randomized Projections dimensionality reduction experiment on the wine data set.
    #run_randomized_projections_experiment(WINE_DATA_SET_FILE_NAME)
    # - Run the Randomized Projections dimensionality reduction experiment on the adult data set.
    #run_randomized_projections_experiment(ADULT_DATA_SET_FILE_NAME)
    # - Run the TSNE dimensionality reduction experiment on the wine data set.
    #run_tsne_experiment(WINE_DATA_SET_FILE_NAME)
    # - Run the TSNE dimensionality reduction experiment on the adult data set.
    #run_tsne_experiment(ADULT_DATA_SET_FILE_NAME)

    ###############################################################################################
    # Part 3 - Rerun clustering algorithms after reducing dimensionality of data sets. ############
    ###############################################################################################
    # - Run the k-means clustering experiment on the wine data set with pca dim redux.
    run_k_means_clustering_experiment_with_pca(WINE_DATA_SET_FILE_NAME)
    # - Run the k-means clustering experiment on the wine data set with ica dim redux.
    run_k_means_clustering_experiment_with_ica(WINE_DATA_SET_FILE_NAME)
    # - Run the k-means clustering experiment on the wine data set with rand-proj. dim redux.
    run_k_means_clustering_experiment_with_randomized_projections(WINE_DATA_SET_FILE_NAME)
    # - Run the k-means clustering experiment on the adult data set with pca dim redux.
    run_k_means_clustering_experiment_with_pca(ADULT_DATA_SET_FILE_NAME)
    # - Run the k-means clustering experiment on the adult data set with ica dim redux.
    run_k_means_clustering_experiment_with_ica(ADULT_DATA_SET_FILE_NAME)
    # - Run the k-means clustering experiment on the adult data set with rand-proj. dim redux.
    run_k_means_clustering_experiment_with_randomized_projections(ADULT_DATA_SET_FILE_NAME)
    # - Run the exp-max. clustering experiment on the wine data set with pca dim redux.
    run_expectation_maximization_clustering_experiment_with_pca(WINE_DATA_SET_FILE_NAME)
    # - Run the exp-max. clustering experiment on the wine data set with ica dim redux.
    run_expectation_maximization_clustering_experiment_with_ica(WINE_DATA_SET_FILE_NAME)
    # - Run the exp-max. clustering experiment on the wine data set with rand-proj. dim redux.
    run_expectation_maximization_clustering_experiment_with_randomized_projections(WINE_DATA_SET_FILE_NAME)
    # - Run the exp-max. clustering experiment on the adult data set with pca dim redux.
    run_expectation_maximization_clustering_experiment_with_pca(ADULT_DATA_SET_FILE_NAME)
    # - Run the exp-max. clustering experiment on the adult data set with ica dim redux.
    run_expectation_maximization_clustering_experiment_with_ica(ADULT_DATA_SET_FILE_NAME)
    # - Run the exp-max. clustering experiment on the adult data set with rand-proj. dim redux.
    run_expectation_maximization_clustering_experiment_with_randomized_projections(ADULT_DATA_SET_FILE_NAME)

    ###############################################################################################
    # Part 4 - Rerun neural network algorithm on 1 data set with reduced dimensionality. ##########
    ###############################################################################################
    # TODO -- Need to implement...

    ###############################################################################################
    # Part 5 - Rerun neural network algorithm on 1 data set with dim redux and reclustering. ######
    ###############################################################################################
    # TODO -- Need to implement...

    print("\nAll done.\n")