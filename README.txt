README for submission of mpalmer47 / Matt Palmer / GTID: 903336804
for Project 3 - Unsupervised Learning and Dimensionality Reduction for CS 7641 Fall 2018.

Source Code Repository:

    - https://github.com/Mattjp213/mpalmer47-project3.git

Depenedencies:

	- Python 3.6
	- matplotlib 3.0.1
	- numpy 1.15.3
	- pandas 0.23.4
	- scikit-learn 0.20.0
	- sklearn 0.0

Source Code Files:

	- Application.py (contains main method of application)

Data Input Files:

	- winequalitydataset.csv (wine quality data set from https://archive.ics.uci.edu/ml/datasets/Wine+Quality - this is
	  the white wine data only)

	- adultdataset.csv (adult data set from https://archive.ics.uci.edu/ml/datasets/Adult)

Running the Application:

	- Running the Application.py file will run any uncommented experiments included in the project automatically. There
	  is 1 call to run each of the individual experiments in the main method of this file and each one is aptly named
	  and also described in a comment directly above each respective method call. All of the calls are commented out
	  except for the call to run the first experiment (clustering with k-means on the first data set). To run other
	  experiments, just uncomment the call for the experiments that you would like to run and then re-run the
	  Application.py file. Some of the experiments genetrate matplotlib graphs which will appear for viewing. Some
	  experiments generate output CSV files which will appear in a folder called "Results" in the root diretory of the
	  project. Each output CSV file will be aptly named so that it will be easy to discern which experiment a given
	  output file was from.