# Cardiologist-XGB
 This project was Team 30's group project for MGT 6203 in the Fall 2023 at Georgia Tech.

## Data Source
Data is coming from [Kaggle - Indicators of Heart Disease](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease/data). We are using the `heart_2022_no_nans.csv` file as our data source.

## Data Processing Code
The following describes the general order that the code files where used in and their purpose:
1. data_preprocessing.ipynb -- contains the code for the initial data mining and preprocessing/cleansing
2. CorrPlot_and_VIF.R -- contains the code for the correlation and VIF analysis
3. stepwise.R -- contains the code that creates the stepwise model that is used to perform feature selection
4. test_train_datasets.ipynb -- split the data into test and train sets for classification model analysis
5. xgb_model.ipynb, adaboost_heart.ipynb, & logistic_regression_heart0.1.ipynb -- contain the code and analysis for the respective classification models XG Boost, Ada Boost, and Logistic Regression
6. k_means_clustering.R -- contains the code that implements the k-means clustering on the data
7. per_cluster_logistic_regression.ipynb -- contains the feature importance analysis for each cluster

## Running the code
A small sample set is provided with the code called `heart_2022_no_nans_sampled.csv`. Change this file name to `heart_2022_no_nans.csv` and it will work with the code correctly. The code expects that this file is located in a subdirectory to the main directory called `Data` - i.e. `Team-30/Data/heart_2022_no_nans_sampled.csv`.

To start, run the data_preprocessing.ipynb to perform data preprocessing and transformations. That will output a file `heart_2022_clean.csv` that is used by CorrPlot_and_VIF.R and stepwise.R. CorrPlot_and_VIF.R is not necessary to proceed, but stepwise.R is as it generates a file required for the test_train_datasets.ipynb file, `heart_2022_clean_stepwise0.1.csv`. The test_train_datasets.ipynb file creates the train-set (`train_df0.1.csv`) and test-set (`test_df0.1.csv`) that is utilized by xgb_model.ipynb, adaboost_heart.ipynb, logistic_regression_heart0.1.ipynb, and k_means_clustering.R. The k_means_clustering.R file creates a dataset, `clustered_df.csv`, that is loaded by per_cluster_logistic_regression.ipynb. 

## Contact
If there are any questions or suggestions for improving this repository, please do not hesitate to contact me at joh78@gatech.edu.
Thank you.