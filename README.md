# Linear-stacking-ensemble
Stacked ensemble linear regression, combining XGBoost and CatBoost with Lasso and Ridge, to enhance house price predictions.

#Setup

## Dependencies

This project requires the following Python libraries:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- scipy
- scikit-optimize
- xgboost
- catboost
- joblib

## Installation

To install the required dependencies for this project, follow these steps:

1. **Clone the Repository**:

	First, clone the repository to your local machine using the following command:

	```bash
	git clone https://github.com/StavrosChatzipavlidis/stacked-ensemble-regression.git
	```
   
2. **Navigate to the Project Directory**:

	Use the cd command to change your current directory to the cloned repository:

	```bash
	cd your-repository
	```

3. **Install Dependencies**:

	Once you're in the project directory, install the required dependencies using the pip command and the requirements.txt file:

	```bash
	pip install -r requirements.txt
	```

By following these steps, you will set up the project environment with all the required dependencies.

## Importing Libraries

To import the necessary libraries, simply execute the following command:

	```python
	from imports import *
	```
	
# Project Overview: Predicting House Prices

Description:

In this project, we aim to predict house prices using a stacked ensemble regression approach. The dataset contains various features such as the number of bedrooms, bathrooms, square footage of living space, and more. By leveraging machine learning techniques, we aim to build a predictive model that can accurately estimate house prices based on these features.

Data Overview:

Below is an overview of the dataset containing information on house prices and related features:

| Price     | Bedrooms | Bathrooms | Sqft Living | Sqft Lot | Floors | Waterfront | View | Condition | Sqft Above | Sqft Basement | Yr Built | Yr Renovated |
|-----------|----------|-----------|-------------|----------|--------|------------|------|-----------|------------|---------------|----------|--------------|
| 313000.0  | 3.0      | 1.50      | 1340        | 7912     | 1.5    | 0          | 0    | 3         | 1340       | 0             | 1955     | 2005         |
| 2384000.0 | 5.0      | 2.50      | 3650        | 9050     | 2.0    | 0          | 4    | 5         | 3370       | 280           | 1921     | 0            |
| 342000.0  | 3.0      | 2.00      | 1930        | 11947    | 1.0    | 0          | 0    | 4         | 1930       | 0             | 1966     | 0            |
| 420000.0  | 3.0      | 2.25      | 2000        | 8030     | 1.0    | 0          | 0    | 4         | 1000       | 1000          | 1963     | 0            |
| 550000.0  | 4.0      | 2.50      | 1940        | 10500    | 1.0    | 0          | 0    | 4         | 1140       | 800           | 1976     | 1992         |

