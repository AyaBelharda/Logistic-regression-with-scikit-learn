# Titanic Survival Prediction

This repository contains a machine learning project that predicts the survival of passengers on the Titanic using a Logistic Regression model. The dataset used is the Titanic dataset from Stanford University's CS109 class.

## Dataset

The dataset can be found at the following link:
[Titanic Dataset](https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv)

The dataset includes the following features:
- `Pclass`: Passenger class (1st, 2nd, or 3rd)
- `Sex`: Gender of the passenger
- `Age`: Age of the passenger
- `Survived`: Survival status (0 = No, 1 = Yes)

## Project Workflow

1. **Data Loading and Exploration:**
   - Load the dataset using pandas.
   - Explore the dataset and count the number of survivors and non-survivors.

2. **Data Preprocessing:**
   - Handle missing values in the `Age` column by filling them with the median age.
   - Drop rows with missing target or features.
   - Convert the `Sex` column to numeric values (0 for male, 1 for female).

3. **Feature Selection:**
   - Select the features `Pclass`, `Sex`, and `Age` for the model.
   - Set the `Survived` column as the target variable.

4. **Train-Test Split:**
   - Split the dataset into training and testing sets with an 80-20 ratio.

5. **Model Training:**
   - Initialize the Logistic Regression model with balanced class weights to handle the imbalanced dataset.
   - Train the model on the training data.

6. **Model Evaluation:**
   - Make predictions on the testing data.
   - Evaluate the model using accuracy score, confusion matrix, and classification report.

7. **Results Visualization:**
   - Plot the confusion matrix using seaborn's heatmap.

## Results

- **Accuracy:** The model achieved an accuracy of approximately 84.83%.
- **Confusion Matrix:**
  ![Confusion Matrix](confusion_matrix.png)
- **Classification Report:**
