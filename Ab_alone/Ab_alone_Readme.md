 “Project Description” :
🎯 Project Description
In this project, I predicted the age of Abalone based on their physical features using the Abalone dataset. The goal is to estimate the age of abalones non-destructively, without using the traditional method of counting rings under a microscope. The project includes data analysis, linear regression modeling, and a user-friendly GUI.
🛠️ Tools and Technologies
Programming Language: Python
Libraries: pandas, numpy, scikit-learn, joblib, tkinter
Dataset: Abalone Dataset (https://archive.ics.uci.edu/ml/datasets/abalone)
📄 Project Steps
Data Cleaning & Preprocessing:
Fixed headers and named the columns properly.
Converted the categorical sex column to numeric values (encoding).
Removed columns containing null (NaN) values.
Removed outliers to improve data quality.
Data Preparation for Modeling:
Prepared the data for standardization and modeling.
Split the dataset into training and testing sets.
Modeling with Linear Regression:
Trained a Linear Regression model on the prepared data.
Evaluated the model and checked prediction performance.
Packaging the Code with Joblib:
Saved and packaged the model and preprocessing functions using joblib.
This makes the code reusable and easy to integrate with other projects or the GUI.
GUI Development:
Created a user interface to input new Abalone features and get predicted age.
Enabled adding new data and performing calculations directly through the GUI
 without modifying the code.