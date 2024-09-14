# Data-Science-Internship
This repository consists Case Study of 10 datasets during my Data Science internship. The objective was to apply various Machine Learning algorithms and data analysis on datasets and gain insights. Each dataset was followed with unique challenges and the goal was to develop solutions for classification, regression and clustering problems.

## Tools & Technologies used
### PyCharm 
### Python Libraries :
* Numpy and Pandas - For data manipulation.
* sklearn - To import machine learning algorithms.
* Matplotlib and seaborn - For data visualization.

## Datasets Summaries
1. Census Income dataset
* Objective: Predict whether a person earns over 50K per year based on census data.
* Techniques: Data cleaning (handling missing values, encoding categorical variables), LightGBM Classifier for prediction.
* Key Features: Preprocessed features like age, workclass, education, occupation, etc. Binary classification problem with labels indicating income <50K and >50K. Achieved high accuracy using LightGBM model.

2. Human Activity dataset
* Objective: Classify human activities (such as walking, sitting, and standing) based on accelerometer and gyroscope data.
* Techniques: Data preprocessing (scaling the features, encoding categorical variables), applying Random Forest Classifier for prediction, and evaluating accuracy.
* Key Features: Preprocessed features including accelerometer and gyroscope readings, such as tBodyAcc-mean()-X, tBodyAcc-mean()-Y, tBodyAcc-std()-X, and angle(X,gravityMean). The dataset represents various activities, such as walking, sitting, and standing. Label encoding was used to convert categorical activity labels into numerical form. The data was standardized using a StandardScaler. Random Forest Classifier was used for classification, and the model achieved 98.16% accuracy.

3. Walmart Sales dataset
* Objective: Predict weekly sales for Walmart stores based on historical sales data.
* Techniques: Data preprocessing (handling boolean values, date columns), and using a variety of regression and classification models to predict sales and evaluate performance.
* Key Features: The dataset consists of fields such as Store, Dept, Date, Weekly_Sales, and IsHoliday. The boolean IsHoliday column was encoded to numerical values (0 for False, 1 for True). Various machine learning algorithms were applied to model the relationship between store features and weekly sales, including: Random Forest Regressor (used to predict sales with a root mean square error of 6904.33). Random Forest Classifier, Gradient Boosting, Decision Tree, SVM, MLP Classifier, and others were also tested to predict the sales trend. The Random Forest Regressor provided the best results in terms of error reduction.

4. Black Friday Sales dataset
* Objective: Predict the purchase amount based on user demographics and product-related features.
* Techniques: Data cleaning (handling missing values, encoding categorical variables like Gender, Age, City_Category, and Stay_In_Current_City_Years). Applied several machine learning models for prediction: Random Forest Classifier, Gradient Boosting Classifier, SVM, MLP Classifier, and Random Forest Regressor.
* Key Features: Missing values in Product_Category_2 and Product_Category_3 were filled with a placeholder value of -2. Categorical columns like Gender, Age, City_Category, and Stay_In_Current_City_Years were label-encoded for numerical processing. The target column Purchase was predicted using regression models, with Random Forest Regressor achieving a root mean squared error of 3044.70.

5. Boston dataset
* Objective: Predict the housing price categories (MEDV) based on various housing-related features.
* Techniques: Data preprocessing involved removing irrelevant columns like CHAS, RAD, and B. The target variable MEDV was binned into three categories (0, 1, 2) based on price ranges using the pd.cut function. Categorical target variable was label-encoded to numeric values. Features were analyzed for importance using techniques like SelectKBest and ExtraTreesClassifier. Models used include: Linear Regression for predicting the housing price category.
* Key Features: Outliers in CRIM and ZN were handled by capping values beyond specific thresholds (25.0 for CRIM and 30 for ZN). Important features identified using ExtraTreesClassifier were analyzed, and a bar chart was plotted to visualize the top features.

6. Crime dataset
* Objective: Predict the location of crime incidents based on various features extracted from the dataset. The goal is to understand and predict where crimes are most likely to occur within a city.
* Techniques: Data Cleaning: Handled missing values and dropped irrelevant or non-informative columns. Feature Selection: Used SelectKBest with chi2 to evaluate feature importance, although this was not included in the final model. Model: Applied Random Forest Regressor to predict the crime locations.
* Key Features: Features used include various numeric and categorical attributes related to the crime incidents, such as timestamps, location-related details, and types of crimes.
Target variable is the crime location, which is categorical.

7. Iris dataset
* Objective: Classify iris species based on features of iris flowers.
* Techniques: Applied various classifiers including Logistic Regression, Gradient Boosting, Decision Tree, Random Forest, Support Vector Machine, MLP Classifier, Multinomial Naive Bayes, and Gaussian Naive Bayes. Evaluated models using accuracy score.
* Key Features: Features include measurements like sepal length, sepal width, petal length, and petal width. The target variable is the species of the iris flower.

8. Titanic dataset
* Objective: Predict the survival status of Titanic passengers.
* Techniques: Data Preprocessing: Removed non-essential columns (PassengerId, Age, Name, Sex, Ticket, Cabin, Embarked) and handled missing values in the Fare column by filling them with the mean fare. Features were selected based on importance, with SibSp also removed. Feature Engineering: Categorical variables were converted to numerical values where applicable, although not explicitly shown in the code. Model Training: Used Logistic Regression to predict survival outcomes based on the remaining features. The model was trained on the training dataset and predictions were made on the test dataset.
* Key Features: Key features selected for prediction included Pclass, Parch, and Fare. Data preprocessing ensured that missing values were addressed and irrelevant features were removed to focus on those contributing to the survival prediction.

9. What's cooking dataset
* Objective: Predict the cuisine type based on ingredients used.
* Techniques: Data Preparation: Feature Extraction: Used CountVectorizer to convert ingredient lists into a matrix of token counts. Ingredients for each dish were concatenated into a single string separated by semicolons. Label Encoding: Encoded cuisine labels into numeric values using LabelEncoder. Model Training: Split the data into training and test sets. Trained a Logistic Regression model on the feature matrix (X) and encoded cuisine labels (y).
* Key Features: Top Ingredients: Identified the top 10 ingredients for each cuisine to understand ingredient significance. Accuracy: The Logistic Regression model achieved an accuracy of approximately 79.03% in predicting the cuisine type based on ingredients.
  
10. Wine Quality dataset
* Objective: Predict the quality of wine based on various chemical features.
* Techniques: Feature Engineering: Combined red and white wine datasets, removed irrelevant columns (color and quality), and handled missing values. Feature Selection: Used SelectKBest with chi2 to determine the importance of features. Removed features with lower scores based on the analysis. Trained and evaluated multiple models
* Key Features:
Feature Importance: Residual sugar, free sulfur dioxide, and total sulfur dioxide were identified as key features based on SelectKBest.
Model Performance:
Linear Regression: Mean Squared Error (MSE) = 0.557
Multinomial Naive Bayes: Accuracy = 36.6%
Gaussian Naive Bayes: Accuracy = 48.5%
Random Forest Classifier: Accuracy = 66.0%
Gradient Boosting Classifier: Accuracy = 52.6%
  
