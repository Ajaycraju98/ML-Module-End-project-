# ML-Module-End-project-
This  Assignment explains about the predicting car price using Machine learning algorithms.
# predicting the price of a car
# Key components to be fulfilled:
# 1. Loading and Preprocessing :
Load the dataset and perform necessary preprocessing steps.

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV 

df=pd.read_csv("CarPrice_Assignment.csv")
df

# Extract brand from CarName
df['CarBrand'] = df['CarName'].apply(lambda x: x.split()[0])
df.drop(columns=['CarName', 'car_ID'], inplace=True)  #(Eyeball check)-Dropping the unnecessry columns 

The column carBrand can be split to get the column carName and drop the carBrand,Car_ID column these are unnecessary columns for the dataset.
df.info()
df.describe()
df.shape
null_values=df.isnull().sum()
print("null values of each column:\n",null_values)

df.duplicated().sum()
There is no duplicate entry in this dataset.

categorical_columns=df.select_dtypes(include=["object","category"]).columns.tolist()
numerical_columns=df.select_dtypes(include=["int64","float64"]).columns.tolist()

print("categorical_columns:",categorical_columns)
print("numerical_columns:",numerical_columns)

# In the dataset,there is categorical column.so opted for  Encoding categorical data using label encoder.
categorical_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])
print(df)

X = df.drop(columns=['price'])  # Features
y = df['price']  # Target

select_k = SelectKBest(score_func=f_classif, k=10)  # Selecting top 10 features
X_selected = select_k.fit_transform(X, y)

# Get selected feature names and scores
selected_features = X.columns[select_k.get_support()]
selected_scores = select_k.scores_[select_k.get_support()]

print("Selected Features:", selected_features)
print("Feature Scores based on select_k:", selected_scores)


# Create a DataFrame to display feature names and scores
feature_scores_df = pd.DataFrame({'Feature': selected_features, 'Score': selected_scores})


# Sort by scores in ascending order
feature_scores_df = feature_scores_df.sort_values(by="Score", ascending=False)

# Print results
print("Selected Features:\n", feature_scores_df)

X = df.drop(columns=['price'])  # Features
y = df['price']  # Target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

print("\n Training data(features):")
print(X_train)
print("\n Testing data(features):")
print(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled

# 2. Model Implementation
 Implement the following five regression algorithms:
1) Linear Regression
2) Decision Tree Regressor
3) Random Forest Regressor
4) Gradient Boosting Regressor
5) Support Vector Regressor

models={
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(),
    "Random Forest Regressor":RandomForestRegressor(random_state=42),
    "Gradient Boosting Regressor": GradientBoostingRegressor(),
    "Support Vector Regressor": SVR()
}

# 3. Model Evaluation :
-- Compare the performance of all the models based on R-squared, Mean Squared Error (MSE), and Mean Absolute Error (MAE).
-- Identify the best performing model and justify why it is the best.

model_results={}

for name,model in models.items():
    model.fit(X_train,y_train)
    predictions=model.predict(X_test)
    # Calculate evaluation metrics
    mae=mean_absolute_error(y_test,predictions)
    mse=mean_squared_error(y_test,predictions)
    r2=r2_score(y_test,predictions)
    rmse=mean_squared_error(y_test,predictions,squared=False)
    model_results[name]={"MAE":mae,"MSE":mse,"R2":r2,"RMSE":rmse}

for name,metrics in model_results.items():
    print(f"\n{name} performance:")
    for metric,value in metrics.items():
        print(f"{metric}: {value:.4f}")

best_model=max(model_results,key=lambda x: model_results[x]["R2"])
worst_model=min(model_results,key=lambda x: model_results[x]["R2"])

print("best_model:", best_model,model_results[best_model])
print("worst_model:", worst_model,model_results[worst_model])

# 4. Feature Importance Analysis:
Identify the significant variables affecting car prices (feature selection)

# Feature Importance (Random Forest)
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train_scaled, y_train)
feature_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 Important Features:\n", feature_importances.head(10))

# 5. Hyperparameter Tuning :
Perform hyperparameter tuning and check whether the performance of the model has increased.

For small dataset here using Gradientsearchcv for hyperparametertuning.

models2 = {
     
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "SVR": SVR()
}
params = {
    "Decision Tree": {"max_depth": [3, 5, None]},
    "Random Forest": {"n_estimators": [50, 100]},
    "Gradient Boosting": {"n_estimators": [50, 100]},
    "SVR": {"C": [0.1, 1], "kernel": ["linear"]}
}
# Evaluate models
results = []
for name, model in models2.items():
    grid = GridSearchCV(model, params[name], cv=3, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)
    y_pred = grid.best_estimator_.predict(X_test)
    results.append({"Model": name, "R2 Score": r2_score(y_test, y_pred)})

# Display results
results_df = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False)
print(results_df)

There is a slight variations in the model performance as compared to before hyper parameter tuning.
Since `LinearRegression()` doesn't have hyperparameters for tuning, we use **Ridge Regression**, which allows tuning of the `alpha` parameter (L2 regularization). 

--The End--




