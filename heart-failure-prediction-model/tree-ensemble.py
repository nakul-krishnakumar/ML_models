import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier, callback
from utils import plot_train_val_metrics

RANDOM_STATE = 47 ## We will pass it to every sklearn call so we ensure reproducibility

"""
    FEATURE DETAILS 
    - Age: age of the patient [years]
    - Sex: sex of the patient [M: Male, F: Female]
    - ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
    - RestingBP: resting blood pressure [mm Hg]
    - Cholesterol: serum cholesterol [mm/dl]
    - FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
    - RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST      elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
    - MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
    - ExerciseAngina: exercise-induced angina [Y: Yes, N: No]
    - Oldpeak: oldpeak = ST [Numeric value measured in depression]
    - ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
    - HeartDisease: output class [1: heart disease, 0: Normal]

"""

df = pd.read_csv('heart.csv')
print(df.head())

cat_variables = [
    'Sex',
    'ChestPainType',
    'RestingECG',
    'ExerciseAngina',
    'ST_Slope'
]

# one-hot encoding features with non-binary values
df = pd.get_dummies(
    data = df,
    prefix = cat_variables,
    columns = cat_variables
)
print(df.head())

# Here, target output is the variable 'HeartDisease'
# All other variables are features that can potentially be used to predict the target, 'HeartDisease'

features = [x for x in df.columns if x not in 'HeartDisease'] ## Removing our target variable
print(len(features))

# Splitting the Dataset
X_train, X_val, y_train, y_val = train_test_split(
    df[features],
    df['HeartDisease'],
    train_size=0.8,
    random_state=RANDOM_STATE,
    # shuffle = True by default
)
# We will keep the shuffle = True since our dataset has not any time dependency.

print(f'train samples: {len(X_train)}')
print(f'validation samples: {len(X_val)}')
print(f'target proportion: {sum(y_train)/len(y_train):.4f}')


# -------------------- DECISION TREE CLASSIFIER ----------------------------------------------------------------------------
"""
    Hyperparameters
        - min_samples_split -> The minimum number of samples required to split an internal node
        - max_depth -> The maximum depth of the tree
"""
min_samples_split_list = [2, 10, 30, 50, 100, 200, 300, 700] ## If the number is an integer, then it is the actual quantity of samples
max_depth_list = [1, 2, 3, 4, 8, 16, 32, 64, None] # None means that there is no depth limit.


# Calculate Train & Validation Set Accuracy on different min_samples_split values
accuracy_list_train = []
accuracy_list_val = []
for min_samples_split in min_samples_split_list:
    # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    model = DecisionTreeClassifier(
        min_samples_split=min_samples_split,
        random_state=RANDOM_STATE,
    ).fit(X_train, y_train)

    predictions_train = model.predict(X_train) ## The predicted values for the train dataset
    predictions_val = model.predict(X_val) ## The predicted values for the test dataset

    accuracy_train = accuracy_score(predictions_train, y_train)
    accuracy_val = accuracy_score(predictions_val, y_val)

    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

plot_train_val_metrics(
    xlabel='min_samples_split',
    xticks=min_samples_split_list,
    accuracy_list_train=accuracy_list_train,
    accuracy_list_val=accuracy_list_val
)

# Calculate Train & Validation Set Accuracy on different max_depth values
accuracy_list_train = []
accuracy_list_val = []
for max_depth in max_depth_list:
    # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    model = DecisionTreeClassifier(max_depth = max_depth,
                                   random_state = RANDOM_STATE).fit(X_train,y_train) 
    predictions_train = model.predict(X_train) ## The predicted values for the train dataset
    predictions_val = model.predict(X_val) ## The predicted values for the test dataset
    accuracy_train = accuracy_score(predictions_train,y_train)
    accuracy_val = accuracy_score(predictions_val,y_val)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

plot_train_val_metrics(
    xlabel='max_depth',
    xticks=max_depth_list,
    accuracy_list_train=accuracy_list_train,
    accuracy_list_val=accuracy_list_val
)

"""
    From the graphs plotted above, we can choose our hyperparameters
    - max_depth = 4
    - min_samples_split = 50
"""

decision_tree_model = DecisionTreeClassifier(
    min_samples_split = 50,
    max_depth = 4,
    random_state = RANDOM_STATE
).fit(X_train, y_train)

print(f"Metrics train:\n\tAccuracy score: {accuracy_score(decision_tree_model.predict(X_train),y_train):.4f}")
print(f"Metrics validation:\n\tAccuracy score: {accuracy_score(decision_tree_model.predict(X_val),y_val):.4f}")

# --------------------------------------------------------------------------------------------------------------------------

# -------------------- RANDOM FOREST CLASSIFIER ----------------------------------------------------------------------------

"""
    Hyperparameters
        - min_samples_split -> The minimum number of samples required to split an internal node
        - max_depth -> The maximum depth of the tree
        - n_estimators -> number of Decision Trees that make up the Random Forest.
        - max_features -> number of features that the model randomly selects and evaluates at each split in the tree-building process.
        - n_jobs -> used for parallel processing and building dif trees parallely
"""
min_samples_split_list = [2,10, 30, 50, 100, 200, 300, 700]  ## If the number is an integer, then it is the actual quantity of samples,
                                             ## If it is a float, then it is the percentage of the dataset
max_depth_list = [2, 4, 8, 16, 32, 64, None]
n_estimators_list = [10,50,100,500]

# Calculate Train & Validation Set Accuracy on different min_samples_split values
accuracy_list_train = []
accuracy_list_val = []
for min_samples_split in min_samples_split_list:
    # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    model = RandomForestClassifier(min_samples_split = min_samples_split,
                                   random_state = RANDOM_STATE).fit(X_train,y_train) 
    predictions_train = model.predict(X_train) ## The predicted values for the train dataset
    predictions_val = model.predict(X_val) ## The predicted values for the test dataset
    accuracy_train = accuracy_score(predictions_train,y_train)
    accuracy_val = accuracy_score(predictions_val,y_val)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

plot_train_val_metrics(
    xlabel='min_samples_split',
    xticks=min_samples_split_list,
    accuracy_list_train=accuracy_list_train,
    accuracy_list_val=accuracy_list_val
)

# Calculate Train & Validation Set Accuracy on different max_depth values
accuracy_list_train = []
accuracy_list_val = []
for max_depth in max_depth_list:
    # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    model = RandomForestClassifier(max_depth = max_depth,
                                   random_state = RANDOM_STATE).fit(X_train,y_train) 
    predictions_train = model.predict(X_train) ## The predicted values for the train dataset
    predictions_val = model.predict(X_val) ## The predicted values for the test dataset
    accuracy_train = accuracy_score(predictions_train,y_train)
    accuracy_val = accuracy_score(predictions_val,y_val)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

plot_train_val_metrics(
    xlabel='max_depth',
    xticks=max_depth_list,
    accuracy_list_train=accuracy_list_train,
    accuracy_list_val=accuracy_list_val
)

# Calculate Train & Validation Set Accuracy on different n_estimators values
accuracy_list_train = []
accuracy_list_val = []
for n_estimators in n_estimators_list:
    # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    model = RandomForestClassifier(n_estimators = n_estimators,
                                   random_state = RANDOM_STATE).fit(X_train,y_train) 
    predictions_train = model.predict(X_train) ## The predicted values for the train dataset
    predictions_val = model.predict(X_val) ## The predicted values for the test dataset
    accuracy_train = accuracy_score(predictions_train,y_train)
    accuracy_val = accuracy_score(predictions_val,y_val)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

plot_train_val_metrics(
    xlabel='n_estimators',
    xticks=n_estimators_list,
    accuracy_list_train=accuracy_list_train,
    accuracy_list_val=accuracy_list_val
)

"""
    From the graphs plotted above, we can choose our hyperparameters
    - max_depth = 16
    - min_samples_split = 10
    - n_estimators = 100
"""

random_forest_model = RandomForestClassifier(
    n_estimators = 100,
    max_depth = 16, 
    min_samples_split = 10
).fit(X_train,y_train)

print(f"Metrics train:\n\tAccuracy score: {accuracy_score(random_forest_model.predict(X_train),y_train):.4f}\n")
print(f"Metrics test:\n\tAccuracy score: {accuracy_score(random_forest_model.predict(X_val),y_val):.4f}")

# --------------------------------------------------------------------------------------------------------------------------

# ------------------------------ XGBClassifier -----------------------------------------------------------------------------

n = int(len(X_train)*0.8) ## Let's use 80% to train and 20% to eval
X_train_fit, X_train_eval, y_train_fit, y_train_eval = X_train[:n], X_train[n:], y_train[:n], y_train[n:]

xgb_model = XGBClassifier(
    n_estimators = 500, 
    learning_rate = 0.1,
    verbosity = 1, 
    random_state = RANDOM_STATE,
    early_stopping_rounds=10,
)
xgb_model.fit(
    X_train_fit,
    y_train_fit, 
    eval_set = [(X_train_eval,y_train_eval)]
)

print("Best iteration round: ", xgb_model.best_iteration)

print(f"Metrics train:\n\tAccuracy score: {accuracy_score(xgb_model.predict(X_train),y_train):.4f}\n")
print(f"Metrics test:\n\tAccuracy score: {accuracy_score(xgb_model.predict(X_val),y_val):.4f}")

