import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model import LogisticRegression

ROOT = Path(__file__).parent
NUMERIC_COLUMNS = ['Age', 'Fare', 'Pclass', 'Parch', 'SibSp']
DATASET_ROOT = ROOT / 'dataset'
SEX_MAPPING = {'male': 0, 'female': 1}

# Load the titanic dataset ----------------------------------------------
train_df = pd.read_csv(DATASET_ROOT / 'train.csv')
test_df = pd.read_csv(DATASET_ROOT / 'test.csv')

# Display the first few rows of the dataset
print(train_df.head())
print(test_df.head())

# Data preprocessing ----------------------------------------------------
# Combine the train and test datasets

# Drop the 'PassengerId', 'Name', 'Ticket', 'Cabin' columns
train_df = train_df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
test_df = test_df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

# Fill missing values
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())
train_df['Fare'] = train_df['Fare'].fillna(train_df['Fare'].median())
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())

# Encode categorical columns
train_df['Sex'] = train_df['Sex'].map(SEX_MAPPING)
test_df['Sex'] = test_df['Sex'].map(SEX_MAPPING)

# One-hot encode the 'Embarked' column
# combined = pd.get_dummies(combined, columns=['Embarked'], dtype=int)
one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
embarked_encoded = one_hot_encoder.fit_transform(train_df['Embarked'].values.reshape(-1, 1))
embarked_encoded = pd.DataFrame(embarked_encoded, columns=one_hot_encoder.get_feature_names_out(['Embarked']))
train_df = pd.concat([train_df, embarked_encoded], axis=1)
train_df = train_df.drop(columns='Embarked')

# Split the dataset into features and target
y = train_df['Survived']
X = train_df.drop(columns='Survived')
X_test = test_df
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_numeric = X_train[NUMERIC_COLUMNS]
X_val_numeric = X_val[NUMERIC_COLUMNS]
X_test_numeric = X_test[NUMERIC_COLUMNS]
X_train[NUMERIC_COLUMNS] = scaler.fit_transform(X_train_numeric)
X_val[NUMERIC_COLUMNS] = scaler.transform(X_val_numeric)
X_test[NUMERIC_COLUMNS] = scaler.transform(X_test_numeric)

# Train the model -------------------------------------------------------
# Create a logistic regression model (numpy implementation)
model = LogisticRegression(
    lr=0.01,
    num_iter=100000,
    fit_intercept=True,
    tol=1e-6,
    verbose=False
)

# Train the model
start = time.time()
model.fit(X_train, y_train)
print(f'Training time: {time.time() - start:.2f} seconds')

# Evaluate the model ----------------------------------------------------
# Make predictions
y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)
# y_pred_test = model.predict(X_test.to_numpy())

# Calculate the accuracy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# train_accuracy = (y_pred_train == y_train).mean()
# val_accuracy = (y_pred_val == y_val).mean()
train_accuracy = accuracy_score(y_train, y_pred_train)
val_accuracy = accuracy_score(y_val, y_pred_val)

train_precision = precision_score(y_train, y_pred_train)
val_precision = precision_score(y_val, y_pred_val)

train_recall = recall_score(y_train, y_pred_train)
val_recall = recall_score(y_val, y_pred_val)

train_f1 = f1_score(y_train, y_pred_train)
val_f1 = f1_score(y_val, y_pred_val)

train_confusion_matrix = confusion_matrix(y_train, y_pred_train)
val_confusion_matrix = confusion_matrix(y_val, y_pred_val)

print(f'Training accuracy: {train_accuracy:.2f}')
print(f'Validation accuracy: {val_accuracy:.2f}')

print(f'Training precision: {train_precision:.2f}')
print(f'Validation precision: {val_precision:.2f}')

print(f'Training recall: {train_recall:.2f}')
print(f'Validation recall: {val_recall:.2f}')

print(f'Training f1: {train_f1:.2f}')
print(f'Validation f1: {val_f1:.2f}')

print('Training confusion matrix:')
print(train_confusion_matrix)

print('Validation confusion matrix:')
print(val_confusion_matrix)

# Save the predictions to a CSV file
# ids = pd.read_csv(DATASET_ROOT / 'test.csv')['PassengerId']
# output_df = pd.DataFrame({'PassengerId': ids, 'Survived': y_pred_test.astype(int)})
# output_df.to_csv(ROOT / 'predictions.csv', index=False)
# print('Predictions saved to predictions.csv')


from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

# Create a logistic regression model (sklearn implementation)
sklearn_model = SklearnLogisticRegression(
    C=1.0,
    max_iter=100000,
)

start = time.time()
# Train the model
sklearn_model.fit(X_train, y_train)
print(f'Sklearn Training time: {time.time() - start:.2f} seconds')

# Evaluate the model
train_accuracy = sklearn_model.score(X_train, y_train)
val_accuracy = sklearn_model.score(X_val, y_val)

print(f'Sklearn Training accuracy: {train_accuracy:.2f}')
print(f'Sklearn Validation accuracy: {val_accuracy:.2f}')
