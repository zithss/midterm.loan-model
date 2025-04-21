import time
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

class LoansData:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def load_data(self):
        # Loads train and test datasets.
        self.data = pd.read_csv(self.data_path)
                
    def clean_outliers(self, column):
        mean = self.data[column].mean()
        std = self.data[column].std()
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std
        self.data[column] = np.where(self.data[column] > upper_bound, upper_bound, self.data[column])
        self.data[column] = np.where(self.data[column] < lower_bound, lower_bound, self.data[column])
        
        return self.data
    
    def ordinal_encode_education(self):
        education_order = {'High School': 1, 'Associate': 2, 'Bachelor': 3, 'Master': 4, 'Doctorate': 5}
        self.data['education_level'] = self.data['person_education'].map(education_order)

        self.data.drop(columns=['person_education'], inplace=True)


    def define_target(self, target_column):
        # Splits the data into features (X) and target (y).
        X = pd.get_dummies(self.data.drop(target_column, axis=1))
        y = self.data[target_column]
        
        # Splits the data into training and testing sets.
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)


class XGBoostModel:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.training_time = None
        self.predictions = None
        self.metrics = None

    def train(self):
        print("Training XGBoost model...")
        start_time = time.time()
        self.model.fit(self.X_train, self.y_train)
        self.training_time = time.time() - start_time
        print(f"XGBoost training time: {self.training_time:.2f} seconds")

    def predict(self):
        print("Generating predictions...")
        start_time = time.time()
        y_test_pred = self.model.predict(self.X_test)
        test_pred_time = time.time() - start_time
        self.predictions = {
            'y_test_pred': y_test_pred,
            'test_pred_time': test_pred_time
        }
        print(f"Prediction time: {test_pred_time:.2f} seconds")

    def evaluate(self):
        print("Calculating metrics...")
        accuracy = accuracy_score(self.y_test, self.predictions['y_test_pred'])
        precision = precision_score(self.y_test, self.predictions['y_test_pred'])
        recall = recall_score(self.y_test, self.predictions['y_test_pred'])
        f1 = f1_score(self.y_test, self.predictions['y_test_pred'])
        auc = roc_auc_score(self.y_test, self.predictions['y_test_pred'])
        
        self.metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
        
        print(f"Test - Accuracy: {self.metrics['accuracy']:.4f}, Precision: {self.metrics['precision']:.4f}, "
            f"Recall: {self.metrics['recall']:.4f}, F1: {self.metrics['f1']:.4f}, AUC: {self.metrics['auc']:.4f}")
    
    def print_classification_report(self):
        print("Printing classification report...")
        print(classification_report(self.y_test, self.predictions['y_test_pred']))
        
        
# Example usage
data_path = 'Dataset_A_loan.csv'

# Load data
data = LoansData(data_path)
data.load_data()

# Data preprocessing
data.clean_outliers(['person_income', 'person_age', 'loan_amnt', 'person_emp_exp'])
data.ordinal_encode_education()
data.define_target('loan_status')

# Model training and evaluation
model = XGBoostModel(data.X_train, data.y_train, data.X_test, data.y_test)
model.train()
model.predict()
model.evaluate()
model.print_classification_report()