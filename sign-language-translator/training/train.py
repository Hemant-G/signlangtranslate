import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import sys
import os

def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    csv_path = os.path.join(data_dir, "landmarks.csv")
    
    if not os.path.isfile(csv_path):
        print(f"Data file not found at {csv_path}")
        return
        
    print("Loading data...")
    df = pd.read_csv(csv_path)
    
    if df.empty:
        print("Dataset is empty. Run collect_data.py first.")
        return

    X = df.drop('label', axis=1).values
    y = df['label'].values
    
    if len(np.unique(y)) < 2:
        print("Warning: Only found ONE class in data. Scikit-Learn needs at least 2 for split/train realistically.")
        
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, 
        stratify=y if len(np.unique(y)) > 1 else None
    )
    
    print("Training Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    if len(np.unique(y)) > 1:
        print(classification_report(y_test, y_pred))
    
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "sign_classifier.joblib")
    
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
