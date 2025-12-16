import pandas as pd
import mlflow
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "telco_churn_clean.csv"

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Error: File not found at {DATA_PATH}")
        return
    
    df = pd.read_csv(DATA_PATH)
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mlflow.autolog()
    
    mlflow.set_experiment("Telco-Churn-Basic-Local")
    
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")
        
        print(acc)
        print(report)
    
if __name__ == "__main__":
    main()