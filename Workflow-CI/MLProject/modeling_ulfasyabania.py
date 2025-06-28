import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Aktifkan autolog MLflow
mlflow.autolog()

def main():
    # Load preprocessed data
    df = pd.read_csv('ionosphere_preprocessing.csv')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run() as run:
        # Logging informasi dataset
        mlflow.log_param('train_size', X_train.shape[0])
        mlflow.log_param('test_size', X_test.shape[0])
        mlflow.log_param('n_features', X.shape[1])

        # Model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Predict & evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric('accuracy', acc) # type: ignore
        mlflow.log_text(classification_report(y_test, y_pred), 'classification_report.txt') # type: ignore
        mlflow.sklearn.log_model(model, "model")
        print(f"Run ID: {run.info.run_id}")
        print(f"Model artifact path: mlartifacts/0/{run.info.run_id}/artifacts/model/")

        print(f'Accuracy: {acc:.4f}')
        print('Model training and logging completed. Check your MLflow UI.')

if __name__ == '__main__':
    main()
