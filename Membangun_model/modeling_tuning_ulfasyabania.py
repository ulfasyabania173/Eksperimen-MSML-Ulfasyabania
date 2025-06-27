
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load preprocessed data

def main():
    df = pd.read_csv('../Preprocessing/ionosphere_preprocessing.csv')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5]
    }

    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy', n_jobs=-1)

    with mlflow.start_run():
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=False)

        # Manual logging
        mlflow.log_param('best_n_estimators', getattr(best_model, 'n_estimators', None))
        mlflow.log_param('best_max_depth', getattr(best_model, 'max_depth', None))
        mlflow.log_param('best_min_samples_split', getattr(best_model, 'min_samples_split', None))
        mlflow.log_param('train_size', X_train.shape[0])
        mlflow.log_param('test_size', X_test.shape[0])
        mlflow.log_param('n_features', X.shape[1])
        mlflow.log_metric('accuracy', float(acc))
        mlflow.log_text(report, 'classification_report.txt')
        import mlflow.sklearn
        mlflow.sklearn.log_model(best_model, 'model')

        print(f'Best Params: {grid.best_params_}')
        print(f'Accuracy: {acc:.4f}')
        print('Model training and logging with hyperparameter tuning completed. Check your MLflow UI.')

if __name__ == '__main__':
    main()
