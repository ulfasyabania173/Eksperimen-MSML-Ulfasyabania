import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_ionosphere(input_path='ionosphere.csv', output_path='ionosphere_preprocessing.csv'):
    # Load dataset
    df = pd.read_csv(input_path)

    # Label encoding for the last column if it's categorical (e.g., 'g'/'b')
    if df.iloc[:, -1].dtype == object:
        le = LabelEncoder()
        df.iloc[:, -1] = le.fit_transform(df.iloc[:, -1])

    # Feature scaling (excluding the label column)
    scaler = StandardScaler()
    features = df.iloc[:, :-1]
    scaled_features = scaler.fit_transform(features)
    df_scaled = pd.DataFrame(scaled_features, columns=features.columns)
    # Assign label column as a Series (ensure correct dtype)
    label_col = df.columns[-1]
    df_scaled[label_col] = df[label_col].reset_index(drop=True)

    # Save preprocessed data
    df_scaled.to_csv(output_path, index=False)
    return df_scaled

if __name__ == '__main__':
    preprocess_ionosphere('ionosphere.csv', 'ionosphere_preprocessing.csv')

