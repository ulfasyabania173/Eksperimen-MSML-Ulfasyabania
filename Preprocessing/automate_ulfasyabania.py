import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_ionosphere(input_csv="ionosphere_raw.csv", output_csv="ionosphere_preprocessing.csv", test_size=0.2, random_state=42):
    """
    Melakukan preprocessing otomatis pada dataset Ionosphere:
    - Menghapus duplikat
    - Encoding target ke numerik
    - Standarisasi fitur numerik
    - Menghapus kolom konstan
    - Menyimpan hasil ke CSV
    - Mengembalikan X_train, X_test, y_train, y_test
    """
    # Load data
    df = pd.read_csv(input_csv)

    # Drop duplicate rows
    df = df.drop_duplicates()

    # Encode target
    if 'Class' in df.columns:
        df['ClassNum'] = df['Class'].map({'g': 1, 'b': 0})
    else:
        raise ValueError("Kolom target 'Class' tidak ditemukan.")

    # Drop kolom konstan (misal: Attribute2)
    nunique = df.nunique()
    const_cols = nunique[nunique == 1].index.tolist()
    df = df.drop(columns=const_cols)

    # Pilih kolom fitur (kecuali target)
    feature_cols = [col for col in df.columns if col.startswith('Attribute')]

    # Standarisasi fitur
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Simpan hasil preprocessing
    df.to_csv(output_csv, index=False)

    # Split data
    X = df[feature_cols]
    y = df['ClassNum']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_ionosphere()
    print(f"Preprocessing selesai. Data siap digunakan.\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")
