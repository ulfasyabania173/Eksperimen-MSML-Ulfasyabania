import requests
import json

def infer(instances, url="http://127.0.0.1:5001/invocations"):
    """
    Kirim data ke endpoint MLflow model serving dan kembalikan hasil prediksi.
    Args:
        instances (list of list): Data input, contoh [[5.1, 3.5, 1.4, 0.2]]
        url (str): Endpoint MLflow model serving
    Returns:
        dict/str: Hasil prediksi dari model
    """
    headers = {"Content-Type": "application/json"}
    payload = {"instances": instances}
    try:
        response = requests.post(url, data=json.dumps(payload), headers=headers, timeout=10)
        response.raise_for_status()
        try:
            return response.json()
        except Exception:
            return response.text
    except Exception as e:
        print(f"Request failed: {e}")
        return None

if __name__ == "__main__":
    # Contoh penggunaan
    sample = [[5.1, 3.5, 1.4, 0.2]]
    result = infer(sample)
    print("Prediction result:", result)