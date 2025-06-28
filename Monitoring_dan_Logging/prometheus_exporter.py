from prometheus_client import start_http_server, Gauge, Counter, Summary
import time
import random

# Metrik 1: Akurasi model
model_accuracy = Gauge('model_accuracy', 'Akurasi model ML yang sedang di-deploy')

# Metrik 2: Jumlah prediksi yang dilakukan
prediction_count = Counter('prediction_count', 'Total jumlah prediksi yang dilakukan oleh model')

# Metrik 3: Waktu inferensi (detik)
inference_time = Summary('inference_time_seconds', 'Waktu inferensi model dalam detik')

def update_metrics():
    while True:
        # Simulasi update akurasi model (ganti dengan pembacaan real dari MLflow/monitoring Anda)
        accuracy = random.uniform(0.8, 1.0)
        model_accuracy.set(accuracy)

        # Simulasi jumlah prediksi bertambah
        prediction_count.inc(random.randint(1, 5))

        # Simulasi waktu inferensi
        with inference_time.time():
            time.sleep(random.uniform(0.01, 0.1))

        time.sleep(10)

if __name__ == "__main__":
    # Jalankan Prometheus exporter di port 8000
    start_http_server(8000)
    print("Prometheus exporter berjalan di http://localhost:8000/metrics")
    update_metrics()