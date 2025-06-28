from prometheus_client import start_http_server, Gauge, Counter
import time, requests

# 1) Latency
inference_latency = Gauge(
    'inference_latency_seconds',
    'Waktu round-trip setiap /invocations call')

# 2) Throughput
inference_requests = Counter(
    'inference_requests_total',
    'Total request ke /invocations')

# 3) Error rate
inference_errors = Counter(
    'inference_errors_total',
    'Total request gagal ke /invocations')

def probe():
    url = "http://127.0.0.1:5001/invocations"
    payload = {"instances": [[5.1,3.5,1.4,0.2]]}
    headers = {"Content-Type":"application/json"}

    start = time.time()
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=5)
        inference_latency.set(time.time() - start)
        inference_requests.inc()
        if r.status_code != 200:
            inference_errors.inc()
    except Exception:
        inference_errors.inc()

if __name__ == '__main__':
    start_http_server(8000)    # expose di :8000
    while True:
        probe()
        time.sleep(5)
