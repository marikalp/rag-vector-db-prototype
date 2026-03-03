import time
import numpy as np

def benchmark_vectorstore(store, embeddings, metadata, queries, k=5):
    results = {}

    # Indicizzazione
    start = time.time()
    store.insert(embeddings, metadata)
    results["index_time"] = time.time() - start

    # Query
    latencies = []
    for q in queries:
        q_start = time.time()
        store.retrieve(q, k=k)
        latencies.append(time.time() - q_start)

    results["latency_mean"] = np.mean(latencies)
    results["latency_p95"] = np.percentile(latencies, 95)
    results["latency_p99"] = np.percentile(latencies, 99)

    return results
