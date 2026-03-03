import time
import numpy as np

class BenchmarkRunner:
    def __init__(self, vectorstore, embeddings, metadata, queries, k=5):
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.metadata = metadata
        self.queries = queries
        self.k = k

    def run(self):
        results = {}

        # Indicizzazione
        start = time.time()
        self.vectorstore.insert(self.embeddings, self.metadata)
        results["index_time"] = time.time() - start
        results["index_throughput"] = len(self.embeddings) / results["index_time"]

        # Query
        latencies = []
        for q in self.queries:
            q_start = time.time()
            self.vectorstore.retrieve(q, k=self.k)
            latencies.append(time.time() - q_start)

        results["latency_mean"] = float(np.mean(latencies))
        results["latency_p95"] = float(np.percentile(latencies, 95))
        results["latency_p99"] = float(np.percentile(latencies, 99))

        return results

