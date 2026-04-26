import numpy as np
import matplotlib.pyplot as plt
import csv

data = []

with open("results.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append([
            float(row["expert"]),
            float(row["kv"]),
            float(row["prefill_throughput"]),
            float(row["avg_ttft"]),
            float(row["p95_ttft"]),
            float(row["p99_ttft"]),
        ])

data = np.array(data)

x = data[:, 0]
y = data[:, 1]

metrics = {
    "Throughput": data[:, 2],
    "Avg_TTFT": data[:, 3],
    "P95_TTFT": data[:, 4],
    "P99_TTFT": data[:, 5],
}

from mpl_toolkits.mplot3d import Axes3D

for name, z in metrics.items():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_trisurf(x, y, z)

    ax.set_xlabel("Expert")
    ax.set_ylabel("KV")
    ax.set_zlabel(name)
    ax.set_title(name)

    plt.savefig(f"{name}.png")

plt.show()