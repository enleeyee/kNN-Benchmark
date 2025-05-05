import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("results/benchmark.csv")
df["method"] = "cuda_global"  # If not already present

# Sort labels for consistent ordering
df["label"] = pd.Categorical(df["label"], categories=["Small", "Medium", "Large"], ordered=True)
df = df.sort_values("label")

# Create grouped bar chart: label on x-axis, bars for each method
labels = df["label"].unique()
methods = df["method"].unique()

x = range(len(labels))
width = 0.35  # Width of each bar

# Plot
plt.figure(figsize=(8, 6))
for i, method in enumerate(methods):
    times = df[df["method"] == method].set_index("label").loc[labels]["time_sec"]
    plt.bar([p + i * width for p in x], times, width, label=method)

# Formatting
plt.xlabel("Dataset Size")
plt.ylabel("Time (seconds)")
plt.title("kNN Runtime Comparison Across Dataset Sizes")
plt.xticks([p + width / 2 for p in x], labels)
plt.legend()
plt.tight_layout()
plt.savefig("results/runtime_all_datasets.png")
print("Saved: results/runtime_all_datasets.png")
