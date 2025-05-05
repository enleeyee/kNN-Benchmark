import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("results/benchmark.csv")
# Parse method and dataset from label
df["method"] = df["label"].str.split("_").str[-1]  # EUCLIDEAN, MANHATTAN, COSINE
df["dataset"] = df["label"].str.split("_").str[0]  # Small, Medium, Large

# Sort datasets for consistent ordering
df["dataset"] = pd.Categorical(df["dataset"], categories=["Small", "Medium", "Large"], ordered=True)
df = df.sort_values("dataset")

# Create grouped bar chart: dataset on x-axis, bars for each method
labels = df["dataset"].unique()
methods = df["method"].unique()

x = range(len(labels))
width = 0.35  # Width of each bar

# Plot
plt.figure(figsize=(8, 6))
for i, method in enumerate(methods):
    times = df[df["method"] == method].set_index("dataset").loc[labels]["time_sec"]
    plt.bar([p + i * width for p in x], times, width, label=method)

# Formatting
plt.xlabel("Dataset Size Category")
plt.ylabel("Time (seconds)")
plt.title("kNN Runtime Comparison Across Dataset Sizes")
plt.xticks([p + width / 2 for p in x], labels)
plt.legend()
plt.tight_layout()
plt.savefig("results/runtime_all_datasets.png")
print("Saved: results/runtime_all_datasets.png")
