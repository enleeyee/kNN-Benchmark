import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

df = pd.read_csv("results/visual_knn.csv", names=["type", "x", "y", "query_id"], keep_default_na=False)

ref = df[df["type"] == "ref"]
queries = df[df["type"] == "query"]
neighbors = df[df["type"] == "neighbor"]

fig, ax = plt.subplots(figsize=(8, 6))

def animate(i):
    ax.clear()
    ax.set_title(f"k-NN Visualization for Query {i}")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    ax.scatter(ref["x"], ref["y"], color="gray", label="Reference Points")
    q = queries.iloc[i]
    ax.scatter(q["x"], q["y"], color="red", label="Query Point", s=100)

    n_i = neighbors[neighbors["query_id"] == str(i)]
    ax.scatter(n_i["x"], n_i["y"], color="blue", marker="x", s=80, label="k-NN")

    ax.legend(loc="upper right")

anim = animation.FuncAnimation(fig, animate, frames=len(queries), interval=1000, repeat=False)
anim.save("results/knn_visual.gif", writer="pillow", fps=1)

plt.tight_layout()
plt.show()
