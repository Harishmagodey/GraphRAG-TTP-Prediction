import matplotlib.pyplot as plt
import numpy as np

models = [
    "Markov\nChain",
    "TransE",
    "GAT/\nR-GCN",
    "Temporal\nGNN",
    "Proposed\nET-CTKG"
]

hits1 = [0.602, 0.531, 0.661, 0.550, 0.742]
hits3 = [0.786, 0.742, 0.808, 0.752, 0.891]
mrr    = [0.711, 0.655, 0.751, 0.667, 0.823]
f1     = [0.292, 0.275, 0.325, 0.279, 0.391]

x = np.arange(len(models))
w = 0.18

plt.figure(figsize=(10,6))

plt.bar(x-1.5*w, hits1, width=w, label="Hits@1")
plt.bar(x-0.5*w, hits3, width=w, label="Hits@3")
plt.bar(x+0.5*w, mrr, width=w, label="MRR")
plt.bar(x+1.5*w, f1, width=w, label="F1")

plt.xticks(x, models)
plt.ylabel("Score")
plt.xlabel("Models")
plt.title("Comparison of Proposed ET-CTKG with Baseline Models")
plt.legend()

plt.tight_layout()
plt.savefig("graphs/Fig4_Proposed_System_Comparison.png", dpi=300)
plt.show()

print("Figure saved successfully.")