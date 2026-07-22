import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Read the experiment results
csv_path = Path("../Experiment3_Comparison.csv")

df = pd.read_csv(csv_path)

# Remove the LLM-only row
df = df[df["Model"] != "LLM-only"]

# Convert values to numeric
for col in ["Hits@1", "Hits@3", "MRR", "F1"]:
    df[col] = pd.to_numeric(df[col])

plt.figure(figsize=(10,6))

plt.plot(df["Model"], df["Hits@1"],
         marker='o',
         linewidth=2,
         label="Hits@1")

plt.plot(df["Model"], df["Hits@3"],
         marker='s',
         linewidth=2,
         label="Hits@3")

plt.plot(df["Model"], df["MRR"],
         marker='^',
         linewidth=2,
         label="MRR")

plt.plot(df["Model"], df["F1"],
         marker='d',
         linewidth=2,
         label="F1-score")

plt.xlabel("Models", fontsize=12)
plt.ylabel("Score", fontsize=12)

plt.title("Temporal Attack Prediction Performance Comparison",
          fontsize=14,
          fontweight='bold')

plt.ylim(0,1)

plt.grid(True, linestyle="--", alpha=0.5)

plt.legend()

plt.tight_layout()

plt.savefig("Fig3_Temporal_Attack_Prediction.png",
            dpi=600,
            bbox_inches="tight")

plt.show()

print("Figure saved successfully.")