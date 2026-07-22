import pandas as pd
import matplotlib.pyplot as plt

# Read comparison results
df = pd.read_csv("Experiment3_Comparison.csv")

# Remove LLM-only row if metrics are empty
df = df[df["Hits@1"].notna()]

# Create graph
plt.figure(figsize=(8,5))
plt.bar(df["Model"], df["Hits@1"], color=["blue","green","orange","red"])

plt.title("Hits@1 Comparison")
plt.xlabel("Models")
plt.ylabel("Hits@1")
plt.xticks(rotation=15)

plt.tight_layout()

# Save graph
plt.savefig("graphs/Fig2_HitsAt1_Comparison.png", dpi=300)

plt.show()

print("Graph saved as graphs/Fig2_HitsAt1_Comparison.png")