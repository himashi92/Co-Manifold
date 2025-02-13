import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams.update({'font.size': 8})
# Data for 20% labeled
data_20 = {
    "Co-Manifolds": [
        0.90772, 0.9084, 0.92597, 0.91908, 0.90547, 0.89653, 0.92842, 0.90355, 0.94502,
        0.93569, 0.87591, 0.90654, 0.93754, 0.93845, 0.92178, 0.93018, 0.88182, 0.90971,
        0.93267, 0.89827
    ],
    "Co-BioNet": [
        0.88607, 0.92738, 0.9273, 0.91674, 0.91499, 0.87415, 0.92094, 0.89346, 0.9388,
        0.935, 0.86867, 0.9099, 0.93562, 0.93404, 0.92257, 0.92924, 0.88119, 0.91028,
        0.93546, 0.88949
    ],
    "MC-Net+": [
        0.899, 0.90947, 0.91281, 0.91582, 0.92722, 0.879, 0.93599, 0.88843, 0.93853,
        0.93194, 0.86017, 0.88839, 0.91786, 0.92531, 0.91328, 0.92818, 0.88679, 0.90144,
        0.94167, 0.90906
    ],
}

# Data for 10% labeled
data_10 = {
    "Co-Manifolds": [
        0.90808, 0.90596, 0.91066, 0.90535, 0.88961, 0.88251, 0.92064, 0.88835, 0.93703,
        0.9358, 0.86115, 0.88487, 0.9213, 0.91826, 0.92921, 0.92569, 0.84034, 0.91065,
        0.8663, 0.88621
    ],
    "Co-BioNet": [
        0.88095, 0.90254, 0.90882, 0.90511, 0.89416, 0.85331, 0.9107, 0.87913, 0.93267,
        0.93442, 0.82643, 0.88419, 0.9237, 0.91638, 0.91667, 0.92087, 0.78864, 0.89334,
        0.89705, 0.87154
    ],
    "MC-Net+": [
        0.88741, 0.89492, 0.88934, 0.91146, 0.8951, 0.83607, 0.92043, 0.86241, 0.91691,
        0.92998, 0.85896, 0.88218, 0.90873, 0.92347, 0.92517, 0.91007, 0.80747, 0.89295,
        0.84606, 0.88038
    ],
}

# Convert to DataFrames
df_20 = pd.DataFrame(data_20).melt(var_name="Method", value_name="Score")
df_10 = pd.DataFrame(data_10).melt(var_name="Method", value_name="Score")

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(5, 4), sharey=True)

# Plot for 20% labeled data
sns.boxplot(
    x="Method", y="Score", data=df_20, palette="pastel", width=0.2, showmeans=True,
    meanprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "black"},
    ax=axes[0]
)
axes[0].set_title("20% Labelled Data", fontsize=8)
axes[0].set_ylabel("Dice Similarity Coefficient", fontsize=8)
axes[0].set_xticklabels(axes[0].get_xticklabels(), fontsize=7)
axes[0].set_xlabel("")

# Plot for 10% labeled data
sns.boxplot(
    x="Method", y="Score", data=df_10, palette="pastel", width=0.2, showmeans=True,
    meanprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "black"},
    ax=axes[1]
)
axes[1].set_title("10% Labelled Data", fontsize=8)
axes[1].set_ylabel("Dice Similarity Coefficient", fontsize=8)
axes[1].set_xticklabels(axes[1].get_xticklabels(), fontsize=7)
axes[1].set_xlabel("")

# Adjust layout and save
plt.tight_layout()
plt.savefig("combined_boxplots.png", dpi=300)
plt.show()
