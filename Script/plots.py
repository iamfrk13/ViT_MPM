#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt

# ====== Load Excel File ======
file_path = "E:/paper_2_accuracy_plots.xlsx"
df = pd.read_excel(file_path)

# ====== Extract Epochs ======
epochs = df["epochs"]

# ====== 1️⃣ CNN vs ViT - Scratch ======
plt.figure(figsize=(8, 5))
plt.plot(epochs, df["cnn_scratch"], 'r--o', label='CNN Scratch')
plt.plot(epochs, df["vit_scratch"], 'b-o', label='ViT Scratch')
plt.title("CNN vs ViT - Scratch")
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("E:/ViT_CNN_Scratch.jpg", dpi=300)
plt.show()
plt.close()

# ====== 2️⃣ CNN vs ViT - Augmented ======
plt.figure(figsize=(8, 5))
plt.plot(epochs, df["cnn_augmented"], 'r--o', label='CNN Augmented')
plt.plot(epochs, df["vit_augmented"], 'b-o', label='ViT Augmented')
plt.title("CNN vs ViT - Augmented")
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("E:/ViT_CNN_Augmented.jpg", dpi=300)
plt.show()
plt.close()

# ====== 3️⃣ CNN vs ViT - Finetuned (Head) ======
plt.figure(figsize=(8, 5))
plt.plot(epochs, df["cnn_finetuned_head"], 'r--o', label='CNN Finetuned Head')
plt.plot(epochs, df["vit_finetuned_head"], 'b-o', label='ViT Finetuned Head')
plt.title("CNN vs ViT - Finetuned (Head)")
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("E:/ViT_CNN_Finetuned_Head.jpg", dpi=300)
plt.show()
plt.close()

# ====== 4️⃣ CNN vs ViT - Core Finetuned (Final) ======
plt.figure(figsize=(8, 5))
plt.plot(epochs, df["cnn_finetuned_final"], 'r--o', label='CNN Finetuned Final')
plt.plot(epochs, df["vit_finetuned_final"], 'b-o', label='ViT Finetuned Final')
plt.title("CNN vs ViT - Finetuned (Final)")
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.ylim(0.5, 1.0)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("E:/ViT_CNN_Finetuned_Final.jpg", dpi=300)
plt.show()
plt.close()

# ====== 5️⃣ Combined Plot: All Comparisons ======
plt.figure(figsize=(10, 6))

plt.plot(epochs, df["cnn_scratch"], 'r--', label='CNN Scratch')
plt.plot(epochs, df["vit_scratch"], 'r-', label='ViT Scratch')

plt.plot(epochs, df["cnn_augmented"], 'b--', label='CNN Augmented')
plt.plot(epochs, df["vit_augmented"], 'b-', label='ViT Augmented')

plt.plot(epochs, df["cnn_finetuned_head"], 'g--', label='CNN Finetuned Head')
plt.plot(epochs, df["vit_finetuned_head"], 'g-', label='ViT Finetuned Head')

plt.plot(epochs, df["cnn_finetuned_final"], 'm--', label='CNN Finetuned Final')
plt.plot(epochs, df["vit_finetuned_final"], 'm-', label='ViT Finetuned Final')

plt.title("CNN vs ViT - Validation Accuracy Comparison (All Experiments)")
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("E:/ViT_CNN_Validation_Comparison.jpg", dpi=300)
plt.show()
plt.close()


# In[6]:


import pandas as pd
import matplotlib.pyplot as plt

# ====== Load Excel File ======
file_path = "E:/paper_1_accuracy_plots.xlsx"
df = pd.read_excel(file_path)

# ====== Extract Epochs ======
epochs = df["epochs"]

# ====== 1️⃣ Scratch ======
plt.figure(figsize=(8, 5))
plt.plot(epochs, df["msi_scratch"], 'r--o', label='MSI Scratch')
plt.plot(epochs, df["msi_br_scratch"], 'b-o', label='MSI+BR Scratch')
plt.plot(epochs, df["msi_pca_mnf_scratch"], 'g-.s', label='MSI+PCA+MNF Scratch')
plt.title("Convnet_Scratch")
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("E:/CNN_Scratch.jpg", dpi=300)
plt.show()
plt.close()

# ====== 2️⃣ Augmented ======
plt.figure(figsize=(8, 5))
plt.plot(epochs, df["msi_aug"], 'r--o', label='MSI Augmented')
plt.plot(epochs, df["msi_br_aug"], 'b-o', label='MSI+BR Augmented')
plt.plot(epochs, df["msi_pca_mnf_aug"], 'g-.s', label='MSI+PCA+MNF Augmented')
plt.title("Convnet_Augmented")
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("E:/CNN_Augmented.jpg", dpi=300)
plt.show()
plt.close()

# ====== 3️⃣ Finetuned (Head) ======
plt.figure(figsize=(8, 5))
plt.plot(epochs, df["msi_finetuned_head"], 'r--o', label='MSI Finetuned Head')
plt.plot(epochs, df["msi_br_finetuned_head"], 'b-o', label='MSI+BR Finetuned Head')
plt.plot(epochs, df["msi_pca_mnf_finetuned_head"], 'g-.s', label='MSI+PCA+MNF Finetuned Head')
plt.title("Convnet_Finetuned_Head")
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("E:/CNN_Finetuned_Head.jpg", dpi=300)
plt.show()
plt.close()

# ====== 4️⃣ Finetuned (Final) ======
plt.figure(figsize=(8, 5))
plt.plot(epochs, df["msi_finetuned_final"], 'r--o', label='MSI Finetuned Final')
plt.plot(epochs, df["msi_br_finetuned_final"], 'b-o', label='MSI+BR Finetuned Final')
plt.plot(epochs, df["msi_pca_mnf_finetuned_final"], 'g-.s', label='MSI+PCA+MNF Finetuned Final')
plt.title("Convnet_Finetuned_Final")
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.ylim(0.5, 1.0)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("E:/CNN_Finetuned_Final.jpg", dpi=300)
plt.show()
plt.close()


# In[58]:


import pandas as pd
import matplotlib.pyplot as plt

# Data extracted from your table
data = {
    'Configuration': ['Scratch', 'Scratch_Augmented', 'Finetuned_Head', 'Finetuned_Final'],
    'Accuracy_MSI': [0.722, 0.785, 0.893, 0.937],
    'Accuracy_MSI_BR': [0.859, 0.863, 0.873, 0.898],
    'Accuracy_MSI_PCA_MNF': [0.824, 0.893, 0.888, 0.898],
    'Precision_MSI': [0.732, 0.789, 0.891, 0.936],
    'Precision_MSI_BR': [0.855, 0.864, 0.871, 0.901],
    'Precision_MSI_PCA_MNF': [0.83, 0.894, 0.889, 0.903],
    'Recall_MSI': [0.75, 0.802, 0.896, 0.936],
    'Recall_MSI_BR': [0.867, 0.866, 0.878, 0.896],
    'Recall_MSI_PCA_MNF': [0.833, 0.895, 0.89, 0.897],
    'F1_MSI': [0.72, 0.789, 0.893, 0.936],
    'F1_MSI_BR': [0.858, 0.864, 0.874, 0.898],
    'F1_MSI_PCA_MNF': [0.829, 0.894, 0.889, 0.899]
}

# Create DataFrame
df = pd.DataFrame(data)

# Metrics to plot
metrics = ['Accuracy', 'Precision', 'Recall', 'F1']

# Create 2x2 figure layout
fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharey=False)
axes = axes.flatten()  # flatten for easy iteration

bar_width = 0.18
colors = ['#4C72B0', '#55A868', '#C44E52']  # color palette

for i, metric in enumerate(metrics):
    ax = axes[i]
    x = range(len(df))

    # Bar positions (tight grouping)
    pos_left  = [p - bar_width*0.7 for p in x]
    pos_mid   = x
    pos_right = [p + bar_width*0.7 for p in x]

    # Bars
    ax.bar(pos_left,  df[f'{metric}_MSI'],           width=bar_width, color=colors[0], label='MSI')
    ax.bar(pos_mid,   df[f'{metric}_MSI_BR'],       width=bar_width, color=colors[1], label='MSI + BR')
    ax.bar(pos_right, df[f'{metric}_MSI_PCA_MNF'],  width=bar_width, color=colors[2], label='MSI + PCA + MNF')

    # Formatting
    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Configuration'], rotation=30, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', labelsize=10)

axes[0].set_ylabel("Performance", fontsize=11)

# Figure title
fig.suptitle("Comparison of CNN-Based Models Under Different Datasets and Configurations",
             fontsize=14, fontweight='bold', y=1.03)

# Legend at bottom
fig.legend(['MSI', 'MSI + BR', 'MSI + PCA + MNF'],
           loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05),
           frameon=False, fontsize=11)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.subplots_adjust(bottom=0.15)

# Show and save
plt.show()
fig.savefig("E:/writeup/paper_1_bar_graph.jpg", dpi=600, bbox_inches='tight')


# In[62]:


import pandas as pd
import matplotlib.pyplot as plt

# Data from your table
data = {
    'Configuration': ['Scratch', 'Scratch_Augmented', 'Finetuned_Head', 'Finetuned_Final'],
    'Accuracy_CNN': [0.722, 0.785, 0.893, 0.937],
    'Accuracy_ViT': [0.761, 0.829, 0.891, 0.935],
    'Precision_CNN': [0.732, 0.789, 0.891, 0.936],
    'Precision_ViT': [0.763, 0.828, 0.941, 0.963],
    'Recall_CNN': [0.75, 0.802, 0.896, 0.936],
    'Recall_ViT': [0.781, 0.842, 0.934, 0.951],
    'F1_CNN': [0.72, 0.789, 0.893, 0.936],
    'F1_ViT': [0.763, 0.831, 0.937, 0.956]
}

# DataFrame
df = pd.DataFrame(data)

# Metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1']

# Create 2x2 subplots (compact academic layout)
fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharey=False)
axes = axes.flatten()

# Bar settings
bar_width = 0.18
colors = ['#4C72B0', '#55A868']  # CNN = blue, ViT = green

for i, metric in enumerate(metrics):
    ax = axes[i]
    x = range(len(df))

    # Shift left & right for closer grouping
    pos_left  = [p - bar_width*0.7 for p in x]
    pos_right = [p + bar_width*0.7 for p in x]

    # Bars
    ax.bar(pos_left, df[f'{metric}_CNN'], width=bar_width, color=colors[0], label='CNN')
    ax.bar(pos_right, df[f'{metric}_ViT'], width=bar_width, color=colors[1], label='ViT')

    # Format axes
    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Configuration'], rotation=30, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', labelsize=10)

axes[0].set_ylabel("Performance", fontsize=11)

# Title
fig.suptitle("Comparison of CNN and ViT Under Different Model Configurations",
             fontsize=14, fontweight='bold', y=1.03)

# Legend positioned under the figure
fig.legend(['CNN', 'ViT'],
           loc='lower center', ncol=2,
           bbox_to_anchor=(0.5, -0.05),
           frameon=False, fontsize=11)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.subplots_adjust(bottom=0.15)

# Show and save
plt.show()
fig.savefig("E:/writeup/paper_2_bar_graph.jpg", dpi=600, bbox_inches='tight')


# In[8]:


import matplotlib.pyplot as plt

# ============================
# MANUAL PCA VALUES (from your sheet)
# ============================
explained = [43.7949, 32.3357, 18.6178, 5.2515]      # %
cumulative = [43.7949, 76.1306, 94.7485, 100.0]       # %
components = [1, 2, 3, 4]

# ============================
# SINGLE PLOT: EXPLAINED + CUMULATIVE
# ============================

plt.figure(figsize=(9, 6))

# Bar chart (explained variance)
plt.bar(components,
        explained,
        alpha=0.7)

# Line plot (cumulative variance)
plt.plot(components,
         cumulative,
         marker='o',
         linewidth=2)

# Labels and formatting
plt.xlabel("Principal Component", fontsize=12)
plt.ylabel("Variance (%)", fontsize=12)
plt.title("PCA Explained & Cumulative Variance", fontsize=14)

plt.xticks(components)
plt.grid(True, linestyle='--', linewidth=0.6)

# Add a vertical line emphasizing selection of PC1 & PC2
plt.axvline(x=2.5, color='gray', linestyle='--', linewidth=1)

# Add text about PC1+PC2
plt.text(1.05, 80,
         "PC1 + PC2 = 76.13% variance\n(Chosen Components)",
         fontsize=11)

plt.tight_layout()
plt.savefig("E:/writeup/pca_selection_plot.jpg", dpi=300)
plt.show()


# In[2]:


import matplotlib.pyplot as plt

# Data
eigen_numbers = [1, 2, 3, 4]
eigen_values = [37, 15, 2.5, 1.5]

# Plot
plt.figure(figsize=(7,5))
plt.plot(eigen_numbers, eigen_values, marker='o', linewidth=2)

# Title
plt.title("MNF")

# Axis ranges
plt.xlim(1, 4)
plt.ylim(0, 40)

# Custom ticks
plt.xticks([1, 2, 3, 4])
plt.yticks([10, 20, 30])  # Only these labels

# Labels
plt.xlabel("Eigenvalue Number")
plt.ylabel("Eigenvalue")

# Grid (optional)
plt.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()

# Save as JPG
plt.savefig("E:/writeup/mnf.jpg", format="jpg", dpi=300)

plt.show()


# In[10]:


import matplotlib.pyplot as plt

# Data
references = [
    "(Keykhay-Hosseinpoor et al., 2024)",
    "(Mahboob et al., 2024)",
    "(Clabaut et al., 2020)",
    "(Zidan et al., 2023)",
    "(Pradhan et al., 2022)",
    "(Mohamed Taha et al., 2022)",
    "Current study"
]

accuracy = [53, 80, 77, 86, 92, 67, 94]

models = [
    "DBN-RF",
    "CNN",
    "CNN",
    "CNN",
    "CNN-SHAP",
    "RF",
    "CNN-Transfer Leraning"
]

# Standard blue color
color = "tab:blue"

plt.figure(figsize=(12, 6))

# Set narrower bar width
bar_width = 0.4
bars = plt.bar(references, accuracy, color=color, width=bar_width)

# Add model names on top of each bar
for bar, model in zip(bars, models):
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 1,
        model,
        ha='center',
        va='bottom',
        fontsize=10
    )

plt.xlabel("Reference / Study")
plt.ylabel("Accuracy (%)")
plt.title("Comparison of Model Accuracies Across Studies")
plt.xticks(rotation=60, ha="right")

plt.tight_layout()

# Save as JPEG
plt.savefig("E:/writeup/model_accuracy_comparison.jpg", format="jpg", dpi=300)

plt.show()


# In[ ]:




