import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Simulate 1000 consumers
np.random.seed(42)
data = pd.DataFrame({
    'income': np.random.normal(loc=70000, scale=25000, size=1000).astype(int),  # Annual income
    'ev_owned': np.random.choice([0, 1], size=1000, p=[0.7, 0.3])  # 1 = owns EV, 0 = doesn't
})

# Optional: Add EV brand for EV owners
brands = ['Tesla', 'Nissan', 'Hyundai', 'BMW', 'Ford']
data['ev_brand'] = np.where(data['ev_owned'] == 1, np.random.choice(brands, size=1000), None)
# Define bins for income segmentation
bins = [0, 40000, 100000, np.inf]
labels = ['Low Income', 'Middle Income', 'High Income']

# Create income group column
data['income_group'] = pd.cut(data['income'], bins=bins, labels=labels)
# EV ownership rate per income group
ev_by_income = data.groupby('income_group')['ev_owned'].mean().reset_index()

# Plot
sns.barplot(data=ev_by_income, x='income_group', y='ev_owned', palette='viridis')
plt.ylabel('EV Ownership Rate')
plt.title('EV Ownership by Income Group')
plt.ylim(0, 1)
plt.show()
# Only EV owners
ev_owners = data[data['ev_owned'] == 1]

# Brand count by income group
brand_dist = pd.crosstab(ev_owners['income_group'], ev_owners['ev_brand'], normalize='index')

# Plot as heatmap
sns.heatmap(brand_dist, annot=True, cmap='coolwarm')
plt.title('EV Brand Preference by Income Group')
plt.ylabel('Income Group')
plt.xlabel('EV Brand')
plt.show()
