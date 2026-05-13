import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the data from the CSV file
df = pd.read_csv('external/examples/EEG_params.csv')

# 2. Pivot the data 
# We set 'window_length' as the Y-axis (index)
# 'hop_fraction' as the X-axis (columns)
# 'PercSS' as the color value
heatmap_data = df.pivot(index="window_length", columns="hop_fraction", values="PercSS")

# 3. Create the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu")

# 4. Add labels and title
plt.title('Heatmap of PercSS')
plt.xlabel('Hop Fraction')
plt.ylabel('Window Length')

# 5. Display or save the plot
plt.savefig('eeg_heatmap.png')
plt.show()