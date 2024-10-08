import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('model_comparison_results.csv')

# Add a new column for the model type
df['Type'] = df['Model'].apply(lambda x: 'Lasso' if 'Lasso' in x else 'Ridge')

# Normalize the data
df_normalized = (df.iloc[:, 1:-1] - df.iloc[:, 1:-1].min()) / (df.iloc[:, 1:-1].max() - df.iloc[:, 1:-1].min())

# Create a list of angles for the axes
angles = [n / float(df_normalized.shape[1]) * 2 * np.pi for n in range(df_normalized.shape[1])]
angles += angles[:1]

# For each unique type, create a spider plot
for model_type in df['Type'].unique():
    df_type = df[df['Type'] == model_type]
    df_normalized_type = df_normalized.loc[df_type.index]

    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, polar=True)

    for i, row in df_normalized_type.iterrows():
        values = row.values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=df.iloc[i, 0])
        ax.fill(angles, values, 'b', alpha=0.1)

    # Add labels to the axes
    labels = df_normalized.columns.tolist()
    labels += labels[:1]  # repeat the first label at the end
    ax.set_thetagrids(np.degrees(angles), labels)
    ax.set_thetagrids(np.degrees(angles), labels)

    # Add legend and title
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title(f'Model Comparison Results - {model_type} Models')
    # Save the plot as an image
    plt.savefig(f'results_{model_type}.png')