import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Open the CSV file
df = pd.read_csv('model_comparison_results.csv')

# Define the model groups
model_groups = {
    'SVC_Group': ['SVC', 'SupportVectorMachineCustom'],
    'Ridge_Group': ['Ridge', 'RidgeRegressionCustom'],
    'Lasso_Group': ['Lasso', 'LassoRegressionCustom'],
    'DecisionTree_Group': ['DecisionTreeClassifier'],
    'RandomForest_Group' : ['RandomForestClassifier','RandomForest']
}

# Normalize the data
df_normalized = (df.iloc[:, 1:] - df.iloc[:, 1:].min()) / (df.iloc[:, 1:].max() - df.iloc[:, 1:].min())

# Create a list of angles for the axes
angles = [n / float(df_normalized.shape[1]) * 2 * np.pi for n in range(df_normalized.shape[1])]
angles += angles[:1]

# For each model group, create a spider plot
for group_name, model_names in model_groups.items():
    df_group = df[df['Model'].isin(model_names)]
    df_normalized_group = df_normalized.loc[df_group.index]

    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, polar=True)

    for i, row in df_normalized_group.iterrows():
        values = row.values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=df.iloc[i, 0])
        ax.fill(angles, values, 'b', alpha=0.1)

    # Add labels to the axes
    labels = df_normalized.columns.tolist()
    labels += labels[:1]  # repeat the first label at the end
    ax.set_thetagrids(np.degrees(angles), labels)

    # Add legend and title
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title(f'Model Comparison Results - {group_name} \n ! normalized !')

    # Save the plot as an image
    print(f"Exporting results_{group_name}.png")
    plt.savefig(f'figs/results_{group_name}.png')