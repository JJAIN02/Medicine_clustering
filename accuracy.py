import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def clustering_accuracy_script(clustered_data, ground_truth):

    # Standardize column names and values
    clustered_data['name'] = clustered_data['name'].str.strip().str.lower()
    ground_truth['name'] = ground_truth['name'].str.strip().str.lower()

    # Merge clustered data with ground truth by medicine name
    merged = pd.merge(clustered_data, ground_truth, on='name', how='inner')

    if merged.empty:
        raise ValueError("The merged DataFrame is empty. Check if 'name' values match in both datasets.")

    # Extract predicted and true labels
    y_pred = merged['Cluster']
    y_true = merged['true_label']

    # Calculate metrics
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    # Calculate purity
    contingency_matrix = pd.crosstab(y_true, y_pred)
    purity = contingency_matrix.max(axis=1).sum() / contingency_matrix.sum().sum()

    return {
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Accuracy': accuracy,
        'Purity': purity
    }

clustered_data = pd.read_csv("clustered_medicines.csv")
# Dynamically create the ground truth sample
unique_names = clustered_data['name'].unique()

# Define logical groupings (you can adjust these rules as needed)
def assign_true_label(name):
    name = name.lower()
    if "dolo" in name:
        return 0
    elif "paracetamol" in name:
        return 1
    elif "cetirizine" in name:
        return 2
    elif "ibuprofen" in name:
        return 3
    elif "aspirin" in name:
        return 4
    else:
        return -1  # Unclassified

# Generate ground truth sample
ground_truth_sample = pd.DataFrame({
    'name': unique_names,
    'true_label': [assign_true_label(name) for name in unique_names]
})

# Remove unclassified entries from the ground truth
ground_truth_sample = ground_truth_sample[ground_truth_sample['true_label'] != -1]

# Calculate clustering metrics
accuracy_metrics = clustering_accuracy_script(clustered_data, ground_truth_sample)
print(accuracy_metrics)
