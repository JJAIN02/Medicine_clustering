import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
def load_data(file_path):
    return pd.read_csv(file_path)

# Step 2: Data Cleaning and Standardization
def clean_data(df):
    # Fill missing values
    df.fillna({
        'name': 'unknown', 
        'manufacturer': 'unknown', 
        'salts': 'unknown', 
        'packaging_form': 'unknown',
        'retail_price': df['retail_price'].median(),
        'discounted_price': df['discounted_price'].median()
    }, inplace=True)

    # Normalize column names and format dosage
    df['name'] = df['name'].str.strip().str.lower()
    df['manufacturer'] = df['manufacturer'].str.strip().str.lower()
    df['salts'] = df['salts'].str.strip().str.lower()
    df['packaging_form'] = df['packaging_form'].str.strip().str.lower()

    return df

# Step 3: Define similarity criteria 
def prepare_features(df):
    df['salts'] = df['salts'].str.replace(' ', '').str.replace(r'\([^)]*\)', '', regex=True)
    features = pd.get_dummies(df[['packaging_form', 'salts']])

    # Normalize prices for additional clustering criteria
    features['retail_price'] = df['retail_price']
    features['discounted_price'] = df['discounted_price']

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    return scaled_features

# Step 4: Clustering using DBSCAN
def cluster_medicines(features):
    # Remove rows with NaN values in the feature set
    if np.isnan(features).any():
        print("NaN values detected in features. Dropping rows with NaN values.")
        features = features[~np.isnan(features).any(axis=1)]

    clustering = DBSCAN(eps=0.5, min_samples=5, metric='euclidean').fit(features)
    return clustering.labels_


# Step 5: Save results and visualize
def save_results(df, labels):
    df['Cluster'] = labels
    df.to_csv('clustered_medicines.csv', index=False)
    print(f"Results saved to clustered_medicines.csv")

    # Visualize clusters
    plt.figure(figsize=(16, 8))
    cluster_counts = pd.Series(labels).value_counts().sort_index()  # Get cluster counts
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette="viridis")
    plt.title('Cluster Distribution', fontsize=16)
    plt.xlabel('Cluster Label', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12)
    for i, count in enumerate(cluster_counts.values):
        plt.text(i, count, str(count), ha='center', va='bottom', fontsize=10, color='black')
    plt.tight_layout()
    plt.show()
    print(f"Number of clusters: {cluster_counts}")

    #calculate unique medicines
    unique_medicines = df['name'].nunique()
    print(f"Number of unique medicines: {unique_medicines}\t")


if __name__ == "__main__":
    # Load raw dataset
    file_path = r"C:\Users\biomi\OneDrive\Desktop\medicine_clustering\medicine_dataset.csv"  # Replace with your dataset path
    df = load_data(file_path)

    # Clean and standardize
    cleaned_df = clean_data(df)

    # Prepare features for clustering
    features = prepare_features(cleaned_df)

    # Perform clustering
    labels = cluster_medicines(features)

    # Save and visualize results
    save_results(cleaned_df, labels)
