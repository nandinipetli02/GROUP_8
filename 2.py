from pathlib import Path
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, silhouette_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

# --- Project Paths ---
project_root = Path.cwd()
dataset_dir = project_root / "Dataset"
candidate = dataset_dir / "DERM7PT"
base_dir = candidate if candidate.exists() and candidate.is_dir() else dataset_dir

IMG_ROOT  = base_dir / "images"
meta_folder = base_dir / "meta"

# --- Metadata CSV Selection ---
if meta_folder.exists() and meta_folder.is_dir():
    meta_csvs = list(meta_folder.glob("*.csv"))
    if not meta_csvs:
        raise FileNotFoundError(f"No .csv found in {meta_folder}")
    META_PATH = meta_csvs[0]
elif (base_dir / "meta.csv").exists():
    META_PATH = base_dir / "meta.csv"
else:
    raise FileNotFoundError(f"Cannot find metadata CSV in {base_dir}")

# --- Load Data ---
df = pd.read_csv(META_PATH)

# --- Drop 'case num' if exists ---
for col in df.columns:
    if 'case' in col.lower() and 'num' in col.lower():
        df = df.drop(columns=[col])
        print(f"Dropped column: {col}")

# --- Image ID Column ---
df['image_id'] = df['derm'].apply(lambda fn: Path(fn).stem)

# --- Filter Rows ---
required_columns = ['diagnosis', 'pigment_network', 'streaks', 'pigmentation', 
                    'regression_structures', 'dots_and_globules', 'blue_whitish_veil',
                    'vascular_structures', 'seven_point_score']
df_clean = df.dropna(subset=required_columns)

# --- Remove rare diagnosis classes ---
diagnosis_counts = df_clean['diagnosis'].value_counts()
valid_diagnoses = diagnosis_counts[diagnosis_counts >= 2].index
df_clean = df_clean[df_clean['diagnosis'].isin(valid_diagnoses)]

# --- Categorical Encoding ---
categorical_features = ['pigment_network', 'streaks', 'pigmentation', 'regression_structures',
                        'dots_and_globules', 'blue_whitish_veil', 'vascular_structures']
le_dict = {}
for col in categorical_features:
    le = LabelEncoder()
    df_clean[col + '_encoded'] = le.fit_transform(df_clean[col].astype(str))
    le_dict[col] = le

le_diagnosis = LabelEncoder()
df_clean['diagnosis_encoded'] = le_diagnosis.fit_transform(df_clean['diagnosis'])

# --- Feature Matrix ---
feature_cols = [col + '_encoded' for col in categorical_features] + ['seven_point_score']
X = df_clean[feature_cols].values
y = df_clean['diagnosis_encoded'].values

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Train Models and Evaluate ---
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {acc:.3f}")
    if name != "Logistic Regression":
        print(classification_report(y_test, y_pred, target_names=le_diagnosis.classes_, zero_division=0))
    results[name] = acc

# --- Feature Importance (Random Forest) ---
rf_model = models['Random Forest']
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nFeature Importance (Random Forest):")
print(feature_importance)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance')
plt.title('Feature Importance for Skin Lesion Diagnosis')
plt.tight_layout()
plt.show()

# --- Confusion Matrix (Random Forest) ---
cm_rf = confusion_matrix(y_test, rf_model.predict(X_test))
plt.figure(figsize=(10, 8))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le_diagnosis.classes_, 
            yticklabels=le_diagnosis.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Random Forest')
plt.tight_layout()
plt.show()

# --- PCA Visualization ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Diagnosis')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title('PCA of Skin Lesion Features')
plt.tight_layout()
plt.show()

# --- K-Means Clustering ---
kmeans = KMeans(n_clusters=len(np.unique(y)), random_state=42)
clusters = kmeans.fit_predict(X)
silhouette_avg = silhouette_score(X, clusters)
print(f"\nSilhouette Score for K-Means: {silhouette_avg:.3f}")

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Cluster')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title('K-Means Clustering of Skin Lesions')
plt.tight_layout()
plt.show()

# --- DBSCAN Clustering ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
dbscan = DBSCAN(eps=3, min_samples=5)
dbscan_clusters = dbscan.fit_predict(X_scaled)
n_clusters = len(set(dbscan_clusters)) - (1 if -1 in dbscan_clusters else 0)
n_noise = list(dbscan_clusters).count(-1)
print(f"\nDBSCAN Results:")
print(f"Estimated number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_clusters, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Cluster')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title('DBSCAN Clustering of Skin Lesions')
plt.tight_layout()
plt.show()

# --- Model Performance Summary ---
results_summary = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': list(results.values())
})
print("\nModel Performance Summary:")
print(results_summary)

plt.figure(figsize=(8, 5))
plt.bar(results_summary['Model'], results_summary['Accuracy'])
plt.ylabel('Accuracy')
plt.title('Model Comparison')
plt.ylim([0, 1])
for i, v in enumerate(results_summary['Accuracy']):
    plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
plt.tight_layout()
plt.show()