from pathlib import Path

project_root = Path.cwd()
dataset_dir = project_root / "Dataset"

print("Project root:", project_root)
print("Dataset folder exists:", dataset_dir.exists())
print("\nContents of Dataset/:")
for p in sorted(dataset_dir.iterdir()):
    typ = "DIR " if p.is_dir() else "FILE"
    print(f"  {typ} {p.name}")

candidate = dataset_dir / "DERM7PT"
if candidate.exists() and candidate.is_dir():
    base_dir = candidate
    print("\nUsing subfolder DERM7PT as base_dir")
else:
    base_dir = dataset_dir
    print("\nUsing Dataset as base_dir")

IMG_ROOT  = base_dir / "images"
meta_folder = base_dir / "meta"

if meta_folder.exists() and meta_folder.is_dir():
    meta_csvs = list(meta_folder.glob("*.csv"))
    if not meta_csvs:
        raise FileNotFoundError(f"No .csv found in {meta_folder}")
    META_PATH = meta_csvs[0]
elif (base_dir / "metadata.csv").exists():
    META_PATH = base_dir / "metadata.csv"
else:
    raise FileNotFoundError(f"Cannot find metadata CSV in {base_dir}")

print(f"\nIMG_ROOT -> {IMG_ROOT}")
print(f"META_PATH -> {META_PATH}")

import pandas as pd
df = pd.read_csv(META_PATH)

print("Metadata shape:", df.shape)
print(df.head())

print("Columns:", list(df.columns))
print(df.head())

from pathlib import Path

df['image_id'] = df['derm'].apply(lambda fn: Path(fn).stem)

print("New columns:", list(df.columns))
print(df[['derm', 'image_id']].head())

subdirs    = [p for p in IMG_ROOT.iterdir() if p.is_dir()]
all_images = list(IMG_ROOT.rglob("*.jpg"))

print(f"\n{len(subdirs)} subdirectories under images/")
print(f"{len(all_images)} total JPEGs found")

img_map = {p.stem: p for p in all_images}
print("Sample entries from img_map:", list(img_map.items())[:5])

import random
import matplotlib.pyplot as plt

sample_ids = random.sample(list(df['image_id']), 10)

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for ax, img_id in zip(axes.flatten(), sample_ids):
    img_path = img_map.get(img_id)
    if img_path and img_path.exists():
        ax.imshow(plt.imread(img_path))
    else:
        ax.text(0.5, 0.5, f"{img_id}\nnot found", ha="center", va="center")
    ax.axis("off")

plt.suptitle("Random Sample of Dermoscopic Images")
plt.tight_layout()
plt.show()

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns

df_clean = df.dropna(subset=['diagnosis', 'pigment_network', 'streaks', 'pigmentation', 
                               'regression_structures', 'dots_and_globules', 'blue_whitish_veil',
                               'vascular_structures', 'seven_point_score'])

print(f"\nClass distribution before filtering:")
print(df_clean['diagnosis'].value_counts())

diagnosis_counts = df_clean['diagnosis'].value_counts()
valid_diagnoses = diagnosis_counts[diagnosis_counts >= 2].index
df_clean = df_clean[df_clean['diagnosis'].isin(valid_diagnoses)]

print(f"\nClass distribution after filtering (min 2 samples):")
print(df_clean['diagnosis'].value_counts())
print(f"Dataset size after filtering: {len(df_clean)}")

le_dict = {}
categorical_features = ['pigment_network', 'streaks', 'pigmentation', 'regression_structures',
                        'dots_and_globules', 'blue_whitish_veil', 'vascular_structures']

for col in categorical_features:
    le = LabelEncoder()
    df_clean[col + '_encoded'] = le.fit_transform(df_clean[col].astype(str))
    le_dict[col] = le

le_diagnosis = LabelEncoder()
df_clean['diagnosis_encoded'] = le_diagnosis.fit_transform(df_clean['diagnosis'])

feature_cols = [col + '_encoded' for col in categorical_features] + ['seven_point_score']
X = df_clean[feature_cols].values
y = df_clean['diagnosis_encoded'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("\nRandom Forest Classifier Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, target_names=le_diagnosis.classes_, zero_division=0))

gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)

print("\nGradient Boosting Classifier Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_gb):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_gb, target_names=le_diagnosis.classes_, zero_division=0))

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("\nLogistic Regression Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.3f}")

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

cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le_diagnosis.classes_, 
            yticklabels=le_diagnosis.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Random Forest')
plt.tight_layout()
plt.show()

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

kmeans = KMeans(n_clusters=len(np.unique(y)), random_state=42)
clusters = kmeans.fit_predict(X)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Cluster')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title('K-Means Clustering of Skin Lesions')
plt.tight_layout()
plt.show()

from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(X, clusters)
print(f"\nSilhouette Score for K-Means: {silhouette_avg:.3f}")

cluster_diagnosis = pd.DataFrame({
    'cluster': clusters,
    'diagnosis': le_diagnosis.inverse_transform(y)
})
print("\nCluster vs Diagnosis Distribution:")
print(pd.crosstab(cluster_diagnosis['cluster'], cluster_diagnosis['diagnosis']))

from sklearn.preprocessing import StandardScaler
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

results_summary = pd.DataFrame({
    'Model': ['Random Forest', 'Gradient Boosting', 'Logistic Regression'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_gb),
        accuracy_score(y_test, y_pred_lr)
    ]
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
