import os
import pandas as pd
from PIL import Image
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm  # <-- progress bar

# --- Parameters ---
image_folder = 'classificador_medico/ISIC2018_Task3_Test_Input'
label_csv = 'classificador_medico/binary_HAM10000_metadata.csv'
image_size = (10, 10)  # Resize images to a fixed size

# Verifica se o CSV existe
print(f"Caminho do CSV: {os.path.abspath(label_csv)}")
print(f"CSV existe? {os.path.exists(label_csv)}")

# Verifica se a pasta de imagens existe
print(f"Caminho da pasta: {os.path.abspath(image_folder)}")
print(f"Pasta existe? {os.path.exists(image_folder)}")

# Lista os primeiros arquivos na pasta de imagens
if os.path.exists(image_folder):
    print(f"Arquivos na pasta: {os.listdir(image_folder)[:5]}")

# --- Load labels ---
df = pd.read_csv(label_csv)

# --- Preprocess images ---
def load_image_as_flat_array(image_path):
    with Image.open(image_path) as img:
        img = img.convert('L')  # Convert to grayscale
        img = img.resize(image_size)
        return np.array(img).flatten()

X = []
y = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
    img_path = os.path.join(image_folder, row['image_name'])
    if os.path.exists(img_path):
        features = load_image_as_flat_array(img_path)
        X.append(features)
        y.append(row['label'])

X = np.array(X)
y = LabelEncoder().fit_transform(y)


# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Exemplo de oversampling manual
df_majority = X[y == 0]
df_minority = X[y == 1]
df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
X_balanced = np.vstack([df_majority, df_minority_upsampled])
y_balanced = np.array([0]*len(df_majority) + [1]*len(df_minority_upsampled))

# --- Train decision tree ---
#clf = DecisionTreeClassifier(random_state=42, max_depth=3)

# --- Train Naive Bayes ---
clf = GaussianNB()

# Create a base Decision Tree (stump is default: depth=1)
#base_estimator = DecisionTreeClassifier(max_depth=1)
# Create AdaBoost using the base tree
#clf = AdaBoostClassifier(estimator=base_estimator, n_estimators=10, random_state=42)

clf.fit(X_train, y_train)

#salva o modelo treinado em uma pasta chamada 'trained_model'

import joblib
model_folder = 'trained_model'
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
model_path = os.path.join(model_folder, 'skin_cancer_classifier.joblib')
joblib.dump(clf, model_path)
print(f"Model saved to {model_path}")


# --- Evaluate ---
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# --- Save predictions ---
predictions = pd.DataFrame({
    'image_name': df['image_name'],
    'predicted_label': y_pred
})
output_file = 'predictions.csv'
predictions.to_csv(output_file, index=False)
print(f"Predictions saved to {output_file}")

#gerar a matriza de confusao e salvar em um arquivo de imagem em uma pasta chamada 'visualization_results'
conf_matrix = confusion_matrix(y_test, y_pred)

sns.set(style='whitegrid')
visualization_folder = 'visualization_results'
if not os.path.exists(visualization_folder):
    os.makedirs(visualization_folder)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Melanoma', 'Melanoma'], yticklabels=['No Melanoma', 'Melanoma'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig(os.path.join(visualization_folder, 'confusion_matrix.png'))
print(f"Confusion matrix saved to {os.path.join(visualization_folder, 'confusion_matrix.png')}")

#