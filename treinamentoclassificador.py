import os
import pandas as pd
from PIL import Image
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tqdm import tqdm  # <-- progress bar

# --- Parameters ---
image_folder = '/home/vilhenajefferson/intcomp/classificador_medico/ISIC2018_Task3_Test_Input'
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

# --- Train decision tree ---
#clf = DecisionTreeClassifier(random_state=42, max_depth=3)

# --- Train Naive Bayes ---
clf = GaussianNB()

# Create a base Decision Tree (stump is default: depth=1)
#base_estimator = DecisionTreeClassifier(max_depth=1)
# Create AdaBoost using the base tree
#clf = AdaBoostClassifier(estimator=base_estimator, n_estimators=10, random_state=42)

clf.fit(X_train, y_train)

# --- Evaluate ---
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))