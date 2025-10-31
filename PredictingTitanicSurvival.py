# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras import layers, models

# ============================================================
# 2. LOAD DATA
# ============================================================
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
print(df.head())
print(df.info())

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
# Create new features
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = np.where(df['FamilySize'] == 1, 1, 0)
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(['Mlle','Ms'], 'Miss')
df['Title'] = df['Title'].replace(['Mme'], 'Mrs')
df['Title'] = df['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr','Major',
                                   'Rev','Sir','Jonkheer','Dona'], 'Rare')

# Drop unneeded columns
df.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True)

# ============================================================
# 4. HANDLE MISSING VALUES
# ============================================================
num_features = ['Age', 'Fare']
cat_features = ['Sex', 'Embarked', 'Pclass', 'Title']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ])

# ============================================================
# 5. SPLIT DATA
# ============================================================
X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ============================================================
# 6. MACHINE LEARNING MODEL PIPELINE
# ============================================================
rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
])

param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [5, 10, 15],
}

grid_search = GridSearchCV(rf_model, param_grid, cv=3, scoring='f1_macro', verbose=2)
grid_search.fit(X_train, y_train)

print(f"Best Params: {grid_search.best_params_}")
print(f"Best F1 Score: {grid_search.best_score_:.4f}")

# Evaluate
y_pred = grid_search.predict(X_test)
print("\n=== Random Forest Classification Report ===")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# ============================================================
# 7. PCA for Dimensionality Reduction (for insight)
# ============================================================
X_transformed = preprocessor.fit_transform(X)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_transformed.toarray() if hasattr(X_transformed, "toarray") else X_transformed)
pca_df = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])
pca_df['Survived'] = y.values

plt.figure(figsize=(8,6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Survived', palette='coolwarm')
plt.title('PCA Visualization of Titanic Data')
plt.show()

# ============================================================
# 8. DEEP LEARNING MODEL (TensorFlow)
# ============================================================
# Preprocess numeric and categorical data for DL
X_dl = pd.get_dummies(df.drop('Survived', axis=1))
y_dl = df['Survived']

scaler = StandardScaler()
X_dl_scaled = scaler.fit_transform(X_dl)

X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(X_dl_scaled, y_dl, test_size=0.2, random_state=42, stratify=y_dl)

# Define Neural Network
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train_dl.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train_dl, y_train_dl, epochs=30, batch_size=32, validation_split=0.2, verbose=0)

# Evaluate Deep Learning model
loss, acc = model.evaluate(X_test_dl, y_test_dl)
print(f"\n=== Deep Learning Model Accuracy: {acc:.4f} ===")

# ROC-AUC
y_pred_proba = model.predict(X_test_dl)
roc = roc_auc_score(y_test_dl, y_pred_proba)
print(f"Deep Learning ROC-AUC: {roc:.4f}")

# ============================================================
# 9. PLOT LEARNING CURVE
# ============================================================
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
