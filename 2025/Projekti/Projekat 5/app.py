# Uvoz potrebnih biblioteka
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Uvoz podataka - dataseta za trening i test modela
df_train = pd.read_csv("./dataset/train.csv")
df_test = pd.read_csv("./dataset/test.csv")

# Pregled podataka
# Ispis prvih nekoliko redova, informacija o DF-u i statističkih podataka
print(df_train.head())
print(df_train.info())
print(df_train.describe(include='all'))

# Provjera da li postoje nedostajući podaci u trening i test datasetu
print(df_train.isnull().sum())
print("------------------")
print(df_test.isnull().sum())

# Drop kolona koje su irelevantne za trening modela
# U ovom slučaju uklanjamo kolone koje ne doprinose modelu, npr. Cabin - previše nedostajućih podataka
cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']

df_train.drop(columns=cols_to_drop, inplace=True)
df_test.drop(columns=cols_to_drop, inplace=True)

print(df_train.head())

# Popunjavanje nedostajućih vrijednosti u koloni 'Age' sa medijanom
age_median = df_train['Age'].median()
df_train['Age'] = df_train['Age'].fillna(age_median)

# Isto se radi i za test dataset
df_test['Age'] = df_test['Age'].fillna(age_median)

print(df_train.isnull().sum())
print("------------------")
print(df_test.isnull().sum())

# Popunjavanje nedostajuće vrijednosti u koloni "Fare" samo za test dataset, upotrebom medijana
fare_median = df_train['Fare'].median()
df_test['Fare'] = df_test['Fare'].fillna(fare_median)

# Provjera da li su sada svi nedostajući podaci popunjeni
print(df_test.isnull().sum())

# Popunjavanje nedostajućih vrijednosti u koloni 'Embarked' sa najčešćom vrijednošću (mod)
embarked_mode = df_train['Embarked'].mode()[0]
df_train['Embarked'] = df_train['Embarked'].fillna(embarked_mode)

# Isto i za test skup, koristimo mod treninga
df_test['Embarked'] = df_test['Embarked'].fillna(embarked_mode)

print(df_train["Embarked"].head())
print("-------------------")
print(df_test["Embarked"].head())

# Feature engineering
# Kreiranje nove kolone 'FamilySize' koja predstavlja veličinu porodice, uključujući putnika
df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1
print(df_train.head())
print("-------------------")
df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] + 1
print(df_test.head())

# Da li je putnik sam ili ne, ukoliko je FamilySize == 1, onda je putnik sam
df_train['IsAlone'] = 0
df_train.loc[df_train['FamilySize'] == 1, 'IsAlone'] = 1

df_test['IsAlone'] = 0
df_test.loc[df_test['FamilySize'] == 1, 'IsAlone'] = 1

print(df_train[['FamilySize', 'IsAlone']].head())
print("-------------------")
print(df_test[['FamilySize', 'IsAlone']].head())

# Mapiranje kategorijskih varijabli u numeričke, male - 0, female - 1
print(df_train["Sex"].head())
print("-------------------")

sex_map = {'male': 0, 'female': 1}
df_train['Sex'] = df_train['Sex'].map(sex_map)
df_test['Sex'] = df_test['Sex'].map(sex_map)

print(df_train["Sex"].head())
print("-------------------")
print(df_test["Sex"].head())

# One-hot encoding za kolonu 'Embarked'
df_train = pd.get_dummies(df_train, columns=['Embarked'])
df_test = pd.get_dummies(df_test, columns=['Embarked'])

print(df_train.head())
print("-------------------")
print(df_test.head())

# Pregled podataka i sortiranje kolona
df_test = df_test[df_train.columns.drop("Survived")]

print(df_train.head())
print("-------------------")
print(df_test.head())

# Podjela podataka na X (features) i y (target) za trening dataset - Features sve kolone osim 'Survived'
X = df_train.drop(columns=['Survived'])
y = df_train['Survived']

# Podjela podataka na trening i validaciju
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Podjela podataka na X (features) za test dataset
X_test = df_test.copy()

# Random Forest Classifier - treniranje modela, hiperparametri
# Definisanje skupa hiperparametara za GridSearchCV
###
param_grid = {
    'n_estimators': [100, 500, 1000], # Broj stabala u šumi
    'max_depth': [None, 10, 20, 30], # Maksimalna dubina stabla
    'min_samples_split': [2, 5], # Minimalan broj uzoraka potreban za podjelu čvora
    'min_samples_leaf': [1, 2, 4], # Minimalan broj uzoraka u listu
    'max_features': ['sqrt', 0.3], # Broj karakteristika koje se koriste za podjelu čvora
    'class_weight': [None, 'balanced'], # Težine klasa za nerazmjerno raspoređene klase
    'bootstrap': [True, False], # Da li koristiti bootstrap uzorkovanje
}

# Inicijalizacija GridSearchCV sa RandomForestClassifier
rf_grid = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                                 param_grid=param_grid,
                                 cv=5, # K-fold cross-validation
                                 scoring='recall', # Metrika za evaluaciju
                                 n_jobs=-1, # Koristi sve dostupne CPU jezgre
                                 verbose=2)

# Treniranje Random Forest modela koristeći GridSearchCV
rf_grid.fit(X_train, y_train)

# Ispis najboljih hiperparametara i najbolje tačnosti
best_rf_model = rf_grid.best_estimator_
print("Best Hyperparameters:", rf_grid.best_params_)


# Logistic Regression - treniranje modela, hiperparametri
# Definisanje skupa hiperparametara za GridSearchCV
lr_param_grid = {
    'C': [0.01, 0.1, 1, 10, 100], # Inverzna regularizacija
    'penalty': ['l1', 'l2'], # Regularizacija
    'solver': ['liblinear'], # Solver za optimizaciju
    'class_weight': ['balanced'], # Težine klasa za nerazmjerno raspoređene klase
    'max_iter': [1000] # Maksimalan broj iteracija
}

# Inicijalizacija GridSearchCV sa LogisticRegression
lr_grid = GridSearchCV(estimator=LogisticRegression(), 
                                 param_grid=lr_param_grid, 
                                 cv=5, # K-fold cross-validation
                                 scoring='accuracy', # Metrika za evaluaciju
                                 n_jobs=-1, # Koristi sve dostupne CPU jezgre
                                 verbose=2)

# Treniranje Logistic Regression modela koristeći GridSearchCV
lr_grid.fit(X_train, y_train)

# Ispis najboljih hiperparametara i najbolje tačnosti
best_lr_model = lr_grid.best_estimator_
print("Best Hyperparameters for Logistic Regression:", lr_grid.best_params_)


# Predikcija sa Random Forest modelom
rf_predictions = best_rf_model.predict(X_val)

# Predikcija sa Logistic Regression modelom
best_lr_model = best_lr_model.predict(X_val)

# Funkcija za evaluaciju modela i vraćanje metrika
def print_metrics(y_true, y_pred, model_name):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred)
    }

    labels = list(metrics.keys())
    values = list(metrics.values())

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color="cornflowerblue")
    plt.ylim(0, 1.05)
    plt.title(f"Evaluacione metrike za {model_name}")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.2f}", ha='center', va='bottom')

    plt.ylabel("Vrijednost")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_classification_report(y_true, y_pred, model_name="Model"):
    # Dobijanje classification report-a kao dict
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Konverzija u DataFrame
    df = pd.DataFrame(report).transpose()

    # Plot heatmap (isključujemo "support" kolonu jer nije metrike)
    metrics = df.drop(columns=["support"], errors="ignore")

    plt.figure(figsize=(8, 5))
    sns.heatmap(metrics, annot=True, cmap="Blues", fmt=".2f")
    plt.title(f"Classification Report - {model_name}")
    plt.yticks(rotation=0)
    plt.show()

# Evaluacija mdoela sa osnovnim metrikama
print_metrics(y_val, rf_predictions, "Random Forest")
plot_classification_report(y_val, rf_predictions, "Random Forest")
# Evaluacija Logistic Regression modela
print_metrics(y_val, best_lr_model, "Logistic Regression")
plot_classification_report(y_val, best_lr_model, "Logistic Regression")


# Plot Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Ispis Confusion Matrix za modele
plot_confusion_matrix(y_val, rf_predictions, "Random Forest")
plot_confusion_matrix(y_val, best_lr_model, "Logistic Regression")