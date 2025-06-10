# Import biblioteka
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# 1. Učitavanje i analiza podataka

# Učitavanje CSV datoteke
df = pd.read_csv("HR_comma_sep.csv")  

# pregled podataka
print(df.head())
print(df.info())

# Grafički prikaz zaposlenika koji su otišli i koji su ostali raditi
df['left_status'] = df['left'].map({0: 'ostali', 1: 'otišli'})
custom_palette = {'ostali': 'green', 'otišli': 'red'}
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Department', hue='left_status', palette=custom_palette)
plt.title("Odlazak po odjelima")
plt.xticks(rotation=45)
plt.xlabel("Odjel")
plt.ylabel("Broj zaposlenika")
plt.legend(title='Status')
plt.show()

# 3. Priprema podataka

# Kopiranje podataka i enkodiranje kategorija
data = df.copy()
label_enc = LabelEncoder()
data['Department'] = label_enc.fit_transform(data['Department'])
data['salary'] = label_enc.fit_transform(data['salary'])

# Odvajanje ulaznih i izlaznih varijabli
X = data.drop('left', axis=1)
y = data['left']

# Podjela na trening i test skup
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Treniranje više modela

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(),
    "Decision Tree": DecisionTreeClassifier()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print(f"\n===== {name} =====")
    print(f"Preciznost: {acc:.4f}")
    print(classification_report(y_test, y_pred))

# 6. Usporedba preciznosti modela

plt.figure(figsize=(10, 6))
sns.barplot(x=list(results.keys()), y=list(results.values()), hue=list(results.keys()), palette="viridis", legend=False)
plt.title("Usporedba preciznosti različitih modela")
plt.ylabel("Preciznost")
plt.ylim(0.7, 1.0)
plt.xticks(rotation=15)
plt.grid(axis='y')
plt.show()

# 5. Konfuzione matrice

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, (name, model) in enumerate(models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ostao", "Otišao"])
    disp.plot(ax=axes[i], cmap='Blues', colorbar=False)
    axes[i].set_title(f"Confusion Matrix: {name}")

plt.tight_layout()
plt.show()