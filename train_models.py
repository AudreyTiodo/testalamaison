import pickle
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Charger le dataset Iris
iris = load_iris()
X, y = iris.data, iris.target

# 1. Logistic Regression
logreg_model = LogisticRegression(max_iter=200)
logreg_model.fit(X, y)

# 2. Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X, y)

# Sauvegarde des modèles
with open("logistic_regression.pkl", "wb") as f:
    pickle.dump(logreg_model, f)

with open("random_forest.pkl", "wb") as f:
    pickle.dump(rf_model, f)

print("Models trained and saved successfully!")