# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Step 1: Load the Iris dataset
df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')

# Step 2: Split into features (X) and target (y)
X = df.drop('species', axis=1)   # Features
y = df['species']                # Labels

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Train a model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 5: Save the trained model to 'iris_model.pkl'
with open('iris_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as 'iris_model.pkl'")
