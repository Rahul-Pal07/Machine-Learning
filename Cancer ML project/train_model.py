import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data= pd.read_csv(r"C:\Users\Rahul.user\Desktop\Machine Learning\Dataset practise\cancer.csv")
data

print(data.head(10))
print(data.shape)

X = data[[
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness"
]]

y= data['target']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
# taking the X_scaled value instead of X in train_test_split(X,y, test_sizze=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)  

model= LogisticRegression(max_iter=1000)

model.fit(X_train,y_train)

y_pred= model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved successfully")