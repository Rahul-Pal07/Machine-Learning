import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

np.random.seed(42)

X = pd.DataFrame({
    'area':np.random.randint(500,3000,100),
    'bedroom':np.random.randint(1,5,100),
    'age':np.random.randint(0,30,100)
})

y = X["area"]*3000+X['bedroom']*50000-X['age']*1000

model = LinearRegression()
model.fit(X,y)

joblib.dump(model,'model.pkl')
print("model save in model.pkl file")