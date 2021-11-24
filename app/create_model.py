import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import dill

df = pd.read_csv("data_banknote_authentication.txt", names = ['variance', 'kewness', 'curtosis', 'entropy', 'class'])
df.head()



features = ['variance', 'kewness', 'curtosis', 'entropy']
target = 'class'

X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.33, random_state=42)
#save test
X_test.to_csv("X_test.csv", index=None)
y_test.to_csv("y_test.csv", index=None)
#save train
X_train.to_csv("X_train.csv", index=None)
y_train.to_csv("y_train.csv", index=None)

pipeline = Pipeline([('normalizer', StandardScaler()), 
                     ('classifier', GradientBoostingClassifier(random_state = 21))])

pipeline.fit(X_train, y_train)

with open("gradient_boosting_pipeline.dill", "wb") as f:
    dill.dump(pipeline, f)
