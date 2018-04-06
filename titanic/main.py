import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

answer_columns = ['Survived']
fetaure_columns = ['Pclass', 'Age']

df = pd.read_csv('./train.csv')
df = df[answer_columns + fetaure_columns].dropna()

x = df.iloc[:, 1:]
y = df.iloc[:, 0]

train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=0.3)

lr = LogisticRegression()
lr.fit(train_x, train_y)

print(lr.score(val_x, val_y))

# pred = lr.predict(val_x)
# print((pred == val_y).sum()/pred.size)

test_df = pd.read_csv('./test.csv')
test_x = test_df[fetaure_columns].fillna(0)
pred = pd.Series(lr.predict(test_x))

result = pd.concat([test_df['PassengerId'], pred], axis=1)
result.to_csv('./submissions/v1.csv',
  index=False, header=['PassengerId', 'Survived'])
