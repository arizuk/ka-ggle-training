import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def process_sex(df):
    return df.Sex.apply(lambda x: 1 if x == 'male' else 2)


def process_embarked(df):
    def convert(x):
        if x == 'C':
            return 1
        elif x == 'Q':
            return 2
        elif x == 'S':
            return 3
        else:
            return 0

    return df.Embarked.apply(convert)


def process_cabin(df):
    def conv(x):
        if isinstance(x, str) and x != '':
            return ord(x[0]) - 64
        else:
            return 0
    return df.Cabin.apply(conv)


def extract_feature(df, is_train):
    columns = [
        'Pclass',
        'Age',
        'Sex',
        'Fare',
        # 'Embarked',
        # 'Cabin',
        'Parch',
    ]

    y = None

    if is_train:
        columns = ['Survived'] + columns
        df = df[columns].dropna(0)

        x = df.iloc[:, 1:]
        y = df.iloc[:, 0]
    else:
        x = df[columns].fillna(0) # TODO: 適当すぎる

    if 'Sex' in x:
        x['Sex'] = process_sex(df)

    if 'Embarked' in x:
        x['Embarked'] = process_embarked(df)

    if 'Cabin' in x:
        x['Cabin'] = process_cabin(df)

    # print(x.head(10))
    # import sys
    # sys.exit()

    return (x, y)


def main():
    df = pd.read_csv('./input/train.csv')
    x, y = extract_feature(df, is_train=True)

    train_x, val_x, train_y, val_y = train_test_split(
        x, y, test_size=0.3, random_state=1, stratify=y)

    lr = LogisticRegression()
    lr.fit(train_x, train_y)

    print("score: {}".format(lr.score(val_x, val_y)))

    # pred = lr.predict(val_x)
    # print((pred == val_y).sum()/pred.size)

    test_df = pd.read_csv('./input/test.csv')
    test_x, _ = extract_feature(test_df, is_train=False)

    pred = lr.predict(test_x)
    result = pd.DataFrame({
        'PassengerId': test_df.PassengerId,
        'Survived': pred
        })
    result.to_csv('./submissions/output.csv', index=False)


if __name__ == '__main__':
    main()
