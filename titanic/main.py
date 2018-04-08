import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def process_age(df):
    age = df.Age.fillna(df.Age.dropna().mean())
    return pd.cut(age, 5, labels=[1, 2, 3, 4, 5]).astype(int)


def process_sex(df):
    return df.Sex.map({'male': 0, 'female': 1})


def process_embarked(df):
    em = df.Embarked.fillna('S')
    return em.map({'C': 0, 'Q': 1, 'S': 2})


def process_cabin(df):
    return df.Cabin


def create_familzy_size(df):
    return df.Parch + df.SibSp + 1


def extract_feature(df, is_train):
    age_band = process_age(df)
    sex = process_sex(df)
    embarked = process_embarked(df)
    cabin = process_cabin(df)
    family_size = create_familzy_size(df)
    fare = df.Fare.fillna(df.Fare.dropna().mean())

    x = pd.DataFrame({
        'Pclass': df.Pclass,
        'Age': age_band,
        'Sex': sex,
        'Fare': fare,
        'Embarked': embarked,
        # 'Cabin': cabin,
        'FamilySize': family_size,
    })
    # print(pd.concat([df.Age, age_band, pd.cut(df.Age, 5)], axis=1).head(10))
    y = df.Survived if is_train else None
    return (x, y)


def main():
    df = pd.read_csv('./input/train.csv')
    x, y = extract_feature(df, is_train=True)
    print('----- features')
    print(x.head())
    print('-' * 20)

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
