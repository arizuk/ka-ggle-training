import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def process_age(df):
    age = df.Age.fillna(df.Age.dropna().mean())
    return pd.cut(age, 5, labels=[1, 2, 3, 4, 5]).astype(int)


def process_sex(df):
    return df.Sex.map({'male': 0, 'female': 1})


def process_embarked(df):
    em = df.Embarked.fillna('S')
    return em.map({'C': 0, 'Q': 1, 'S': 2})


def process_cabin(df):
    cabin = df.Cabin.fillna('U').astype(str)
    return cabin.apply(lambda x: 0 if x[0] == 'U' else 1)


def create_familzy_size(df):
    return df.Parch + df.SibSp + 1


def extract_feature(df, is_train):
    age_band = process_age(df)
    sex = process_sex(df)
    embarked = process_embarked(df)
    cabin = process_cabin(df)
    fare = df.Fare.fillna(df.Fare.dropna().mean())

    family_size = create_familzy_size(df)
    is_alone = family_size.apply(lambda x: 1 if x == 1 else 0)

    x = pd.DataFrame({
        'Pclass': df.Pclass,
        'Age': age_band,
        'Sex': sex,
        'Fare': fare,
        'Embarked': embarked,
        # 'Cabin': cabin,
        # 'FamilySize': family_size,
        'IsAlone': is_alone,
    })
    # print(pd.concat([df.Age, age_band, pd.cut(df.Age, 5)], axis=1).head(10))
    y = df.Survived if is_train else None
    return (x, y)


def make_model(name):
    if name == 'lr':
        c = 0.1
        model = LogisticRegression(C=c)
    elif name == 'random_forest':
        model = RandomForestClassifier(n_estimators=100)
    else:
        raise Exception('Invalid model name')

    return model


def main():
    df = pd.read_csv('./input/train.csv')
    x, y = extract_feature(df, is_train=True)
    print('----- features')
    print(x.head())
    print('-' * 20)

    train_x, val_x, train_y, val_y = train_test_split(
        x, y, test_size=0.3, random_state=1, stratify=y)

    model_name = 'random_forest'
    model = make_model(model_name)
    model.fit(train_x, train_y)

    val_pred = model.predict(val_x)
    # print("c: {}, score: {}".format(c, model.score(val_x, val_y)))
    print("score: {}".format(model.score(val_x, val_y)))
    print('-' * 20)
    print(classification_report(val_y, val_pred))

    # pred = model.predict(val_x)
    # print((pred == val_y).sum()/pred.size)

    test_df = pd.read_csv('./input/test.csv')
    test_x, _ = extract_feature(test_df, is_train=False)

    pred = model.predict(test_x)
    result = pd.DataFrame({
        'PassengerId': test_df.PassengerId,
        'Survived': pred
        })
    result.to_csv('./submissions/{}.csv'.format(model_name), index=False)


if __name__ == '__main__':
    main()
