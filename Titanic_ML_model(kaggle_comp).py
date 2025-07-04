
from google.colab import files
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
import numpy as np
from google.colab import files

files.download('submission_voting.csv')

uploaded = files.upload()
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
 
train['Age'] = train['Age'].fillna(train['Age'].median())
test['Age'] = test['Age'].fillna(test['Age'].median())

test['Fare'] = test['Fare'].fillna(test['Fare'].median())

train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])
test['Embarked'] = test['Embarked'].fillna(test['Embarked'].mode()[0])

train = train.drop(columns=['Cabin'])
test = test.drop(columns=['Cabin'])

train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})

embarked_map = {'S': 0, 'C': 1, 'Q': 2}
train['Embarked'] = train['Embarked'].map(embarked_map)
test['Embarked'] = test['Embarked'].map(embarked_map)

train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1

train['IsAlone'] = (train['FamilySize'] == 1).astype(int)
test['IsAlone'] = (test['FamilySize'] == 1).astype(int)

train['Title'] = train['Name'].str.extract(' ([A-Z][a-z]+)\.', expand=False)
test['Title'] = test['Name'].str.extract(' ([A-Z][a-z]+)\.', expand=False)

for dataset in [train, test]:
    dataset['Title'] = dataset['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace(['Mlle', 'Ms'], 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    title_map = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
    train['Title'] = train['Title'].map(title_map)
    test['Title'] = test['Title'].map(title_map)

train['Title'] = train['Title'].fillna(0)
test['Title'] = test['Title'].fillna(0)

train['Age*Class'] = train['Age'] * train['Pclass']
test['Age*Class'] = test['Age'] * test['Pclass']

train = train.drop(columns=['Name', 'Ticket', 'SibSp', 'Parch', 'PassengerId'])
test_passenger_ids = test['PassengerId']
test = test.drop(columns=['Name', 'Ticket', 'SibSp', 'Parch', 'PassengerId'])

X = train.drop(columns=['Survived'])
y = train['Survived']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

log_clf = LogisticRegression(max_iter=1000)
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_clf = XGBClassifier(n_estimators=100, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
voting_clf = VotingClassifier(
    estimators=[
        ('lr', log_clf),
        ('rf', rf_clf),
        ('gb', gb_clf),
        ('xgb', xgb_clf)
    ],
    voting='soft'
)

voting_clf.fit(X_train, y_train)
y_pred = voting_clf.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print("Voting ENsemble Validation Accuracy:", acc)

log_clf.fit(X_train, y_train)
rf_clf.fit(X_train, y_train)
xgb_clf.fit(X_train, y_train)

val_log = log_clf.predict_proba(X_val)[:, 1]
val_rf = rf_clf.predict_proba(X_val)[:, 1]
val_xgb = xgb_clf.predict_proba(X_val)[:, 1]

meta_X = np.column_stack((val_log, val_rf, val_xgb))

meta_model = LogisticRegression()
meta_model.fit(meta_X, y_val)

test_log = log_clf.predict_proba(test)[:, 1]
test_rf = rf_clf.predict_proba(test)[:, 1]
test_xgb = xgb_clf.predict_proba(test)[:, 1]

meta_test = np.column_stack((test_log, test_rf, test_xgb))

final_preds = meta_model.predict(meta_test)

submission = pd.DataFrame({
    'PassengerId': test_passenger_ids,
    'Survived': final_preds
})

submission.to_csv('submission_voting.csv', index=False)
print("Sea God Submission Ready")
