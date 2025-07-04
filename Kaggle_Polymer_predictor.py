
import pandas as pd  
import numpy as np
from rdkit import Chem 
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from tqdm import tqdm 

tqdm.pandas()

# Load competition data
train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')

print(train.columns)
print(train.shape)
train.head()

def featurize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [np.nan] * len(descriptor_names)
    return [func(mol) for func in descriptor_funcs] 

descriptor_names, descriptor_funcs = zip(*Descriptors.descList)

X = train['SMILES'].progress_apply(featurize_smiles)
X = pd.DataFrame(X.tolist(), columns=descriptor_names)
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X = X.fillna(0)
X = X.clip(-1e5, 1e5)
X = X.astype(np.float32)
X_test = test['SMILES'].progress_apply(featurize_smiles)
X_test = pd.DataFrame(X_test.tolist(), columns=descriptor_names)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test = X_test.fillna(0)
X_test = X_test.clip(-1e5, 1e5)
X_test = X_test.astype(np.float32)

print("Max value in X:", X.max().max())
print("Max value in X_test:", X_test.max().max())


print("Any NaN in X?", X.isna().any().any())
print("Any inf in X?", np.isinf(X.values).any())
print("Any NaN in X_test?", X_test.isna().any().any())
print("Any inf in X_test?", np.isinf(X_test.values).any())


targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
submission = pd.DataFrame({'id': test['id']})

for target in targets:
    print(f"\nTraining model for target: {target}")
    y = train[target]
    valid_idx = y.notnull() & y.apply(np.isfinite)
    X_target = X[valid_idx]
    y_clean = y[valid_idx]


    print("Any NaN in X_target?", X_target.isna().any().any())
    print("Any inf in X_target?", np.isinf(X_target.values).any())
    print("Any NaN in y_clean?", y_clean.isna().any())
    print("Any inf in y_clean?", np.isinf(y_clean.values).any())


    X_train, X_val, y_train, y_val = train_test_split(X_target, y_clean, test_size=0.1, random_state=42)
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    val_preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"Validation RMSE for {target}: {rmse:.4f}")
    submission[target] = model.predict(X_test)

submission.to_csv('submission.csv', index=False)
submission.head()