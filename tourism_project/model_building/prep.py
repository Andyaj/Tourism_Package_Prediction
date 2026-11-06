import pandas as pd
from sklearn.model_selection import train_test_split

# Load the registered dataset from the repository data folder
df = pd.read_csv("tourism_project/data/tourism.csv")
print("Raw shape:", df.shape)

# Data cleaning
# 1. Drop columns that carry no predictive information:
#    "Unnamed: 0" is a row index written by the CSV export and CustomerID is an identifier.
drop_cols = [c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns]
df = df.drop(columns=drop_cols)
print("Dropped columns:", drop_cols)

# 2. Fix the inconsistent spelling in Gender ("Fe Male" is the same category as "Female")
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
print("Gender categories after cleaning:", sorted(df["Gender"].unique()))

# 3. Report missing values and duplicate rows after cleaning
print("Missing values:", int(df.isnull().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))

# Categorical columns are kept as raw strings. The training pipeline one-hot
# encodes them, and the Streamlit app sends the same raw strings, so training
# and serving use one representation.

target = "ProdTaken"
X = df.drop(columns=[target])
y = df[target]

# Stratify on the target so the purchase ratio is the same in both splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print("Train shape:", Xtrain.shape, "Test shape:", Xtest.shape)
print("ProdTaken distribution in train:")
print(ytrain.value_counts())
print("ProdTaken distribution in test:")
print(ytest.value_counts())
