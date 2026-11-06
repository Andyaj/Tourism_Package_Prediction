import pandas as pd

RAW_PATH = "tourism_project/data/tourism.csv"

# Load the raw dataset from the repository data folder
df = pd.read_csv(RAW_PATH)

# Validate that every column in the data dictionary is present
expected_columns = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]
missing = [c for c in expected_columns if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

extra = [c for c in df.columns if c not in expected_columns]

# Summary of the registered dataset
print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", list(df.columns))
print("Columns not in the data dictionary:", extra)
print("Missing values per column:")
print(df.isnull().sum())
print("Duplicate CustomerID values:", df["CustomerID"].duplicated().sum())
print("ProdTaken distribution:")
print(df["ProdTaken"].value_counts())
