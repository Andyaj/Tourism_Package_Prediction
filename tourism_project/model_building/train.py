import os
import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
import mlflow

# The GitHub Actions job starts an MLflow server on port 5000 before this script runs.
# MLFLOW_TRACKING_URI can override it, for example when the script is run locally.
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("tourism-package-prediction")

# Xtrain/Xtest/ytrain/ytest are downloaded from the previous job's artifact
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()

numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome",
]
categorical_features = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]

# Ratio of non-buyers to buyers, used to weight the minority class
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
print(f"scale_pos_weight = {class_weight:.3f}")

# Preprocessing: scale numeric features, one-hot encode categorical features
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)

# Base XGBoost classifier
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Hyperparameter grid
param_grid = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5],
    "xgbclassifier__colsample_bytree": [0.8, 1.0],
    "xgbclassifier__colsample_bylevel": [0.8, 1.0],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__reg_lambda": [1, 5],
}

# Preprocessing and model in one pipeline so the saved object accepts raw inputs
model_pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run():
    # Tune on F1 of the positive class because the target is imbalanced
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log every parameter combination as a nested run
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_test_score", results["mean_test_score"][i])
            mlflow.log_metric("std_test_score", results["std_test_score"][i])

    # Log the best hyperparameters in the main run
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_cv_f1", grid_search.best_score_)
    print("Combinations tried:", len(results["params"]))
    print("Best parameters:", grid_search.best_params_)
    print(f"Best cross-validated F1: {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_

    # Choose the classification threshold from out-of-fold predictions on the training set.
    # The test set is not used for this choice and is kept for the final evaluation.
    threshold_grid = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    oof_proba = cross_val_predict(best_model, Xtrain, ytrain, cv=5, method="predict_proba")[:, 1]
    oof_f1 = [f1_score(ytrain, (oof_proba >= t).astype(int)) for t in threshold_grid]
    classification_threshold = threshold_grid[int(np.argmax(oof_f1))]
    print(f"Classification threshold selected on the training folds: {classification_threshold}")

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    mlflow.log_param("classification_threshold", classification_threshold)
    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1-score": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1-score": test_report["1"]["f1-score"],
    })

    print("\nTrain classification report:")
    print(classification_report(ytrain, y_pred_train))
    print("Test classification report:")
    print(classification_report(ytest, y_pred_test))
    print("Test confusion matrix (rows: actual 0/1, columns: predicted 0/1):")
    print(confusion_matrix(ytest, y_pred_test))

    # Save the best model next to app.py so the workflow can commit it and Streamlit can load it
    model_path = "tourism_project/deployment/best_tourism_package_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"\nModel saved to {model_path}")
