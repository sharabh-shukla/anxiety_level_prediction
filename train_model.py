import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

DATA_FILE = "enhanced_anxiety_dataset.csv"
TARGET = "Anxiety Level (1-10)"

df = pd.read_csv(DATA_FILE)

X = df.drop(columns=[TARGET])
y = df[TARGET].astype(float)

categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_features = [c for c in X.columns if c not in categorical_features]

preprocessor = ColumnTransformer(
    [
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ]
)

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    max_features=0.8,
    min_samples_leaf=2,
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

pipeline.fit(X_train, y_train)

pred = pipeline.predict(X_test)
print("MAE:", mean_absolute_error(y_test, pred))
print("R2:", r2_score(y_test, pred))

joblib.dump(pipeline.named_steps["preprocessor"], "preprocessor.pkl", compress=3)
joblib.dump(pipeline.named_steps["model"], "best_anxiety_model.pkl", compress=3)

print("Saved preprocessor.pkl and best_anxiety_model.pkl")
