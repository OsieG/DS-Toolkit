from fastapi import FastAPI, HTTPException, status, Query
from fastapi.responses import FileResponse
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score,
)

from pathlib import Path

app = FastAPI()
api = KaggleApi()
api.authenticate()


def clean_nan(data):
    return data.astype(object).where(pd.notnull(data), None)

def get_dataset_dir(dataset_ref):
    dataset_dir = Path("downloads") / dataset_ref.replace("/", "__")
    if not dataset_dir.exists() or not any(dataset_dir.iterdir()):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{dataset_ref}' not downloaded yet — call /download first."
        )
    return dataset_dir

def find_csv_file(dataset_dir):
    files = sorted(f.name for f in dataset_dir.iterdir() if f.is_file())
    csv_files = [f for f in files if f.lower().endswith(".csv")]
    if not csv_files:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No CSV file found in this dataset")
    return csv_files[0]

def load_dataframe(dataset_ref):
    dataset_dir = get_dataset_dir(dataset_ref)
    primary_file = find_csv_file(dataset_dir)
    return pd.read_csv(dataset_dir / primary_file)

def preprocess_imputate(df, strategy):
    match strategy:
        case "none":
            return df
        case "mean":
            return df.fillna(df.mean(numeric_only=True))
        case "median":
            return df.fillna(df.median(numeric_only=True))
        case "mode":
            return df.fillna(df.mode().iloc[0])
        case "drop":
            return df.dropna()
        case _:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid strategy '{strategy}'. Must be one of: none, mean, median, mode, drop."
            )

def preprocess_encoding(df, method, columns=None):
    match method:
        case "none":
            return df
        case "onehot":
            if columns is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="One-hot encoding requires specifying columns to encode."
                )
            return pd.get_dummies(df, columns=columns)
        case "label":
            if columns is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Label encoding requires specifying columns to encode."
                )
            le = LabelEncoder()
            for col in columns:
                if col in df.columns:
                    df[col] = le.fit_transform(df[col])
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Column '{col}' not found in dataset."
                    )
            return df
        case _:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid encode_method '{method}'. Must be one of: none, onehot, label."
            )

def preprocess_scaling(df, method, columns=None):
    match method:
        case "none":
            return df
        case "standard":
            scaler = StandardScaler()
            if columns is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Standard scaling requires specifying columns to scale."
                )
            df[columns] = scaler.fit_transform(df[columns])
            return df
        case "minmax":
            scaler = MinMaxScaler()
            if columns is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Min-Max scaling requires specifying columns to scale."
                )
            df[columns] = scaler.fit_transform(df[columns])
            return df
        case _:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid scaling_method '{method}'. Must be one of: none, standard, minmax."
            )

@app.get("/health", status_code=status.HTTP_200_OK)
async def healthcheck():
    return {"status": "ok"}

@app.get("/search")
def search_datasets(query: str, max_results: int = 10):
    try:
        results = api.dataset_list(search=query)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, 
            detail=f"Kaggle search failed: {e}"
            )

    datasets = []
    for d in results[:max_results]:
        datasets.append({
            "ref": d.ref,
            "title": d.title,
            "size": str(getattr(d, "total_bytes", None)),
        })

    return datasets

@app.post("/download")
def download_dataset(dataset_ref:str):
    dataset_dir = Path("downloads") / dataset_ref.replace("/", "__")
    dataset_dir.mkdir(parents=True, exist_ok=True)

    try:
        api.dataset_list_files(dataset_ref)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{dataset_ref}' not found: {e}"
        )

    already_cached = any(dataset_dir.iterdir())

    if not already_cached:
        try:
            api.dataset_download_files(dataset_ref, path=str(dataset_dir), unzip=True)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Kaggle download failed: {e}"
            )

    files = sorted(f.name for f in dataset_dir.iterdir() if f.is_file())
    if not files:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No files found after download")

    return {
        "ref": dataset_ref,
        "cached": already_cached,
        "files": files,
    }


@app.get("/stats")
def dataset_stats(dataset_ref: str):
    df = load_dataframe(dataset_ref)

    #handles specifically for Python Nan values changed for JSON serialization
    df_clean = clean_nan(df)
    summary_clean = clean_nan(df.describe(include='all'))

    stats = {
        'head': df_clean.head().to_dict(orient="records"),
        'shape' : df_clean.shape,
        'columns': df_clean.columns.tolist(),
        'null_counts': df_clean.isnull().sum().to_dict(),
        'summary': summary_clean.to_dict()
    }

    return stats

@app.get("/preprocess")
def preprocess_dataset(
    dataset_ref: str,
    impute_strategy: str = "none",
    encode_method: str = "none",
    encode_columns: list[str] = Query(None),
    scale_method: str = "none",
    scale_columns: list[str] = Query(None),
):
    df = load_dataframe(dataset_ref)
    before_preview = clean_nan(df.head(3)).to_dict(orient="records")

    df = preprocess_imputate(df, impute_strategy)
    df = preprocess_encoding(df, encode_method, encode_columns)
    df = preprocess_scaling(df, scale_method, scale_columns)

    preview = clean_nan(df.head(3)).to_dict(orient="records")

    return {
        "ref": dataset_ref,
        "before_preprocessing": before_preview,
        "after_preprocessing": preview
    }

@app.get("/split")
def split_dataset(
    dataset_ref: str,
    train_size: float,
    val_size: float,
    test_size: float,
    random_state: int = 69
):
    if abs((train_size + val_size + test_size) - 1.0) > 1e-6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="train_size + val_size + test_size must sum to 1.0"
        )
    df = load_dataframe(dataset_ref)

    df_train = df.sample(frac=train_size, random_state=random_state)
    val_ratio = val_size / (val_size + test_size)
    remaining = df.drop(df_train.index)
    df_val = remaining.sample(frac=val_ratio, random_state=random_state)
    df_test = remaining.drop(df_val.index)

    return {
        "ref": dataset_ref,
        "train_shape": df_train.shape,
        "val_shape": df_val.shape,
        "test_shape": df_test.shape,
    }

@app.get("/correlation_matrix")
def correlation_matrix(dataset_ref: str):
    df = load_dataframe(dataset_ref)
    corr_matrix = clean_nan(df.corr(numeric_only=True))
    return {
        "ref": dataset_ref,
        "correlation_matrix": corr_matrix.to_dict()
    }

@app.get("/duplicates")
def find_duplicates(dataset_ref:str, remove: bool = False, subset: list[str] = Query(None)):
    df = load_dataframe(dataset_ref)
    original_shape = df.shape
    duplicate_count = int(df.duplicated(subset=subset).sum())
    before_preview = clean_nan(df.head(3)).to_dict(orient="records")
    
    if remove:
        df = df.drop_duplicates(subset=subset)
        new_shape = df.shape

    return{
        "ref": dataset_ref,
        "original_shape": original_shape,
        "duplicate_count": duplicate_count,
        "before_preview": before_preview,
        "after": clean_nan(df.head(3)).to_dict(orient="records"),
        "new_shape": df.shape if remove else original_shape
    }

@app.get("/outliers")
def find_outliers(dataset_ref: str, columns: list[str] = Query(None)):
    if columns is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one column must be specified for outlier detection."
        )

    df = load_dataframe(dataset_ref)
    outlier_info = {}
    clean_mask = pd.Series(True, index=df.index) 
    for col in columns:
        if col not in df.columns:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Column '{col}' not found in dataset."
            ) 
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        lower_bound = Q1 - 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_info[col] = {
            "outlier_count": len(outliers),
            "percentage": round(len(outliers) / len(df) * 100, 2),
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outlier_indices": outliers.index.tolist()
        }
        # Allows nan values to be preserved in the cleaned dataframe, while still flagging outliers
        clean_mask &= ~((df[col] < lower_bound) | (df[col] > upper_bound))

    clean_df = df[clean_mask]
    outlier_rows_combined = df[~clean_mask]
    
    return {
        "ref": dataset_ref,
        "outlier_info": outlier_info,
        "cleaned_shape": clean_df.shape,
        "flagged_rows_preview": clean_nan(outlier_rows_combined.head()).to_dict(orient="records")
    }

@app.get("/export")
def export_dataset(
    dataset_ref: str,
    impute_strategy: str = "none",
    encode_method: str = "none",
    encode_columns: list[str] = Query(None),
    scale_method: str = "none",
    scale_columns: list[str] = Query(None),
):
    df = load_dataframe(dataset_ref)

    df = preprocess_imputate(df, impute_strategy)
    df = preprocess_encoding(df, encode_method, encode_columns)
    df = preprocess_scaling(df, scale_method, scale_columns)

    dataset_dir = get_dataset_dir(dataset_ref)
    export_file_path = dataset_dir / "exported_dataset.csv"
    df.to_csv(export_file_path, index=False)

    return FileResponse(export_file_path, filename="processed_dataset.csv", media_type="text/csv")

@app.get("/train")
def train_dataset(
        dataset_ref: str,
        target_column: str,
        task_type: str,  # "classification" or "regression"
        impute_strategy: str = "none",
        encode_method: str = "none",
        encode_columns: list[str] = Query(None),
        scale_method: str = "none",
        scale_columns: list[str] = Query(None),
        train_size: float = 0.7,
        test_size: float = 0.3,
        random_state: int = 69,
    ):
    df = load_dataframe(dataset_ref)

    if target_column not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Column '{target_column}' not found in dataset."
        )

    df = preprocess_imputate(df, impute_strategy)
    df = preprocess_encoding(df, encode_method, encode_columns)
    df = preprocess_scaling(df, scale_method, scale_columns)

    # a model can't train on a row with a missing target value
    df = df.dropna(subset=[target_column])

    if abs((train_size + test_size) - 1.0) > 1e-6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="train_size + test_size must sum to 1.0"
        )

    df_train = df.sample(frac=train_size, random_state=random_state)
    df_test = df.drop(df_train.index)

    feature_columns = [c for c in df.columns if c != target_column]
    numeric_features = df[feature_columns].select_dtypes(include=["int64", "float64", "bool"]).columns.tolist()

    if not numeric_features:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No numeric feature columns available after preprocessing. Try encoding more columns first."
        )

    X_train, y_train = df_train[numeric_features], df_train[target_column]
    X_test, y_test = df_test[numeric_features], df_test[target_column]

    match task_type:
        case "classification":
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            return {
                "ref": dataset_ref,
                "task_type": task_type,
                "target_column": target_column,
                "feature_names": numeric_features,
                "train_shape": X_train.shape,
                "test_shape": X_test.shape,
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
                "recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
                "f1_score": f1_score(y_test, predictions, average="weighted", zero_division=0),
                "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
                "coefficients": model.coef_[0].tolist(),
            }
        case "regression":
            model = LinearRegression()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            return {
                "ref": dataset_ref,
                "task_type": task_type,
                "target_column": target_column,
                "feature_names": numeric_features,
                "train_shape": X_train.shape,
                "test_shape": X_test.shape,
                "mse": mean_squared_error(y_test, predictions),
                "r2_score": r2_score(y_test, predictions),
                "coefficients": model.coef_.tolist(),
                "predictions_sample": predictions[:10].tolist(),
                "actual_sample": y_test.head(10).tolist(),
            }
        case _:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid task_type '{task_type}'. Must be one of: classification, regression."
            )

#yasserh/titanic-dataset
#uciml/iris
#uvicorn main:app --reload --port 8000
