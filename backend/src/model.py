import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import warnings
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn

warnings.filterwarnings('ignore')

model = None
feature_columns = None
label_encoders = {}

app = FastAPI(title="Income Prediction API", 
              description="API для предсказания дохода на основе финансовых данных",
              version="1.0.0")

class PredictionInput(BaseModel):
    data: List[Dict[str, Any]]

class PredictionResponse(BaseModel):
    predictions: List[float]
    ids: List[int]

def load_data():
    train_df = pd.read_csv('hackathon_income_train.csv', sep=';', decimal=',', low_memory=False)
    test_df = pd.read_csv('hackathon_income_test.csv', sep=';', decimal=',', low_memory=False)
    return train_df, test_df

def create_targeted_features(df):
    """Признаки на основе топ-15 важных фичей"""
    df = df.copy()
    
    for col in df.columns:
        if col not in ['id', 'dt'] and df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    top_features = [
        'turn_cur_cr_avg_act_v2', 'incomeValue', 'hdb_bki_total_max_limit',
        'hdb_bki_total_cc_max_limit', 'salary_6to12m_avg', 'age'
    ]
    
    for i, feat1 in enumerate(top_features):
        for feat2 in top_features[i+1:]:
            if feat1 in df.columns and feat2 in df.columns:
                if df[feat1].dtype in [np.int64, np.float64] and df[feat2].dtype in [np.int64, np.float64]:
                    df[f'{feat1}_{feat2}_interaction'] = df[feat1] * df[feat2]
                    df[f'{feat1}_{feat2}_ratio'] = df[feat1] / (df[feat2] + 1)
    
    for col in top_features:
        if col in df.columns and df[col].dtype in [np.int64, np.float64]:
            df[f'log_{col}'] = np.log1p(np.abs(df[col].fillna(0)))
    
    if 'age' in df.columns and 'salary_6to12m_avg' in df.columns:
        df['salary_to_age_ratio'] = df['salary_6to12m_avg'] / (df['age'] + 20)
    
    if 'turn_cur_cr_avg_act_v2' in df.columns and 'salary_6to12m_avg' in df.columns:
        df['turnover_to_salary_ratio'] = df['turn_cur_cr_avg_act_v2'] / (df['salary_6to12m_avg'] + 1)
    
    return df

def preprocess_data(train_df, test_df):
    train_ids = train_df['id']
    test_ids = test_df['id']
    weights = train_df['w']
    target = train_df['target']
    
    features = [col for col in train_df.columns if col not in ['id', 'dt', 'w', 'target']]
    
    features_df = train_df[features].copy()
    test_features = test_df[features].copy()
    
    print("🎯 Создание целевых признаков на основе важности...")
    features_df = create_targeted_features(features_df)
    test_features = create_targeted_features(test_features)
    
    print(f"📈 После feature engineering: {features_df.shape[1]} признаков")
    
    categorical_features = features_df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_features:
        features_df[col] = features_df[col].astype(str)
        test_features[col] = test_features[col].astype(str)
        
        le = LabelEncoder()
        combined = pd.concat([features_df[col], test_features[col]], axis=0)
        le.fit(combined)
        features_df[col] = le.transform(features_df[col])
        test_features[col] = le.transform(test_features[col])
        
        label_encoders[col] = le

    numeric_features = features_df.select_dtypes(include=[np.number]).columns
    features_df[numeric_features] = features_df[numeric_features].fillna(features_df[numeric_features].median())
    test_features[numeric_features] = test_features[numeric_features].fillna(features_df[numeric_features].median())
    
    return features_df, test_features, target, weights, train_ids, test_ids

def train_final_xgboost(X, y, weights, test_data):
    """Финальная версия XGBoost с кросс-валидацией"""
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_predictions = np.zeros(len(X))
    test_predictions = np.zeros(test_data.shape[0])
    models = []
    
    print("Обучение с K-Fold кросс-валидацией...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"Fold {fold + 1}/5")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        w_train, w_val = weights.iloc[train_idx], weights.iloc[val_idx]
        
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
        dval = xgb.DMatrix(X_val, label=y_val, weight=w_val)
        dtest = xgb.DMatrix(test_data)
        
        params = {
            'objective': 'reg:absoluteerror',
            'eval_metric': 'mae',
            'learning_rate': 0.005,
            'max_depth': 10,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.05,
            'reg_lambda': 0.8,
            'gamma': 0.05,
            'random_state': 42 + fold,
            'tree_method': 'hist'
        }
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=3500,
            evals=[(dval, 'validation')],
            early_stopping_rounds=350,
            verbose_eval=False
        )
        
        models.append(model)
        
        oof_predictions[val_idx] = model.predict(dval)
        test_predictions += model.predict(dtest) / kf.n_splits
    
    oof_wmae = (weights * np.abs(y - oof_predictions)).mean()
    print(f"Final XGBoost OOF WMAE: {oof_wmae:.2f}")
    
    return test_predictions, oof_wmae, models

def save_model(model, feature_columns, filename='xgboost_model.pkl'):
    """Сохраняет модель и информацию о фичах"""
    model_data = {
        'model': model,
        'feature_columns': feature_columns,
        'label_encoders': label_encoders
    }
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Модель сохранена в {filename}")

def load_model(filename='xgboost_model.pkl'):
    """Загружает модель и информацию о фичах"""
    global model, feature_columns, label_encoders
    try:
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        model = model_data['model']
        feature_columns = model_data['feature_columns']
        label_encoders = model_data.get('label_encoders', {})
        print(f"Модель загружена из {filename}")
        return True
    except FileNotFoundError:
        print(f"Файл модели {filename} не найден")
        return False

def preprocess_inference_data(df):
    """Препроцессинг для инференса"""
    df = df.copy()
    
    ids = df['id'].tolist() if 'id' in df.columns else list(range(len(df)))
    
    inference_features = [col for col in df.columns if col not in ['id', 'dt', 'w', 'target']]
    features_df = df[inference_features].copy()
    
    features_df = create_targeted_features(features_df)

    categorical_features = features_df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_features:
        if col in label_encoders:
            features_df[col] = features_df[col].astype(str)
            le = label_encoders[col]
            unknown_mask = ~features_df[col].isin(le.classes_)
            if unknown_mask.any():
                features_df.loc[unknown_mask, col] = le.classes_[0]
            features_df[col] = le.transform(features_df[col])
        else:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
    
    numeric_features = features_df.select_dtypes(include=[np.number]).columns
    if hasattr(features_df[numeric_features], 'median'):
        features_df[numeric_features] = features_df[numeric_features].fillna(features_df[numeric_features].median())
    else:
        features_df[numeric_features] = features_df[numeric_features].fillna(0)
    
    return features_df, ids

@app.on_event("startup")
async def startup_event():
    """Загружаем модель при старте приложения"""
    if not load_model():
        print("⚠️ Модель не загружена. Обучите модель сначала.")

@app.get("/")
async def root():
    return {"message": "Income Prediction API", "status": "active"}

@app.get("/health")
async def health_check():
    """Проверка статуса API и модели"""
    model_loaded = model is not None
    return {
        "status": "healthy" if model_loaded else "model_not_loaded",
        "model_loaded": model_loaded,
        "timestamp": pd.Timestamp.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: PredictionInput):
    """Предсказание дохода для новых данных"""
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена. Сначала обучите модель.")
    
    try:
        df = pd.DataFrame(input_data.data)
        
        processed_data, ids = preprocess_inference_data(df)
        
        missing_features = set(feature_columns) - set(processed_data.columns)
        if missing_features:
            for feature in missing_features:
                processed_data[feature] = 0
        
        processed_data = processed_data[feature_columns]
        
        dmatrix = xgb.DMatrix(processed_data)
        predictions = model.predict(dmatrix)
        
        return PredictionResponse(
            predictions=predictions.tolist(),
            ids=ids
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка при предсказании: {str(e)}")

@app.post("/train")
async def train_model():
    """Переобучение модели"""
    try:
        global model, feature_columns
        
        print("🎯 Начинаем обучение модели...")
        train_df, test_df = load_data()
        X, X_test, y, w, train_ids, test_ids = preprocess_data(train_df, test_df)
        
        feature_columns = X.columns.tolist()
        
        predictions, val_wmae, models = train_final_xgboost(X, y, w, X_test)
        
        model = models[0]
        
        save_model(model, feature_columns)
        
        submission = pd.DataFrame({'id': test_ids, 'target': predictions})
        submission_name = f'xgboost_final_wmae_{int(val_wmae)}.csv'
        submission.to_csv(submission_name, index=False)
        
        return {
            "message": "Модель успешно обучена",
            "validation_wmae": float(val_wmae),
            "submission_file": submission_name,
            "features_count": len(feature_columns)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обучении: {str(e)}")

if __name__ == "__main__":
    print("Запуск обучения и API...")
    
    train_df, test_df = load_data()
    X, X_test, y, w, train_ids, test_ids = preprocess_data(train_df, test_df)
    feature_columns = X.columns.tolist()
    
    predictions, val_wmae, models = train_final_xgboost(X, y, w, X_test)
    model = models[0]
    
    save_model(model, feature_columns)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)