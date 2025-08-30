#!/usr/bin/env python3
"""
하이퍼파라미터 튜닝
- 최고 성능 모델들의 하이퍼파라미터 최적화
- Optuna를 사용한 베이지안 최적화
- 최종 테스트 세트 평가
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import optuna
import warnings
warnings.filterwarnings('ignore')

# 머신러닝 라이브러리
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

class HyperparameterTuner:
    def __init__(self, modeling_results_path, resampling_path, output_path):
        self.modeling_results_path = Path(modeling_results_path)
        self.resampling_path = Path(resampling_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        self.target_column = 'mortality_48h'
        self.tuned_models = {}
        self.final_results = []
        
    def load_modeling_results(self):
        """모델링 결과 로드하여 모든 모델 유형 확인"""
        print("📊 모델링 결과 로드 중...")
        
        results_file = self.modeling_results_path / "modeling_results.csv"
        results_df = pd.read_csv(results_file)
        
        # 모든 고유 모델 유형 추출
        unique_models = results_df['model_name'].unique()
        
        print("🎯 튜닝 대상 모델 유형:")
        for idx, model_name in enumerate(unique_models):
            best_for_model = results_df[results_df['model_name'] == model_name].iloc[0]
            print(f"  {idx+1}. {model_name} (최고 ROC-AUC: {best_for_model['roc_auc']:.4f})")
        
        return results_df
    
    def load_resampling_data(self, resampling_method):
        """특정 리샘플링 방법의 데이터 로드"""
        print(f"📂 {resampling_method.upper()} 데이터 로드 중...")
        
        data_path = self.resampling_path / resampling_method
        
        train_df = pd.read_csv(data_path / "mimic_mortality_train.csv")
        val_df = pd.read_csv(data_path / "mimic_mortality_validation.csv")
        test_df = pd.read_csv(data_path / "mimic_mortality_test.csv")
        
        print(f"✅ 데이터 로드 완료: Train({len(train_df)}), Val({len(val_df)}), Test({len(test_df)})")
        return train_df, val_df, test_df
    
    def preprocess_data(self, train_df, val_df, test_df):
        """데이터 전처리 (06번과 동일)"""
        print("🔄 데이터 전처리...")
        
        # ID 컬럼 제거
        id_columns = ['subject_id', 'hadm_id', 'stay_id']
        existing_id_cols = [col for col in id_columns if col in train_df.columns]
        
        # 특성과 타겟 분리
        X_train = train_df.drop(columns=existing_id_cols + [self.target_column])
        y_train = train_df[self.target_column]
        X_val = val_df.drop(columns=existing_id_cols + [self.target_column])
        y_val = val_df[self.target_column]
        X_test = test_df.drop(columns=existing_id_cols + [self.target_column])
        y_test = test_df[self.target_column]
        
        # 문자열 컬럼 인코딩 (데이터 누수 방지)
        from sklearn.preprocessing import LabelEncoder
        
        categorical_columns = X_train.select_dtypes(include=['object']).columns
        label_encoders = {}
        
        for col in categorical_columns:
            le = LabelEncoder()
            # Train 데이터로만 fit하여 데이터 누수 방지
            le.fit(X_train[col].astype(str))
            label_encoders[col] = le
            
            # 각 세트 변환
            X_train[col] = le.transform(X_train[col].astype(str))
            
            # Validation과 Test에서 새로운 값 처리
            for val in X_val[col].astype(str):
                if val not in le.classes_:
                    # 새로운 값을 가장 빈번한 클래스로 대체
                    most_frequent = X_train[col].mode()[0] if not X_train[col].empty else 0
                    X_val.loc[X_val[col].astype(str) == val, col] = most_frequent
            X_val[col] = le.transform(X_val[col].astype(str))
            
            for val in X_test[col].astype(str):
                if val not in le.classes_:
                    # 새로운 값을 가장 빈번한 클래스로 대체
                    most_frequent = X_train[col].mode()[0] if not X_train[col].empty else 0
                    X_test.loc[X_test[col].astype(str) == val, col] = most_frequent
            X_test[col] = le.transform(X_test[col].astype(str))
        
        # 결측치 처리
        for col in X_train.columns:
            if X_train[col].isnull().any():
                mean_val = X_train[col].mean()
                X_train[col].fillna(mean_val, inplace=True)
                X_val[col].fillna(mean_val, inplace=True)
                X_test[col].fillna(mean_val, inplace=True)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def tune_xgboost(self, X_train, X_val, y_train, y_val, n_trials=50):
        """XGBoost 하이퍼파라미터 튜닝"""
        print("🎯 XGBoost 튜닝 중...")
        
        def objective(trial):
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'verbosity': 0,
                'random_state': 42,
                
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            }
            
            model = xgb.XGBClassifier(**params)
            
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, y_pred_proba)
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # 최적 모델 학습
        best_params = study.best_params
        best_params.update({'objective': 'binary:logistic', 'eval_metric': 'auc', 'verbosity': 0, 'random_state': 42})
        
        best_model = xgb.XGBClassifier(**best_params)
        best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        print(f"   최적 ROC-AUC: {study.best_value:.4f}")
        return best_model, best_params
    
    def tune_lightgbm(self, X_train, X_val, y_train, y_val, n_trials=50):
        """LightGBM 하이퍼파라미터 튜닝"""
        print("🎯 LightGBM 튜닝 중...")
        
        def objective(trial):
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'verbosity': -1,
                'random_state': 42,
                
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            }
            
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, y_pred_proba)
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # 최적 모델 학습
        best_params = study.best_params
        best_params.update({'objective': 'binary', 'metric': 'auc', 'verbosity': -1, 'random_state': 42})
        
        best_model = lgb.LGBMClassifier(**best_params)
        best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        
        print(f"   최적 ROC-AUC: {study.best_value:.4f}")
        return best_model, best_params
    
    def tune_random_forest(self, X_train, X_val, y_train, y_val, n_trials=30):
        """Random Forest 하이퍼파라미터 튜닝"""
        print("🎯 Random Forest 튜닝 중...")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1
            }
            
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
            
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, y_pred_proba)
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # 최적 모델 학습
        best_params = study.best_params
        best_params.update({'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1})
        
        best_model = RandomForestClassifier(**best_params)
        best_model.fit(X_train, y_train)
        
        print(f"   최적 ROC-AUC: {study.best_value:.4f}")
        return best_model, best_params
    
    def tune_extra_trees(self, X_train, X_val, y_train, y_val, n_trials=30):
        """Extra Trees 하이퍼파라미터 튜닝"""
        print("🎯 Extra Trees 튜닝 중...")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1
            }
            
            model = ExtraTreesClassifier(**params)
            model.fit(X_train, y_train)
            
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, y_pred_proba)
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # 최적 모델 학습
        best_params = study.best_params
        best_params.update({'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1})
        
        best_model = ExtraTreesClassifier(**best_params)
        best_model.fit(X_train, y_train)
        
        print(f"   최적 ROC-AUC: {study.best_value:.4f}")
        return best_model, best_params
    
    def tune_svc(self, X_train, X_val, y_train, y_val, n_trials=30):
        """SVC 하이퍼파라미터 튜닝"""
        print("🎯 SVC 튜닝 중...")
        
        def objective(trial):
            params = {
                'C': trial.suggest_float('C', 0.1, 10.0, log=True),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear']),
                'class_weight': 'balanced',
                'probability': True,
                'random_state': 42
            }
            
            model = SVC(**params)
            model.fit(X_train, y_train)
            
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, y_pred_proba)
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # 최적 모델 학습
        best_params = study.best_params
        best_params.update({'class_weight': 'balanced', 'probability': True, 'random_state': 42})
        
        best_model = SVC(**best_params)
        best_model.fit(X_train, y_train)
        
        print(f"   최적 ROC-AUC: {study.best_value:.4f}")
        return best_model, best_params
    
    def tune_logistic_regression(self, X_train, X_val, y_train, y_val, n_trials=30):
        """Logistic Regression 하이퍼파라미터 튜닝"""
        print("🎯 Logistic Regression 튜닝 중...")
        
        def objective(trial):
            params = {
                'C': trial.suggest_float('C', 0.1, 10.0, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                'solver': 'liblinear',
                'class_weight': 'balanced',
                'random_state': 42,
                'max_iter': 1000
            }
            
            model = LogisticRegression(**params)
            model.fit(X_train, y_train)
            
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, y_pred_proba)
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # 최적 모델 학습
        best_params = study.best_params
        best_params.update({'class_weight': 'balanced', 'random_state': 42, 'max_iter': 1000, 'solver': 'liblinear'})
        
        best_model = LogisticRegression(**best_params)
        best_model.fit(X_train, y_train)
        
        print(f"   최적 ROC-AUC: {study.best_value:.4f}")
        return best_model, best_params
    
    def evaluate_final_model(self, model, model_name, X_test, y_test):
        """최종 테스트 세트 평가"""
        print(f"📊 {model_name} 최종 평가...")
        
        # 예측
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # 지표 계산
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'pr_auc': average_precision_score(y_test, y_pred_proba),
            'confusion_matrix': {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}
        }
        
        print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"   F1-Score: {metrics['f1_score']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        
        return metrics
    
    def run_tuning(self):
        """하이퍼파라미터 튜닝 실행"""
        print("🎯 하이퍼파라미터 튜닝 시작")
        print("=" * 60)
        
        # 모델링 결과 로드
        all_results = self.load_modeling_results()
        
        # 모든 고유 모델 유형 추출
        unique_models = all_results['model_name'].unique()
        
        # 모든 리샘플링 방법 정의
        all_resampling_methods = ['original', 'smote', 'downsampling']
        
        # 각 모델별 처리
        for model_name in unique_models:
            print(f"\n🔧 {model_name} 튜닝 시작...")
            
            # XGBoost와 LightGBM은 모든 데이터셋에 대해 튜닝
            if model_name in ['XGBoost', 'LightGBM']:
                print(f"   🎯 {model_name}: 모든 데이터셋에 대해 튜닝 수행")
                
                for resampling_method in all_resampling_methods:
                    print(f"\n   📂 {resampling_method.upper()} 데이터셋으로 {model_name} 튜닝...")
                    
                    try:
                        # 해당 리샘플링 데이터 로드
                        train_df, val_df, test_df = self.load_resampling_data(resampling_method)
                        
                        # 데이터 전처리
                        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocess_data(train_df, val_df, test_df)
                        
                        # 모델 튜닝
                        if model_name == 'XGBoost':
                            tuned_model, best_params = self.tune_xgboost(X_train, X_val, y_train, y_val)
                        elif model_name == 'LightGBM':
                            tuned_model, best_params = self.tune_lightgbm(X_train, X_val, y_train, y_val)
                        
                        # 모델 저장
                        model_filename = f"tuned_{model_name.lower()}_{resampling_method}.pkl"
                        model_path = self.output_path / model_filename
                        joblib.dump(tuned_model, model_path)
                        
                        # 최종 평가
                        final_metrics = self.evaluate_final_model(tuned_model, f"{model_name}_{resampling_method}", X_test, y_test)
                        final_metrics['best_params'] = best_params
                        final_metrics['model_path'] = str(model_path)
                        final_metrics['resampling_method'] = resampling_method
                        final_metrics['base_model'] = model_name
                        
                        self.final_results.append(final_metrics)
                        print(f"      ✅ {model_name}_{resampling_method} 튜닝 완료!")
                        
                    except Exception as e:
                        print(f"      ❌ {model_name}_{resampling_method} 튜닝 실패: {str(e)}")
                        continue
            
            # 다른 모델들은 튜닝하지 않음
            else:
                print(f"   ⚠️ {model_name}: 튜닝 대상이 아닙니다. 건너뜁니다.")
    
    def save_results(self):
        """결과 저장"""
        print("\n💾 튜닝 결과 저장 중...")
        
        if not self.final_results:
            print("❌ 저장할 결과가 없습니다.")
            return
        
        # 결과 DataFrame 생성 (best_params 제외)
        results_data = []
        for result in self.final_results:
            result_copy = result.copy()
            del result_copy['best_params']  # 파라미터는 별도 저장
            del result_copy['confusion_matrix']  # 혼동행렬 제외
            results_data.append(result_copy)
        
        results_df = pd.DataFrame(results_data)
        results_df = results_df.sort_values('roc_auc', ascending=False)
        
        # CSV 저장
        results_csv = self.output_path / "tuning_results.csv"
        results_df.to_csv(results_csv, index=False)
        print(f"✅ 튜닝 결과: {results_csv}")
        
        # 요약 리포트
        self.create_final_report(results_df)
    
    def create_final_report(self, results_df):
        """최종 리포트 생성"""
        summary_file = self.output_path / "final_report.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("MIMIC-IV 48시간 사망률 예측 - 최종 결과\n")
            f.write("=" * 60 + "\n")
            f.write(f"완료 일시: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("🎯 하이퍼파라미터 튜닝 결과:\n")
            f.write("-" * 40 + "\n")
            
            # 최고 성능 모델
            if len(results_df) > 0:
                best_model = results_df.iloc[0]
                f.write(f"🏆 최고 성능 모델: {best_model['model_name']}\n")
                f.write(f"   리샘플링 방법: {best_model['resampling_method'].upper()}\n")
                f.write(f"   Test ROC-AUC: {best_model['roc_auc']:.4f}\n")
                f.write(f"   Test F1-Score: {best_model['f1_score']:.4f}\n")
                f.write(f"   Test Precision: {best_model['precision']:.4f}\n")
                f.write(f"   Test Recall: {best_model['recall']:.4f}\n")
                f.write(f"   Test Accuracy: {best_model['accuracy']:.4f}\n\n")
            
            # 모든 모델 성능
            f.write("📊 전체 모델 성능 (Test Set):\n")
            f.write("-" * 40 + "\n")
            for _, row in results_df.iterrows():
                f.write(f"{row['model_name']}:\n")
                f.write(f"   ROC-AUC: {row['roc_auc']:.4f}\n")
                f.write(f"   F1-Score: {row['f1_score']:.4f}\n")
                f.write(f"   Precision: {row['precision']:.4f}\n")
                f.write(f"   Recall: {row['recall']:.4f}\n\n")
            
            f.write("💡 결론:\n")
            f.write("-" * 40 + "\n")
            f.write("- MIMIC-IV 데이터를 사용한 48시간 사망률 예측 모델링 완료\n")
            f.write("- 하이퍼파라미터 튜닝을 통한 최적화 수행\n")
            f.write("- 모든 모델과 결과가 재현 가능하도록 저장됨\n")
            f.write(f"- 최종 모델: {self.output_path}\n")
        
        print(f"✅ 최종 리포트: {summary_file}")

def main():
    """메인 실행 함수"""
    print("🎯 하이퍼파라미터 튜닝 시작")
    print("=" * 60)
    
    # 프로젝트 루트 디렉토리 찾기
    project_root = Path(__file__).parent.parent
    
    # 경로 설정 - 상대 경로 사용
    modeling_results_path = project_root / "dataset" / "4_modeling"
    resampling_path = project_root / "dataset" / "3_resampled"
    output_path = project_root / "dataset" / "5_final_models"
    
    # 튜너 초기화
    tuner = HyperparameterTuner(modeling_results_path, resampling_path, output_path)
    
    # 튜닝 실행
    tuner.run_tuning()
    
    # 결과 저장
    tuner.save_results()
    
    print("\n" + "=" * 60)
    print("✅ 하이퍼파라미터 튜닝 완료!")
    print(f"💾 최종 모델 저장 위치: {output_path}")
    print("🎉 MIMIC-IV 48시간 사망률 예측 프로젝트 완료!")
    print("=" * 60)

if __name__ == "__main__":
    main()
