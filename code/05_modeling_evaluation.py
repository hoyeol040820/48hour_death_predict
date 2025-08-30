#!/usr/bin/env python3
"""
모델링 및 평가
- 다양한 머신러닝 모델 학습 및 성능 평가
- 모델: Logistic Regression, SVC, Random Forest, XGBoost, LightGBM, Extra Trees
- 평가 지표: Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

# 머신러닝 라이브러리
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
import xgboost as xgb
import lightgbm as lgb

class ModelTrainer:
    def __init__(self, input_path, output_path):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        self.models_path = self.output_path / "models"
        self.models_path.mkdir(exist_ok=True)
        
        self.target_column = 'mortality_48h'
        self.results = []
        
    def define_models(self):
        """사용할 모델들 정의"""
        self.models = {
            'Logistic_Regression': LogisticRegression(
                random_state=42, max_iter=1000, class_weight='balanced'
            ),
            'SVC': SVC(
                random_state=42, probability=True, class_weight='balanced'
            ),
            'Random_Forest': RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                random_state=42, eval_metric='logloss', use_label_encoder=False
            ),
            'LightGBM': lgb.LGBMClassifier(
                random_state=42, verbosity=-1, class_weight='balanced'
            ),
            'Extra_Trees': ExtraTreesClassifier(
                n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1
            )
        }
        
        print(f"✅ 정의된 모델: {list(self.models.keys())}")
        
    def load_resampling_data(self, method='smote'):
        """리샘플링된 데이터 또는 원본 데이터 로드"""
        print(f"📂 {method.upper()} 데이터 로딩 중...")
        
        if method == 'original':
            # 리샘플링된 original 데이터 (3_resampled/original/)
            data_path = self.input_path / method
        else:
            # 리샘플링된 데이터
            data_path = self.input_path / method
        
        train_df = pd.read_csv(data_path / "mimic_mortality_train.csv")
        val_df = pd.read_csv(data_path / "mimic_mortality_validation.csv")
        test_df = pd.read_csv(data_path / "mimic_mortality_test.csv")
        
        print(f"✅ Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
        
        # 클래스 분포 출력
        train_mortality = train_df[self.target_column].mean()
        print(f"   Train 사망률: {train_mortality:.1%}")
        
        return train_df, val_df, test_df
    
    def preprocess_data(self, train_df, val_df, test_df):
        """데이터 전처리"""
        print("🔄 데이터 전처리 중...")
        
        # 불필요한 컬럼 제거 (ID 컬럼들)
        id_columns = ['subject_id', 'hadm_id', 'stay_id']
        existing_id_cols = [col for col in id_columns if col in train_df.columns]
        
        # 특성과 타겟 분리
        X_train = train_df.drop(columns=existing_id_cols + [self.target_column])
        y_train = train_df[self.target_column]
        
        X_val = val_df.drop(columns=existing_id_cols + [self.target_column])
        y_val = val_df[self.target_column]
        
        X_test = test_df.drop(columns=existing_id_cols + [self.target_column])
        y_test = test_df[self.target_column]
        
        # 문자열 컬럼 인코딩
        categorical_columns = X_train.select_dtypes(include=['object']).columns
        label_encoders = {}
        
        if len(categorical_columns) > 0:
            print(f"   문자열 컬럼 인코딩: {list(categorical_columns)}")
            
            for col in categorical_columns:
                le = LabelEncoder()
                
                # Train set으로만 fit (데이터 누수 방지)
                le.fit(X_train[col].astype(str))
                
                # Transform - 처리되지 않은 값은 -1로 처리
                def safe_transform(series, encoder):
                    """안전한 변환: 새로운 카테고리는 -1로 처리"""
                    result = series.astype(str).copy()
                    mask = result.isin(encoder.classes_)
                    
                    # 알려진 값만 변환
                    result_encoded = np.full(len(result), -1, dtype=int)
                    result_encoded[mask] = encoder.transform(result[mask])
                    
                    return result_encoded
                
                X_train[col] = le.transform(X_train[col].astype(str))
                X_val[col] = safe_transform(X_val[col], le)
                X_test[col] = safe_transform(X_test[col], le)
                
                label_encoders[col] = le
        
        # 결측치 처리 (단순 평균 대체)
        if X_train.isnull().any().any():
            print("   결측치 평균 대체")
            for col in X_train.columns:
                if X_train[col].isnull().any():
                    mean_val = X_train[col].mean()
                    X_train[col].fillna(mean_val, inplace=True)
                    X_val[col].fillna(mean_val, inplace=True)
                    X_test[col].fillna(mean_val, inplace=True)
        
        # 스케일링 (SVC용)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"✅ 전처리 완료 - 특성 수: {X_train.shape[1]}")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'X_train_scaled': X_train_scaled,
            'X_val_scaled': X_val_scaled,
            'X_test_scaled': X_test_scaled,
            'scaler': scaler,
            'label_encoders': label_encoders
        }
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """평가 지표 계산"""
        # 혼동 행렬
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # 기본 지표
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # 특이도 (Specificity)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # AUC 지표
        if len(np.unique(y_true)) > 1:
            roc_auc = roc_auc_score(y_true, y_pred_proba)
            pr_auc = average_precision_score(y_true, y_pred_proba)
        else:
            roc_auc = 0
            pr_auc = 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'confusion_matrix': {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}
        }
    
    def train_and_evaluate_model(self, model_name, model, data, resampling_method):
        """개별 모델 학습 및 평가"""
        print(f"🔄 {resampling_method.upper()}-{model_name} 학습 중...")
        
        try:
            # 모델별 데이터 선택
            if model_name == 'SVC':
                X_train, X_val, X_test = data['X_train_scaled'], data['X_val_scaled'], data['X_test_scaled']
            else:
                X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
            
            y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']
            
            # 클래스 불균형 처리 비활성화 (리샘플링으로 이미 균형 조정됨)
            # if model_name == 'XGBoost':
            #     pos_ratio = (y_train == 0).sum() / (y_train == 1).sum()
            #     model.set_params(scale_pos_weight=pos_ratio)
            #     print(f"   XGBoost scale_pos_weight: {pos_ratio:.2f}")
            # elif model_name == 'LightGBM':
            #     if resampling_method == 'original':
            #         # 원본 데이터의 경우 클래스 불균형이 심함
            #         model.set_params(is_unbalance=True, class_weight='balanced')
            #     else:
            #         model.set_params(is_unbalance=True)
            
            # 모든 모델에서 기본 설정 사용 (리샘플링에 의존)
            
            # 모델 학습
            model.fit(X_train, y_train)
            
            # 모델 저장
            model_filename = f"{resampling_method}_{model_name}.pkl"
            model_path = self.models_path / model_filename
            joblib.dump(model, model_path)
            
            # 검증 세트 예측 (모델 선택용)
            y_val_pred = model.predict(X_val)
            
            if hasattr(model, 'predict_proba'):
                y_val_proba = model.predict_proba(X_val)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_val_proba = model.decision_function(X_val)
            else:
                y_val_proba = y_val_pred.astype(float)
            
            # 성능 계산 (Validation 기준)
            val_metrics = self.calculate_metrics(y_val, y_val_pred, y_val_proba)
            
            result = {
                'resampling_method': resampling_method,
                'model_name': model_name,
                'val_metrics': val_metrics,  # Validation 기준으로 변경
                'model_path': str(model_path),
                'feature_count': X_train.shape[1],
                'train_samples': len(y_train)
            }
            
            print(f"✅ {resampling_method.upper()}-{model_name} 완료")
            print(f"   Validation ROC-AUC: {val_metrics['roc_auc']:.4f}")
            print(f"   Validation F1-Score: {val_metrics['f1_score']:.4f}")
            
            return result
            
        except Exception as e:
            print(f"❌ {resampling_method.upper()}-{model_name} 실패: {e}")
            return {
                'resampling_method': resampling_method,
                'model_name': model_name,
                'error': str(e),
                'val_metrics': None
            }
    
    def run_experiments(self):
        """모든 모델 실험 실행"""
        print("🚀 모델링 실험 시작")
        print("=" * 60)
        
        # 모델 정의
        self.define_models()
        
        resampling_methods = ['original', 'smote', 'downsampling']
        
        for method in resampling_methods:
            print(f"\n📊 {method.upper()} 데이터셋으로 실험 중...")
            
            # 데이터 로드
            try:
                train_df, val_df, test_df = self.load_resampling_data(method)
            except Exception as e:
                print(f"❌ {method} 데이터 로드 실패: {e}")
                continue
            
            # 데이터 전처리
            data = self.preprocess_data(train_df, val_df, test_df)
            
            # 각 모델 학습
            for model_name, model in self.models.items():
                # 모델 복사 (재사용을 위해)
                from sklearn.base import clone
                model_copy = clone(model)
                
                result = self.train_and_evaluate_model(
                    model_name, model_copy, data, method
                )
                self.results.append(result)
        
        print(f"\n✅ 전체 실험 완료! 총 {len(self.results)}개 모델")
        print("   - Original(원본): 클래스 불균형 상태")
        print("   - SMOTE: 소수 클래스 오버샘플링")  
        print("   - Downsampling: 다수 클래스 다운샘플링")
    
    def save_results(self):
        """결과 저장"""
        print("\n💾 결과 저장 중...")
        
        # 성공한 실험만 필터링
        successful_results = [r for r in self.results if 'error' not in r and r['val_metrics'] is not None]
        
        if not successful_results:
            print("❌ 저장할 성공적인 결과가 없습니다.")
            return
        
        # 결과 DataFrame 생성
        rows = []
        for result in successful_results:
            metrics = result['val_metrics']
            row = {
                'resampling_method': result['resampling_method'],
                'model_name': result['model_name'],
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'specificity': metrics['specificity'],
                'roc_auc': metrics['roc_auc'],
                'pr_auc': metrics['pr_auc'],
                'feature_count': result['feature_count'],
                'train_samples': result['train_samples'],
                'model_path': result['model_path']
            }
            rows.append(row)
        
        results_df = pd.DataFrame(rows)
        
        # 성능 순 정렬 (ROC-AUC 기준)
        results_df = results_df.sort_values('roc_auc', ascending=False)
        
        # CSV 저장
        results_csv = self.output_path / "modeling_results.csv"
        results_df.to_csv(results_csv, index=False)
        print(f"✅ 결과 저장: {results_csv}")
        
        # 요약 리포트
        self.create_summary_report(results_df)
        
        return results_df
    
    def create_summary_report(self, results_df):
        """요약 리포트 생성"""
        summary_file = self.output_path / "modeling_summary.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("MIMIC-IV 48시간 사망률 예측 모델링 결과\n")
            f.write("=" * 60 + "\n")
            f.write(f"실험 일시: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("실험 설정:\n")
            f.write("- 모델: Logistic Regression, SVC, Random Forest, XGBoost, LightGBM, Extra Trees\n")
            f.write("- 리샘플링: Original(원본), SMOTE, Downsampling\n")
            f.write("- 평가: Validation 세트 기준 (모델 선택용)\n")
            f.write("- 지표: ROC-AUC, F1-Score, Precision, Recall, Accuracy\n\n")
            
            # 최고 성능 모델
            best_model = results_df.iloc[0]
            f.write("🏆 최고 성능 모델:\n")
            f.write(f"- 모델: {best_model['resampling_method'].upper()}-{best_model['model_name']}\n")
            f.write(f"- ROC-AUC: {best_model['roc_auc']:.4f}\n")
            f.write(f"- F1-Score: {best_model['f1_score']:.4f}\n")
            f.write(f"- Precision: {best_model['precision']:.4f}\n")
            f.write(f"- Recall: {best_model['recall']:.4f}\n\n")
            
            # 리샘플링별 최고 성능
            f.write("리샘플링별 최고 성능:\n")
            for method in results_df['resampling_method'].unique():
                method_best = results_df[results_df['resampling_method'] == method].iloc[0]
                f.write(f"- {method.upper()}: {method_best['model_name']} (ROC-AUC: {method_best['roc_auc']:.4f})\n")
            
            f.write(f"\n전체 결과: {len(results_df)}개 모델\n")
            f.write(f"저장된 모델: {self.models_path}\n")
            f.write(f"다음 단계: 하이퍼파라미터 튜닝 (07_hyperparameter_tuning.py)\n")
        
        print(f"✅ 요약 리포트: {summary_file}")

def main():
    """메인 실행 함수"""
    print("🤖 모델링 및 평가 시작")
    print("=" * 60)
    
    # 프로젝트 루트 디렉토리 찾기
    project_root = Path(__file__).parent.parent
    
    # 경로 설정 - 상대 경로 사용
    input_path = project_root / "dataset" / "3_resampled"
    output_path = project_root / "dataset" / "4_modeling"
    
    # 모델 트레이너 초기화
    trainer = ModelTrainer(input_path, output_path)
    
    # 모든 실험 실행
    trainer.run_experiments()
    
    # 결과 저장
    results_df = trainer.save_results()
    
    if results_df is not None:
        print("\n" + "=" * 60)
        print("✅ 모델링 완료!")
        print(f"🏆 최고 성능: {results_df.iloc[0]['resampling_method'].upper()}-{results_df.iloc[0]['model_name']}")
        print(f"📊 Validation ROC-AUC: {results_df.iloc[0]['roc_auc']:.4f}")
        print(f"💾 저장 위치: {output_path}")
        print(f"📊 총 실험: {len(results_df)}개 모델 (Original + SMOTE + Downsampling)")
        print("💡 다음 단계: 06_hyperparameter_tuning.py 실행")
        print("=" * 60)

if __name__ == "__main__":
    main()
