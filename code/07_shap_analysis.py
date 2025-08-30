#!/usr/bin/env python3
"""
SHAP Analysis
- TreeSHAP으로 최고 성능 모델들의 feature importance 분석
- Global/Local 해석성 제공
- 상위 모델들의 feature interaction 분석
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
import json
warnings.filterwarnings('ignore')

# SHAP 패키지 확인
try:
    import shap
    # SHAP 설정
    shap.initjs()
    SHAP_AVAILABLE = True
except ImportError:
    print("⚠️ SHAP 패키지가 설치되지 않았습니다.")
    print("   다음 명령으로 설치해주세요: pip install shap")
    SHAP_AVAILABLE = False

class SHAPAnalyzer:
    def __init__(self, modeling_results_path, data_path, output_path, tuning_results_path=None):
        self.modeling_results_path = Path(modeling_results_path)
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.tuning_results_path = Path(tuning_results_path) if tuning_results_path else None
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # SHAP 결과 저장용
        self.shap_results = {}
        
    def load_modeling_results(self):
        """모델링 결과 로드"""
        print("📊 모델링 결과 로드 중...")
        
        results_df = pd.read_csv(self.modeling_results_path)
        print(f"✅ 총 {len(results_df)}개 모델 결과 로드")
        
        return results_df
    
    def load_tuning_results(self):
        """하이퍼파라미터 튜닝 결과 로드"""
        if self.tuning_results_path and self.tuning_results_path.exists():
            print("🎯 하이퍼파라미터 튜닝 결과 로드 중...")
            tuning_df = pd.read_csv(self.tuning_results_path)
            print(f"✅ 총 {len(tuning_df)}개 튜닝된 모델 결과 로드")
            return tuning_df
        else:
            print("⚠️ 하이퍼파라미터 튜닝 결과 파일이 없습니다.")
            return None
    
    def select_top_models(self, results_df, top_k=5):
        """특정 모델들 선택 (XGBoost, LightGBM의 original, SMOTE만)"""
        print("🏆 XGBoost, LightGBM의 original, SMOTE 모델들 선택...")
        
        # XGBoost, LightGBM만 필터링
        target_models = ['XGBoost', 'LightGBM']
        target_resampling = ['original', 'smote']
        
        # 조건에 맞는 모델들 필터링
        filtered_results = results_df[
            (results_df['model_name'].isin(target_models)) & 
            (results_df['resampling_method'].isin(target_resampling))
        ].copy()
        
        # ROC-AUC 기준 정렬
        top_models = filtered_results.sort_values('roc_auc', ascending=False)
        
        print("선택된 모델들:")
        for idx, row in top_models.iterrows():
            print(f"  {row['resampling_method'].upper()}-{row['model_name']}: ROC-AUC {row['roc_auc']:.4f}")
        
        return top_models
    
    def select_tuned_models(self, tuning_df):
        """튜닝된 모델들 선택 (XGBoost, LightGBM의 original, SMOTE만)"""
        if tuning_df is None:
            return None
            
        print("🎯 튜닝된 XGBoost, LightGBM의 original, SMOTE 모델들 선택...")
        
        # base_model과 resampling_method 기준으로 필터링
        target_models = ['XGBoost', 'LightGBM']
        target_resampling = ['original', 'smote']
        
        filtered_results = tuning_df[
            (tuning_df['base_model'].isin(target_models)) & 
            (tuning_df['resampling_method'].isin(target_resampling))
        ].copy()
        
        # ROC-AUC 기준 정렬
        tuned_models = filtered_results.sort_values('roc_auc', ascending=False)
        
        print("선택된 튜닝된 모델들:")
        for idx, row in tuned_models.iterrows():
            print(f"  {row['resampling_method'].upper()}-{row['base_model']}: ROC-AUC {row['roc_auc']:.4f} (튜닝됨)")
        
        return tuned_models
    
    def load_model_and_data(self, model_info):
        """모델과 해당 데이터 로드"""
        model_path = model_info['model_path']
        resampling_method = model_info['resampling_method']
        # 튜닝된 모델인 경우 base_model 사용, 아니면 model_name 사용
        model_name = model_info.get('base_model', model_info.get('model_name', 'Unknown'))
        
        print(f"🔄 {resampling_method.upper()}-{model_name} 로딩 중...")
        
        # 모델 로드
        try:
            model = joblib.load(model_path)
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            return None, None, None, None
        
        # 데이터 로드
        data_base_path = self.data_path / resampling_method
        
        try:
            train_df = pd.read_csv(data_base_path / "mimic_mortality_train.csv")
            test_df = pd.read_csv(data_base_path / "mimic_mortality_test.csv")
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            return None, None, None, None
        
        # 전처리 (05_modeling_evaluation.py와 동일한 방식)
        target_column = 'mortality_48h'
        id_columns = ['subject_id', 'hadm_id', 'stay_id']
        existing_id_cols = [col for col in id_columns if col in train_df.columns]
        
        X_train = train_df.drop(columns=existing_id_cols + [target_column])
        X_test = test_df.drop(columns=existing_id_cols + [target_column])
        y_test = test_df[target_column]
        
        # 문자열 컬럼 인코딩 (Train으로만 fit)
        from sklearn.preprocessing import LabelEncoder
        
        categorical_columns = X_train.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            for col in categorical_columns:
                le = LabelEncoder()
                le.fit(X_train[col].astype(str))
                
                # Safe transform function
                def safe_transform(series, encoder):
                    result = series.astype(str).copy()
                    mask = result.isin(encoder.classes_)
                    result_encoded = np.full(len(result), -1, dtype=int)
                    result_encoded[mask] = encoder.transform(result[mask])
                    return result_encoded
                
                X_train[col] = le.transform(X_train[col].astype(str))
                X_test[col] = safe_transform(X_test[col], le)
        
        # 결측치 처리
        if X_train.isnull().any().any():
            for col in X_train.columns:
                if X_train[col].isnull().any():
                    mean_val = X_train[col].mean()
                    X_train[col].fillna(mean_val, inplace=True)
                    X_test[col].fillna(mean_val, inplace=True)
        
        print(f"✅ 데이터 로드 완료: Train {X_train.shape}, Test {X_test.shape}")
        return model, X_train, X_test, y_test
    
    def calculate_shap_values(self, model, X_train, X_test, model_info, sample_size=1000):
        """SHAP 값 계산"""
        base_model_name = model_info.get('base_model', model_info.get('model_name', 'Unknown'))
        model_name = f"{model_info['resampling_method']}_{base_model_name}"
        print(f"🔍 {model_name} SHAP 분석 중...")
        
        # 샘플 크기 조정 (메모리 절약)
        if len(X_train) > sample_size:
            background_sample = X_train.sample(n=sample_size, random_state=42)
        else:
            background_sample = X_train.copy()
        
        if len(X_test) > sample_size:
            test_sample = X_test.sample(n=sample_size, random_state=42)
        else:
            test_sample = X_test.copy()
        
        try:
            # TreeExplainer 사용
            explainer = shap.TreeExplainer(model)
            
            # SHAP values 계산
            shap_values = explainer.shap_values(test_sample)
            
            print(f"   Raw SHAP values type: {type(shap_values)}")
            if isinstance(shap_values, list):
                print(f"   SHAP values list length: {len(shap_values)}")
                for i, sv in enumerate(shap_values):
                    print(f"   SHAP values[{i}] shape: {np.array(sv).shape}")
            else:
                print(f"   SHAP values shape: {np.array(shap_values).shape}")
            
            # XGBoost/LightGBM/RandomForest의 경우 처리
            if isinstance(shap_values, list):
                if len(shap_values) == 2:
                    # 이진 분류의 경우 positive class (index 1) 사용
                    shap_values = shap_values[1]
                else:
                    # 단일 클래스인 경우 첫 번째 사용
                    shap_values = shap_values[0]
            
            # numpy array로 변환
            shap_values = np.array(shap_values)
            
            # 차원 확인
            if shap_values.ndim != 2:
                print(f"   ⚠️ 예상치 못한 SHAP values 차원: {shap_values.shape}")
                # 2차원으로 만들기
                if shap_values.ndim == 1:
                    shap_values = shap_values.reshape(1, -1)
                elif shap_values.ndim > 2:
                    # 첫 두 차원만 사용
                    shap_values = shap_values.reshape(shap_values.shape[0], -1)
            
            # 기댓값
            expected_value = explainer.expected_value
            if isinstance(expected_value, list):
                expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
            
            print(f"✅ SHAP 값 계산 완료: {shap_values.shape}")
            
            # 결과 저장
            shap_result = {
                'model_name': model_name,
                'shap_values': shap_values,
                'test_sample': test_sample,
                'expected_value': expected_value,
                'feature_names': list(X_test.columns),
                'model_info': model_info
            }
            
            self.shap_results[model_name] = shap_result
            return shap_result
            
        except Exception as e:
            print(f"❌ SHAP 분석 실패: {e}")
            return None
    
    def analyze_feature_importance(self, shap_result):
        """Feature importance 분석"""
        model_name = shap_result['model_name']
        shap_values = shap_result['shap_values']
        feature_names = shap_result['feature_names']
        
        print(f"📊 {model_name} Feature Importance 분석...")
        print(f"   SHAP values shape: {shap_values.shape}")
        
        # Global feature importance (평균 절댓값)
        feature_importance = np.mean(np.abs(shap_values), axis=0)
        
        # 차원 확인 및 평탄화
        if feature_importance.ndim > 1:
            feature_importance = feature_importance.flatten()
        
        print(f"   Feature importance shape: {feature_importance.shape}")
        print(f"   Feature names count: {len(feature_names)}")
        
        # 길이 일치 확인
        if len(feature_importance) != len(feature_names):
            min_len = min(len(feature_importance), len(feature_names))
            feature_importance = feature_importance[:min_len]
            feature_names = feature_names[:min_len]
            print(f"   ⚠️ 길이 불일치 - {min_len}개로 조정")
        
        # DataFrame으로 정리
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        # 상위 20개 특성
        top_features = importance_df.head(20)
        
        print(f"상위 10개 특성:")
        for idx, row in top_features.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return importance_df, top_features
    
    def save_shap_results(self):
        """SHAP 분석 결과 저장"""
        print("\n💾 SHAP 분석 결과 저장 중...")
        
        # 각 모델별 결과 저장
        for model_name, shap_result in self.shap_results.items():
            # Feature importance 분석
            importance_df, top_features = self.analyze_feature_importance(shap_result)
            
            # CSV 저장
            importance_path = self.output_path / f"{model_name}_feature_importance.csv"
            importance_df.to_csv(importance_path, index=False)
            
            # SHAP 값 저장 (상위 특성만)
            top_feature_names = top_features['feature'].tolist()
            shap_values = shap_result['shap_values']
            test_sample = shap_result['test_sample']
            
            # 상위 특성들의 SHAP 값만 저장
            top_indices = [shap_result['feature_names'].index(f) for f in top_feature_names if f in shap_result['feature_names']]
            top_shap_values = shap_values[:, top_indices]
            
            # 저장용 데이터
            shap_data = {
                'model_name': model_name,
                'feature_names': top_feature_names,
                'shap_values': top_shap_values.tolist(),
                'expected_value': float(shap_result['expected_value']),
                'model_info': {
                    'resampling_method': shap_result['model_info']['resampling_method'],
                    'model_name': shap_result['model_info']['model_name'],
                    'roc_auc': float(shap_result['model_info']['roc_auc'])
                }
            }
            
            # JSON으로 저장
            shap_path = self.output_path / f"{model_name}_shap_values.json"
            with open(shap_path, 'w') as f:
                json.dump(shap_data, f, indent=2)
            
            print(f"✅ {model_name} 결과 저장: {importance_path}")
        
        # 전체 요약 생성
        self.create_summary_report()
    
    def create_summary_report(self):
        """SHAP 분석 요약 보고서"""
        summary_path = self.output_path / "shap_analysis_summary.txt"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("MIMIC-IV 48시간 사망률 예측 - SHAP 분석 결과\n")
            f.write("=" * 60 + "\n")
            f.write(f"분석 일시: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("분석 개요:\n")
            f.write("- 방법: TreeSHAP (Tree-based 모델 전용)\n") 
            f.write("- 대상: 상위 성능 Tree-based 모델들\n")
            f.write("- 해석성: Global feature importance + Local explanations\n\n")
            
            f.write(f"분석 모델 ({len(self.shap_results)}개):\n")
            for model_name, shap_result in self.shap_results.items():
                model_info = shap_result['model_info']
                f.write(f"- {model_name}: ROC-AUC {model_info['roc_auc']:.4f}\n")
            
            f.write("\n상위 특성 요약 (모델별 상위 5개):\n")
            f.write("-" * 40 + "\n")
            
            for model_name, shap_result in self.shap_results.items():
                importance_df, _ = self.analyze_feature_importance(shap_result)
                top_5 = importance_df.head(5)
                
                f.write(f"\n{model_name}:\n")
                for idx, row in top_5.iterrows():
                    f.write(f"  {idx+1}. {row['feature']}: {row['importance']:.4f}\n")
            
            f.write(f"\n저장 파일:\n")
            f.write(f"- Feature importance: *_feature_importance.csv\n")
            f.write(f"- SHAP values: *_shap_values.json\n")
            f.write(f"- 시각화: figure_generator.py 실행 시 생성\n")
        
        print(f"✅ 요약 보고서: {summary_path}")

    def run_analysis(self, top_k=5):
        """전체 SHAP 분석 실행 (튜닝된 모델들만)"""
        print("🚀 SHAP 분석 시작 (하이퍼파라미터 튜닝된 모델들)")
        print("=" * 60)
        
        # SHAP 패키지 가용성 확인
        if not SHAP_AVAILABLE:
            print("❌ SHAP 패키지가 필요합니다. 설치 후 다시 실행해주세요.")
            return
        
        # 1. 하이퍼파라미터 튜닝 결과 로드
        tuning_df = self.load_tuning_results()
        if tuning_df is None:
            print("❌ 하이퍼파라미터 튜닝 결과를 찾을 수 없습니다.")
            return
        
        # 2. 튜닝된 모델들 선택 (XGBoost, LightGBM의 original, SMOTE만)
        selected_models = self.select_tuned_models(tuning_df)
        if selected_models is None or len(selected_models) == 0:
            print("❌ 분석할 튜닝된 모델이 없습니다.")
            return
        
        # 3. 각 모델별 SHAP 분석
        for idx, (_, model_info) in enumerate(selected_models.iterrows()):
            model_display_name = f"{model_info['resampling_method']}-{model_info['base_model']}"
            print(f"\n📊 모델 {idx+1}/{len(selected_models)}: {model_display_name} (튜닝됨)")
            
            try:
                # 모델과 데이터 로드
                model, X_train, X_test, y_test = self.load_model_and_data(model_info)
                
                if model is None:
                    print(f"⚠️ 모델 로드 실패 - 건너뜀")
                    continue
                
                # SHAP 분석
                shap_result = self.calculate_shap_values(model, X_train, X_test, model_info)
                
                if shap_result is None:
                    print(f"⚠️ SHAP 분석 실패 - 건너뜀")
                    continue
                    
            except Exception as e:
                print(f"❌ 모델 처리 중 오류: {e}")
                model_name = model_info.get('base_model', model_info.get('model_name', 'Unknown'))
                print(f"⚠️ {model_info['resampling_method']}-{model_name} 건너뜀")
                continue
        
        # 4. 결과 저장
        if self.shap_results:
            self.save_shap_results()
            print(f"\n✅ SHAP 분석 완료! 총 {len(self.shap_results)}개 모델 분석")
        else:
            print("\n❌ 분석된 모델이 없습니다.")

def main():
    """메인 실행 함수"""
    print("🔍 SHAP 분석 시작")
    print("=" * 60)
    
    # 프로젝트 루트 디렉토리 찾기
    project_root = Path(__file__).parent.parent
    
    # 경로 설정 - 상대 경로 사용
    modeling_results_path = project_root / "dataset" / "4_modeling" / "modeling_results.csv"
    tuning_results_path = project_root / "dataset" / "5_final_models" / "tuning_results.csv"
    data_path = project_root / "dataset" / "3_resampled"
    output_path = project_root / "results" / "07_shap_analysis"
    
    # SHAP 분석기 초기화 (튜닝 결과 경로 포함)
    analyzer = SHAPAnalyzer(modeling_results_path, data_path, output_path, tuning_results_path)
    
    # 분석 실행
    analyzer.run_analysis(top_k=5)
    
    print("\n" + "=" * 60)
    print("✅ SHAP 분석 완료!")
    print(f"📁 결과 저장 위치: {output_path}")
    print("💡 다음 단계: figure_generator.py 실행하여 시각화 생성")
    print("=" * 60)

if __name__ == "__main__":
    main()
