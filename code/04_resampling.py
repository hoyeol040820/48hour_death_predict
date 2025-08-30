#!/usr/bin/env python3
"""
리샘플링
- 클래스 불균형 해결을 위한 리샘플링 기법 적용
- SMOTE (Synthetic Minority Oversampling Technique)
- Downsampling (다운샘플링)
- Train 세트에만 적용, Validation/Test는 원본 분포 유지
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class DataResampler:
    def __init__(self, input_path, output_path):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.target_column = 'mortality_48h'
        
    def create_output_directories(self):
        """출력 디렉토리 생성"""
        self.original_path = self.output_path / "original"
        self.smote_path = self.output_path / "smote"
        self.downsampling_path = self.output_path / "downsampling"
        
        self.original_path.mkdir(parents=True, exist_ok=True)
        self.smote_path.mkdir(parents=True, exist_ok=True)
        self.downsampling_path.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Original 출력: {self.original_path}")
        print(f"📁 SMOTE 출력: {self.smote_path}")
        print(f"📁 Downsampling 출력: {self.downsampling_path}")
    
    def load_split_data(self):
        """분할된 데이터 로드"""
        print("📂 분할된 데이터 로딩 중...")
        
        train_df = pd.read_csv(self.input_path / "mimic_mortality_train.csv")
        val_df = pd.read_csv(self.input_path / "mimic_mortality_validation.csv")
        test_df = pd.read_csv(self.input_path / "mimic_mortality_test.csv")
        
        print(f"✅ Train: {train_df.shape}")
        print(f"✅ Validation: {val_df.shape}")
        print(f"✅ Test: {test_df.shape}")
        
        return train_df, val_df, test_df
    
    def analyze_class_distribution(self, df, title="클래스 분포"):
        """클래스 분포 분석"""
        print(f"\n📊 {title}")
        
        if self.target_column in df.columns:
            dist = df[self.target_column].value_counts().sort_index()
            total = len(df)
            
            print(f"   - 총 샘플: {total:,}개")
            print(f"   - 생존 (0): {dist[0]:,}개 ({dist[0]/total:.1%})")
            print(f"   - 사망 (1): {dist[1]:,}개 ({dist[1]/total:.1%})")
            print(f"   - 불균형 비율: {dist[0]/dist[1]:.1f}:1")
            
            return dist
        
        return None
    
    def apply_downsampling(self, train_df):
        """다운샘플링 적용"""
        print("\n⬇️ 다운샘플링 적용 중...")
        
        # 클래스별 분리
        majority_class = train_df[train_df[self.target_column] == 0]  # 생존
        minority_class = train_df[train_df[self.target_column] == 1]  # 사망
        
        print(f"   원본 - 생존: {len(majority_class):,}개, 사망: {len(minority_class):,}개")
        
        # 다수 클래스를 소수 클래스 수에 맞춰 다운샘플링
        majority_downsampled = resample(
            majority_class,
            replace=False,
            n_samples=len(minority_class),
            random_state=42
        )
        
        # 균형 잡힌 데이터셋 생성
        balanced_df = pd.concat([majority_downsampled, minority_class])
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"   ✅ 다운샘플링 후 - 총: {len(balanced_df):,}개")
        self.analyze_class_distribution(balanced_df, "다운샘플링 후 분포")
        
        return balanced_df
    
    def apply_smote(self, train_df):
        """SMOTE 적용"""
        print("\n⬆️ SMOTE 적용 중...")
        
        # 특성과 타겟 분리
        X = train_df.drop(columns=[self.target_column])
        y = train_df[self.target_column]
        
        print(f"   원본 클래스 분포: {Counter(y)}")
        
        # 문자열 컬럼 인코딩
        label_encoders = {}
        X_encoded = X.copy()
        
        categorical_columns = X.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            print(f"   문자열 컬럼 인코딩: {list(categorical_columns)}")
            
            for col in categorical_columns:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
        
        # 결측치 제거 (SMOTE는 결측치를 처리할 수 없음)
        if X_encoded.isnull().any().any():
            print("   ⚠️ SMOTE 적용 전 결측치 제거")
            missing_mask = X_encoded.isnull().any(axis=1)
            X_encoded = X_encoded[~missing_mask]
            y = y[~missing_mask]
            print(f"   결측치 제거 후: {len(X_encoded):,}개 샘플")
        
        # SMOTE 적용
        try:
            smote = SMOTE(
                random_state=42,
                k_neighbors=min(5, (y == 1).sum() - 1),  # 소수 클래스 크기에 맞춰 조정
                sampling_strategy='auto'
            )
            
            X_resampled, y_resampled = smote.fit_resample(X_encoded, y)
            print(f"   SMOTE 후 클래스 분포: {Counter(y_resampled)}")
            
        except Exception as e:
            print(f"   ❌ SMOTE 적용 실패: {e}")
            print("   원본 데이터 반환")
            return train_df
        
        # 데이터프레임 재구성
        balanced_df = pd.DataFrame(X_resampled, columns=X_encoded.columns)
        
        # 문자열 컬럼 디코딩
        for col, le in label_encoders.items():
            # 반올림 후 클리핑으로 유효한 범위 보장
            balanced_df[col] = np.round(balanced_df[col]).astype(int)
            balanced_df[col] = np.clip(balanced_df[col], 0, len(le.classes_) - 1)
            balanced_df[col] = le.inverse_transform(balanced_df[col])
        
        # 타겟 추가
        balanced_df[self.target_column] = y_resampled
        
        print(f"   ✅ SMOTE 후 - 총: {len(balanced_df):,}개")
        self.analyze_class_distribution(balanced_df, "SMOTE 후 분포")
        
        return balanced_df
    
    def save_resampled_data(self, method, train_resampled, val_df, test_df):
        """리샘플링된 데이터 저장"""
        print(f"\n💾 {method} 데이터 저장 중...")
        
        # 출력 경로 설정
        if method == "Original":
            output_dir = self.original_path
        elif method == "SMOTE":
            output_dir = self.smote_path
        else:  # Downsampling
            output_dir = self.downsampling_path
        
        # 파일 저장
        datasets = {
            'train': train_resampled,
            'validation': val_df,
            'test': test_df
        }
        
        saved_files = {}
        
        for name, df in datasets.items():
            filename = f"mimic_mortality_{name}.csv"
            filepath = output_dir / filename
            df.to_csv(filepath, index=False)
            saved_files[name] = filepath
            print(f"   ✅ {name}: {len(df):,}개 → {filepath}")
        
        return saved_files
    
    def save_resampling_summary(self, train_original, train_smote, train_down, val_df, test_df):
        """리샘플링 요약 저장"""
        summary_file = self.output_path / "resampling_summary.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("리샘플링 요약\n")
            f.write("=" * 50 + "\n")
            f.write(f"처리 일시: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("리샘플링 방법:\n")
            f.write("1. Original (원본)\n")
            f.write("   - 리샘플링 없이 원본 데이터 그대로 사용\n")
            f.write("2. SMOTE (Synthetic Minority Oversampling Technique)\n")
            f.write("   - 소수 클래스 합성 데이터 생성으로 균형 맞춤\n")
            f.write("3. Downsampling (다운샘플링)\n")
            f.write("   - 다수 클래스를 소수 클래스 수준으로 축소\n\n")
            
            f.write("적용 원칙:\n")
            f.write("- Train 세트에만 리샘플링 적용\n")
            f.write("- Validation/Test 세트는 원본 분포 유지 (모든 방법에서 동일)\n")
            f.write("- 데이터 누수 방지를 위한 올바른 분할 후 처리\n\n")
            
            # 데이터 크기 비교
            f.write("데이터 크기 비교:\n")
            f.write(f"- Original Train: {len(train_original):,}개\n")
            f.write(f"- SMOTE Train: {len(train_smote):,}개\n")
            f.write(f"- Downsampling Train: {len(train_down):,}개\n")
            f.write(f"- Validation: {len(val_df):,}개 (모든 방법에서 동일)\n")
            f.write(f"- Test: {len(test_df):,}개 (모든 방법에서 동일)\n\n")
            
            # 클래스 분포 비교
            f.write("48시간 사망률 비교:\n")
            datasets = [
                ("Original Train", train_original),
                ("SMOTE Train", train_smote), 
                ("Downsampling Train", train_down),
                ("Validation", val_df),
                ("Test", test_df)
            ]
            
            for name, df in datasets:
                if self.target_column in df.columns:
                    mortality_rate = df[self.target_column].mean()
                    mortality_count = df[self.target_column].sum()
                    f.write(f"- {name}: {mortality_rate:.1%} ({mortality_count:,}명)\n")
            
            f.write(f"\n저장 위치:\n")
            f.write(f"- Original: {self.original_path}\n")
            f.write(f"- SMOTE: {self.smote_path}\n")
            f.write(f"- Downsampling: {self.downsampling_path}\n")
            
            f.write(f"\n다음 단계: 모델링 및 평가 (05_modeling_evaluation.py)\n")
        
        print(f"✅ 리샘플링 요약: {summary_file}")

def main():
    """메인 실행 함수"""
    print("⚖️ 리샘플링 시작")
    print("=" * 60)
    
    # 프로젝트 루트 디렉토리 찾기
    project_root = Path(__file__).parent.parent
    
    # 경로 설정 - 상대 경로 사용
    input_path = project_root / "dataset" / "2_split"
    output_path = project_root / "dataset" / "3_resampled"
    
    # 리샘플러 초기화
    resampler = DataResampler(input_path, output_path)
    resampler.create_output_directories()
    
    # 1. 분할된 데이터 로드
    train_df, val_df, test_df = resampler.load_split_data()
    
    # 2. 원본 train 분포 분석
    resampler.analyze_class_distribution(train_df, "원본 Train 분포")
    
    # 3. Original 데이터 저장 (리샘플링 없음)
    original_files = resampler.save_resampled_data("Original", train_df.copy(), val_df, test_df)
    
    # 4. SMOTE 적용
    train_smote = resampler.apply_smote(train_df.copy())
    
    # 5. 다운샘플링 적용  
    train_downsampled = resampler.apply_downsampling(train_df.copy())
    
    # 6. 리샘플링된 데이터 저장
    smote_files = resampler.save_resampled_data("SMOTE", train_smote, val_df, test_df)
    down_files = resampler.save_resampled_data("Downsampling", train_downsampled, val_df, test_df)
    
    # 7. 요약 정보 저장
    resampler.save_resampling_summary(train_df, train_smote, train_downsampled, val_df, test_df)
    
    print("\n" + "=" * 60)
    print("✅ 리샘플링 완료!")
    print(f"📊 결과:")
    print(f"   - Original Train: {len(train_df):,}개 (리샘플링 없음)")
    print(f"   - SMOTE Train: {len(train_smote):,}개")
    print(f"   - Downsampling Train: {len(train_downsampled):,}개")
    print(f"📁 저장 위치:")
    print(f"   - Original: {output_path}/original/")
    print(f"   - SMOTE: {output_path}/smote/")
    print(f"   - Downsampling: {output_path}/downsampling/")
    print("💡 다음 단계: 05_modeling_evaluation.py 실행")
    print("=" * 60)

if __name__ == "__main__":
    main()
