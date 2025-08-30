#!/usr/bin/env python3
"""
데이터 정제
1. 24시간 이내 사망자 제거 (데이터 누수 방지)
2. 의학적으로 불가능한 이상치 제거
3. 결측치 처리 (50% 이상 결측 컬럼 드랍, 나머지 결측 행 드랍)
4. 7일 사망 여부 컬럼 삭제
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DataCleaner:
    def __init__(self, input_path, output_path):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        self.cleaning_log = []
        
    def load_data(self):
        """데이터 로드"""
        print("📂 데이터 로딩 중...")
        df = pd.read_csv(self.input_path)
        print(f"✅ 데이터 로드 완료: {df.shape}")
        
        self.cleaning_log.append(f"원본 데이터: {df.shape[0]:,}행 x {df.shape[1]}열")
        return df
    
    def remove_early_deaths(self, df):
        """24시간 이내 사망자 제거 (데이터 누수 방지)"""
        print("⏰ 24시간 이내 사망자 제거 중...")
        
        original_count = len(df)
        
        # los (재원기간) 기준으로 24시간(1일) 미만 환자 제거
        if 'los' in df.columns:
            df_filtered = df[df['los'] >= 1.0].copy()
            removed_count = original_count - len(df_filtered)
            
            print(f"✅ 24시간 미만 재원 환자 제거: {removed_count:,}명")
            print(f"   남은 환자: {len(df_filtered):,}명")
            
            self.cleaning_log.append(f"24시간 미만 재원 환자 제거: {removed_count:,}명")
        else:
            print("⚠️ 재원기간(los) 컬럼이 없어 24시간 필터링 생략")
            df_filtered = df.copy()
        
        return df_filtered
    
    def remove_medical_impossibilities(self, df):
        """의학적으로 불가능한 이상치 제거"""
        print("🔬 의학적 불가능한 이상치 제거 중...")
        
        original_count = len(df)
        invalid_rows = set()
        removed_details = []
        
        # 의학적 불가능성 필터 정의
        medical_filters = {
            # 기본 정보 (음수 불가능)
            'age': {'min': 0, 'max': None},
            'height_cm': {'min': 0, 'max': None},
            'weight_kg': {'min': 0, 'max': None},
            'bmi': {'min': 0, 'max': None},
            
            # 생체징후 (0 이하 불가능)
            'heart_rate_mean': {'min': 0.1, 'max': None},
            'sbp_mean': {'min': 0.1, 'max': None},
            'dbp_mean': {'min': 0, 'max': None},
            'temperature_mean': {'min': 0, 'max': None},
            'spo2_mean': {'min': 0, 'max': 100},  # 산소포화도는 0-100%
            'respiratory_rate_mean': {'min': 0.1, 'max': None},
            
            # 실험실 수치 (음수 불가능)
            'glucose_mean': {'min': 0, 'max': None},
            'hemoglobin_mean': {'min': 0, 'max': None},
            'creatinine_mean': {'min': 0, 'max': None},
            'lactate_mean': {'min': 0, 'max': None},
            'potassium_mean': {'min': 0, 'max': None},
            'sodium_mean': {'min': 0, 'max': None},
            'wbc_mean': {'min': 0, 'max': None},
            'platelet_mean': {'min': 0, 'max': None},
            
            # GCS 점수 (표준 범위)
            'gcs_total_mean': {'min': 3, 'max': 15},
            'gcs_eye_mean': {'min': 1, 'max': 4},
            'gcs_verbal_mean': {'min': 1, 'max': 5},
            'gcs_motor_mean': {'min': 1, 'max': 6},
        }
        
        for column, criteria in medical_filters.items():
            if column not in df.columns:
                continue
                
            min_val = criteria.get('min')
            max_val = criteria.get('max')
            
            # 최소값 조건 확인
            if min_val is not None:
                invalid_min = df[column] < min_val
                invalid_count = invalid_min.sum()
                if invalid_count > 0:
                    invalid_rows.update(df[invalid_min].index.tolist())
                    removed_details.append(f"{column} < {min_val}: {invalid_count}건")
            
            # 최대값 조건 확인
            if max_val is not None:
                invalid_max = df[column] > max_val
                invalid_count = invalid_max.sum()
                if invalid_count > 0:
                    invalid_rows.update(df[invalid_max].index.tolist())
                    removed_details.append(f"{column} > {max_val}: {invalid_count}건")
        
        # 이상치 제거
        if invalid_rows:
            df_cleaned = df.drop(index=list(invalid_rows)).reset_index(drop=True)
            removed_count = len(invalid_rows)
            
            print(f"✅ 의학적 불가능한 값 제거: {removed_count:,}명")
            print(f"   남은 환자: {len(df_cleaned):,}명")
            
            self.cleaning_log.append(f"이상치 제거: {removed_count:,}명")
            for detail in removed_details[:5]:  # 상위 5개만 로그
                self.cleaning_log.append(f"  - {detail}")
        else:
            print("✅ 의학적 불가능한 값 없음")
            df_cleaned = df.copy()
        
        return df_cleaned
    
    def remove_mortality_7d_column(self, df):
        """7일 사망 여부 컬럼 제거"""
        print("📊 7일 사망 여부 컬럼 제거 중...")
        
        if 'mortality_7d' in df.columns:
            df_processed = df.drop(columns=['mortality_7d'])
            print("✅ mortality_7d 컬럼 제거 완료")
            self.cleaning_log.append("7일 사망 여부 컬럼 제거")
        else:
            print("⚠️ mortality_7d 컬럼이 존재하지 않음")
            df_processed = df.copy()
        
        return df_processed
    
    def handle_missing_data(self, df, missing_threshold=0.5):
        """결측치 처리"""
        print(f"🔍 결측치 처리 시작 (임계값: {missing_threshold:.0%})")
        
        original_columns = df.shape[1]
        original_rows = df.shape[0]
        
        # 1. 결측치 비율 계산
        missing_ratios = df.isnull().sum() / len(df)
        high_missing_cols = missing_ratios[missing_ratios >= missing_threshold].index.tolist()
        
        if high_missing_cols:
            print(f"📉 {missing_threshold:.0%} 이상 결측 컬럼 제거: {len(high_missing_cols)}개")
            for col in high_missing_cols[:5]:  # 상위 5개만 표시
                ratio = missing_ratios[col]
                print(f"   - {col}: {ratio:.1%}")
            
            # 고결측 컬럼 제거
            df_cols_dropped = df.drop(columns=high_missing_cols)
            
            self.cleaning_log.append(f"{missing_threshold:.0%} 이상 결측 컬럼 제거: {len(high_missing_cols)}개")
        else:
            print(f"✅ {missing_threshold:.0%} 이상 결측 컬럼 없음")
            df_cols_dropped = df.copy()
        
        # 2. 결측치가 있는 행 제거
        missing_rows_count = df_cols_dropped.isnull().any(axis=1).sum()
        if missing_rows_count > 0:
            df_cleaned = df_cols_dropped.dropna().reset_index(drop=True)
            print(f"📉 결측치 행 제거: {missing_rows_count:,}개")
            
            self.cleaning_log.append(f"결측치 행 제거: {missing_rows_count:,}개")
        else:
            print("✅ 결측치 행 없음")
            df_cleaned = df_cols_dropped.copy()
        
        final_columns = df_cleaned.shape[1]
        final_rows = df_cleaned.shape[0]
        
        print(f"✅ 결측치 처리 완료:")
        print(f"   컬럼: {original_columns} → {final_columns} ({original_columns - final_columns}개 제거)")
        print(f"   행: {original_rows:,} → {final_rows:,} ({original_rows - final_rows:,}개 제거)")
        
        return df_cleaned
    
    def analyze_target_distribution(self, df):
        """타겟 변수 분포 분석"""
        print("🎯 타겟 변수 분포 분석...")
        
        if 'mortality_48h' in df.columns:
            dist = df['mortality_48h'].value_counts().sort_index()
            total = len(df)
            mortality_rate = df['mortality_48h'].mean()
            
            print(f"✅ 48시간 사망률 분석:")
            print(f"   - 생존 (0): {dist[0]:,}명 ({dist[0]/total:.1%})")
            print(f"   - 사망 (1): {dist[1]:,}명 ({dist[1]/total:.1%})")
            print(f"   - 사망률: {mortality_rate:.1%}")
            print(f"   - 불균형 비율: {dist[0]/dist[1]:.1f}:1")
            
            self.cleaning_log.append(f"최종 사망률: {mortality_rate:.1%}")
            self.cleaning_log.append(f"클래스 불균형: {dist[0]/dist[1]:.1f}:1")
        
        return df
    
    def save_dataset(self, df, filename='mimic_mortality_cleaned.csv'):
        """정제된 데이터셋 저장"""
        output_file = self.output_path / filename
        df.to_csv(output_file, index=False)
        print(f"✅ 정제된 데이터셋 저장: {output_file}")
        return output_file
    
    def save_cleaning_log(self, filename='cleaning_log.txt'):
        """정제 과정 로그 저장"""
        log_file = self.output_path / filename
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("데이터 정제 과정 로그\n")
            f.write("=" * 50 + "\n")
            for log_entry in self.cleaning_log:
                f.write(f"{log_entry}\n")
        
        print(f"✅ 정제 로그 저장: {log_file}")
        return log_file

def main():
    """메인 실행 함수"""
    print("🧹 데이터 정제 시작")
    print("=" * 60)
    
    # 프로젝트 루트 디렉토리 찾기
    project_root = Path(__file__).parent.parent
    
    # 경로 설정 - 상대 경로 사용
    input_path = project_root / "dataset" / "0_raw" / "mimic_mortality_raw.csv"
    output_path = project_root / "dataset" / "1_cleaned"
    
    # 데이터 정제기 초기화
    cleaner = DataCleaner(input_path, output_path)
    
    # 데이터 로드
    df = cleaner.load_data()
    original_shape = df.shape
    
    # 단계별 정제 수행
    df = cleaner.remove_early_deaths(df)
    df = cleaner.remove_medical_impossibilities(df) 
    df = cleaner.remove_mortality_7d_column(df)
    df = cleaner.handle_missing_data(df, missing_threshold=0.5)
    df = cleaner.analyze_target_distribution(df)
    
    # 결과 저장
    cleaner.save_dataset(df)
    cleaner.save_cleaning_log()
    
    final_shape = df.shape
    
    print("\n" + "=" * 60)
    print("✅ 데이터 정제 완료!")
    print(f"   - 원본: {original_shape[0]:,}행 x {original_shape[1]}열")
    print(f"   - 최종: {final_shape[0]:,}행 x {final_shape[1]}열")
    print(f"   - 데이터 보존율: {final_shape[0]/original_shape[0]:.1%}")
    print("💡 다음 단계: 04_data_splitting.py 실행")
    print("=" * 60)

if __name__ == "__main__":
    main()
