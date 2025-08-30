#!/usr/bin/env python3
"""
데이터 분할
- 정제된 데이터를 train/validation/test로 분할 (6:2:2)
- 층화 샘플링으로 각 세트의 클래스 분포 유지
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class DataSplitter:
    def __init__(self, input_path, output_path):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        self.target_column = 'mortality_48h'
        
    def load_data(self):
        """정제된 데이터 로드"""
        print("📂 정제된 데이터 로딩 중...")
        df = pd.read_csv(self.input_path)
        
        print(f"✅ 데이터 로드 완료: {df.shape}")
        
        # 필수 컬럼 확인
        if self.target_column not in df.columns:
            raise ValueError(f"타겟 컬럼({self.target_column})이 없습니다.")
        
        return df
    
    def analyze_distribution(self, df, title="데이터 분포"):
        """클래스 분포 분석"""
        print(f"\n📊 {title}")
        
        if self.target_column in df.columns:
            dist = df[self.target_column].value_counts().sort_index()
            total = len(df)
            mortality_rate = df[self.target_column].mean()
            
            print(f"   - 전체: {total:,}명")
            print(f"   - 생존 (0): {dist[0]:,}명 ({dist[0]/total:.1%})")
            print(f"   - 사망 (1): {dist[1]:,}명 ({dist[1]/total:.1%})")
            print(f"   - 사망률: {mortality_rate:.1%}")
            print(f"   - 불균형 비율: {dist[0]/dist[1]:.1f}:1")
            
            return dist, mortality_rate
        
        return None, None
    
    def stratified_split(self, df, test_size=0.2, val_size=0.2, random_state=42):
        """층화 분할 (6:2:2)"""
        print(f"\n🔄 층화 분할 수행 중...")
        print(f"   - Train: {1-test_size-val_size:.0%}")
        print(f"   - Validation: {val_size:.0%}")
        print(f"   - Test: {test_size:.0%}")
        
        # 특성과 타겟 분리
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        # 1단계: Train + Temp(Val+Test) 분할
        temp_size = test_size + val_size
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, 
            test_size=temp_size, 
            stratify=y, 
            random_state=random_state
        )
        
        # 2단계: Temp를 Validation과 Test로 분할
        val_ratio = val_size / temp_size
        X_test, X_val, y_test, y_val = train_test_split(
            X_temp, y_temp,
            test_size=(1-val_ratio),
            stratify=y_temp,
            random_state=random_state
        )
        
        # 데이터프레임으로 재구성
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        print(f"✅ 분할 완료:")
        print(f"   - Train: {len(train_df):,}명")
        print(f"   - Validation: {len(val_df):,}명")
        print(f"   - Test: {len(test_df):,}명")
        
        return train_df, val_df, test_df
    
    def validate_splits(self, original_df, train_df, val_df, test_df):
        """분할 결과 검증"""
        print(f"\n🔍 분할 결과 검증...")
        
        # 데이터 개수 확인
        total_original = len(original_df)
        total_split = len(train_df) + len(val_df) + len(test_df)
        
        print(f"   - 원본 데이터: {total_original:,}명")
        print(f"   - 분할 데이터 합계: {total_split:,}명")
        
        if total_original != total_split:
            print(f"   ❌ 데이터 손실 발생: {total_original - total_split:,}명")
        else:
            print(f"   ✅ 데이터 손실 없음")
        
        # 각 세트별 클래스 분포 확인
        datasets = [("Train", train_df), ("Validation", val_df), ("Test", test_df)]
        
        print(f"\n   클래스 분포 비교:")
        for name, data in datasets:
            if self.target_column in data.columns:
                mortality_rate = data[self.target_column].mean()
                print(f"   - {name}: {mortality_rate:.1%}")
        
        return True
    
    def save_splits(self, train_df, val_df, test_df):
        """분할된 데이터 저장"""
        print(f"\n💾 분할 데이터 저장 중...")
        
        datasets = {
            'train': train_df,
            'validation': val_df,
            'test': test_df
        }
        
        saved_files = {}
        
        for name, df in datasets.items():
            filename = f"mimic_mortality_{name}.csv"
            filepath = self.output_path / filename
            df.to_csv(filepath, index=False)
            saved_files[name] = filepath
            print(f"   ✅ {name}: {filepath}")
        
        return saved_files
    
    def save_split_summary(self, original_df, train_df, val_df, test_df, saved_files):
        """분할 요약 정보 저장"""
        summary_file = self.output_path / "split_summary.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("데이터 분할 요약\n")
            f.write("=" * 50 + "\n")
            f.write(f"분할 일시: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("분할 방법: 층화 샘플링 (Stratified Sampling)\n")
            f.write("분할 비율: Train 60% : Validation 20% : Test 20%\n")
            f.write("층화 기준: 48시간 사망률 (mortality_48h)\n\n")
            
            f.write("분할 결과:\n")
            f.write(f"- 원본 데이터: {len(original_df):,}명\n")
            f.write(f"- Train 세트: {len(train_df):,}명 ({len(train_df)/len(original_df):.1%})\n")
            f.write(f"- Validation 세트: {len(val_df):,}명 ({len(val_df)/len(original_df):.1%})\n")
            f.write(f"- Test 세트: {len(test_df):,}명 ({len(test_df)/len(original_df):.1%})\n\n")
            
            # 클래스 분포
            f.write("클래스 분포 (48시간 사망률):\n")
            datasets = [("원본", original_df), ("Train", train_df), ("Validation", val_df), ("Test", test_df)]
            
            for name, df in datasets:
                if self.target_column in df.columns:
                    mortality_rate = df[self.target_column].mean()
                    mortality_count = df[self.target_column].sum()
                    f.write(f"- {name}: {mortality_rate:.1%} ({mortality_count:,}명 사망)\n")
            
            f.write(f"\n저장 파일:\n")
            for name, filepath in saved_files.items():
                f.write(f"- {name}: {filepath}\n")
            
            f.write(f"\n다음 단계: 리샘플링 (05_resampling.py)\n")
        
        print(f"   ✅ 분할 요약: {summary_file}")

def main():
    """메인 실행 함수"""
    print("✂️ 데이터 분할 시작")
    print("=" * 60)
    
    # 프로젝트 루트 디렉토리 찾기
    project_root = Path(__file__).parent.parent
    
    # 경로 설정 - 상대 경로 사용
    input_path = project_root / "dataset" / "1_cleaned" / "mimic_mortality_cleaned.csv"
    output_path = project_root / "dataset" / "2_split"
    
    # 데이터 분할기 초기화
    splitter = DataSplitter(input_path, output_path)
    
    # 1. 데이터 로드
    df = splitter.load_data()
    
    # 2. 원본 분포 분석
    splitter.analyze_distribution(df, "원본 데이터 분포")
    
    # 3. 층화 분할
    train_df, val_df, test_df = splitter.stratified_split(df)
    
    # 4. 분할 결과 검증
    splitter.validate_splits(df, train_df, val_df, test_df)
    
    # 5. 각 세트별 분포 확인
    for name, data in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        splitter.analyze_distribution(data, f"{name} 세트 분포")
    
    # 6. 데이터 저장
    saved_files = splitter.save_splits(train_df, val_df, test_df)
    
    # 7. 요약 정보 저장
    splitter.save_split_summary(df, train_df, val_df, test_df, saved_files)
    
    print("\n" + "=" * 60)
    print("✅ 데이터 분할 완료!")
    print(f"   📁 저장 위치: {output_path}")
    print("💡 다음 단계: 05_resampling.py 실행")
    print("=" * 60)

if __name__ == "__main__":
    main()
