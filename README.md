# MIMIC-IV 48시간 사망률 예측

**MIMIC-IV 데이터만 있으면 누구든지 동일한 분석 결과를 얻을 수 있는 완전 재현 가능한 프로젝트**

> ⚠️ **중요**: 이 프로젝트는 PhysioNet의 MIMIC-IV 데이터베이스에 대한 정당한 접근 승인이 필요합니다. 데이터는 민감한 의료 정보를 포함하므로 승인된 연구 목적으로만 사용해야 합니다.

## 🏥 프로젝트 개요

- **목표**: ICU 입실 후 24시간 데이터로 48시간 이내 사망 예측
- **데이터**: MIMIC-IV v3.1 중환자실 데이터
- **특징**: 생체징후, 검사수치, 인구학적 정보, 동반질환
- **모델**: 6개 모델 (Logistic Regression, SVC, Random Forest, XGBoost, LightGBM, Extra Trees)
- **최적화**: Optuna 베이지안 하이퍼파라미터 튜닝
- **리샘플링**: SMOTE, Downsampling으로 클래스 불균형 해결
- **재현성**: 100% (모든 random_state=42 고정)

## ⚠️ 중요 사전 요구사항

### 1. MIMIC-IV 데이터 접근
- **PhysioNet 계정**: https://physionet.org/register/
- **MIMIC-IV 데이터베이스 접근 승인**: 교육과정 이수 및 승인 필요
- **MIMIC-IV v3.1 데이터**: 최소 50GB 저장공간 필요

### 2. 시스템 요구사항
- **Python 3.8+**
- **메모리**: 최소 16GB RAM 권장
- **저장공간**: 100GB 이상 (원본 데이터 + 중간 결과 + 모델 파일)
- **실행시간**: 전체 파이프라인 약 2-6시간 (데이터 크기에 따라)

## 📁 프로젝트 구조

```
📦 48hour_death_predict/
├── 📂 code/                           # 모든 분석 코드
│   ├── 01_data_extraction.py          # 데이터 추출 + 파생변수 생성 (MIMIC 원본 → 완전 처리)
│   ├── 02_data_cleaning.py            # 데이터 정제 (24h 사망자 제거, 이상치)
│   ├── 03_data_splitting.py           # 데이터 분할 (Train/Val/Test = 6:2:2)
│   ├── 04_resampling.py               # 리샘플링 (SMOTE, Downsampling)
│   ├── 05_modeling_evaluation.py      # 모델링 (6개 모델 학습 및 평가)
│   ├── 06_hyperparameter_tuning.py    # 하이퍼파라미터 튜닝 (Optuna)
│   ├── 07_shap_analysis.py            # SHAP 해석성 분석
│   ├── master_pipeline.py             # 마스터 스크립트 (2-6 자동 실행)
│   └── figure_generator.py            # 모든 시각화 생성
├── 📂 dataset/                        # 데이터 저장소 (Git에서 제외됨)
│   ├── 📂 0_raw/                      # 원본 추출 데이터 (파생변수 포함)
│   ├── 📂 1_cleaned/                  # 데이터 정제 후
│   ├── 📂 2_split/                    # Train/Val/Test 분할 후
│   ├── 📂 3_resampled/                # 리샘플링 후
│   │   ├── 📂 original/               # 원본 데이터 복사본 (04_resampling.py에서 생성)
│   │   ├── 📂 smote/                  # SMOTE 적용 데이터
│   │   └── 📂 downsampling/           # Downsampling 적용 데이터
│   ├── 📂 4_modeling/                 # 모델링 결과 및 저장된 모델들
│   └── 📂 5_final_models/             # 최종 튜닝된 모델들
├── 📂 results/                        # 분석 결과 (Git에서 제외됨)
│   └── 📂 07_shap_analysis/           # SHAP 분석 결과
├── 📂 figures/                        # 모든 시각화 결과
├── 📄 .gitignore                      # Git 제외 파일 목록
├── 📄 README.md                       # 이 파일
└── 📄 LICENSE                         # 라이선스 파일
```

## 🚀 사용법

### 1단계: 환경 설정

#### Python 패키지 설치
```bash
# 필수 패키지 설치
pip install pandas numpy scikit-learn xgboost lightgbm
pip install imbalanced-learn optuna joblib
pip install matplotlib seaborn
pip install shap  # SHAP 분석용 (선택사항)
```

#### MIMIC-IV 데이터 준비
1. PhysioNet에서 MIMIC-IV v3.1 데이터 다운로드
2. 데이터를 적절한 폴더에 압축 해제
3. 데이터 경로를 기억해두세요 (다음 단계에서 설정)

### 2단계: 데이터 경로 설정

**중요**: `code/01_data_extraction.py` 파일을 열어 데이터 경로를 설정하세요:

```python
# 01_data_extraction.py의 main() 함수에서 수정
data_root = "C:/path/to/your/mimic-iv/3.1"  # ← 여기를 실제 경로로 변경

# 예시:
# Windows: data_root = "C:/data/mimic-iv/3.1"
# Linux/Mac: data_root = "/home/user/data/mimic-iv/3.1"
```

### 3단계: 실행

#### 옵션 1: 전자동 실행 (권장)
```bash
# 1단계: 데이터 추출 (첫 실행시만)
python code/01_data_extraction.py

# 2단계: 전체 파이프라인 자동 실행
python code/master_pipeline.py

# 3단계: 시각화 생성
python code/figure_generator.py

# 4단계: SHAP 분석 (선택사항)
python code/07_shap_analysis.py
```

#### 옵션 2: 단계별 수동 실행
```bash
# 데이터 처리 단계
python code/01_data_extraction.py    # 데이터 추출 (10-30분)
python code/02_data_cleaning.py      # 데이터 정제 (5분)
python code/03_data_splitting.py     # 데이터 분할 (1분)
python code/04_resampling.py         # 리샘플링 + Original 복사 (5분)

# 모델링 단계
python code/05_modeling_evaluation.py  # 모델링 (30-60분)
python code/06_hyperparameter_tuning.py  # 튜닝 (60-120분)

# 분석 및 시각화
python code/07_shap_analysis.py      # SHAP 분석 (10-20분)
python code/figure_generator.py      # 시각화 생성 (5분)
```

## 🎯 주요 특징

### ✅ **과학적 엄밀성**
- **완전 재현 가능**: 모든 random_state=42 고정
- **데이터 누수 방지**: 올바른 시간적 분할
- **의학적 타당성**: 24시간 미만 사망자 제거

### 🤖 **포괄적 모델링**
- **6개 모델**: Logistic Regression, SVC, Random Forest, XGBoost, LightGBM, Extra Trees
- **3가지 데이터셋**: Original, SMOTE, Downsampling으로 클래스 불균형 해결
- **최적화**: Optuna 베이지안 하이퍼파라미터 튜닝

### 📊 **완전한 분석**
- **자동화된 파이프라인**: 한 번의 명령으로 전체 분석
- **풍부한 시각화**: 9개 종합 대시보드 + SHAP 분석
- **상세한 로깅**: 모든 단계별 실행 기록
- **해석성 분석**: SHAP으로 모델 설명 가능성 제공

### 🔄 **리샘플링 전략**
- **Original**: 리샘플링 없이 원본 데이터 사용 (클래스 불균형 상태)
- **SMOTE**: 합성 데이터 생성으로 소수 클래스 증강
- **Downsampling**: 다수 클래스 축소로 균형 맞춤
- **일관된 검증**: 모든 방법에서 동일한 Validation/Test 세트 사용

## 🎨 생성되는 결과물

### 📊 시각화 (figures/ 폴더)
1. **01_data_distribution.png**: 데이터 분포 분석
2. **02_missing_data_analysis.png**: 결측치 히트맵
3. **03_resampling_comparison.png**: 리샘플링 효과
4. **04_model_performance.png**: 모델 성능 비교
5. **05_final_dashboard.png**: 최종 결과 대시보드

### 🤖 모델 파일
- **6_final_models/**: 최적화된 최종 모델들 (.pkl)
- **5_modeling/**: 모든 모델 성능 결과 (.csv)

### 📄 보고서
- **modeling_summary.txt**: 모델링 요약
- **final_report.txt**: 최종 프로젝트 결과
- **pipeline_summary.txt**: 전체 실행 로그

## ⚙️ 설정 변경

각 스크립트의 상단에서 경로나 파라미터를 쉽게 변경할 수 있습니다:

```python
# 01_data_extraction.py에서
MIMIC_DATA_PATH = "/path/to/your/mimic-iv/data"

# master_pipeline.py에서  
TIMEOUT_MINUTES = 60  # 각 단계 최대 실행 시간
```

## 🔧 문제 해결

### 자주 발생하는 문제들

#### 1. MIMIC-IV 데이터 관련 오류

**문제**: `❌ 오류: MIMIC-IV 데이터 경로가 존재하지 않습니다!`
```bash
해결책:
1. 01_data_extraction.py에서 data_root 경로를 정확히 설정
2. MIMIC-IV 데이터가 올바르게 압축 해제되었는지 확인
3. 경로 구분자 확인 (Windows: /, Linux/Mac: /)
```

**문제**: `FileNotFoundError: [파일명].csv`
```bash
해결책:
1. MIMIC-IV 데이터 다운로드 완료 확인
2. 모든 필요한 파일이 올바른 위치에 있는지 확인
3. 파일 압축 해제 완료 확인
```

#### 2. 메모리 관련 오류

**문제**: `MemoryError` 또는 시스템 멈춤
```python
해결책 (01_data_extraction.py에서):
# 청킹 크기 줄이기
chunk_size = 100000  # 기본값: 500000

# 또는 더 작게
chunk_size = 50000
```

**문제**: 느린 실행 속도
```bash
해결책:
1. SSD 사용 권장
2. 16GB+ RAM 확보
3. 다른 프로그램 종료
4. 청킹 크기 조정
```

#### 3. 패키지 설치 오류

**문제**: `ModuleNotFoundError`
```bash
# 가상환경 생성 및 활성화 (권장)
python -m venv mimic_env
source mimic_env/bin/activate  # Linux/Mac
# 또는
mimic_env\Scripts\activate     # Windows

# 패키지 설치
pip install --upgrade pip
pip install pandas numpy scikit-learn
pip install xgboost lightgbm imbalanced-learn
pip install optuna joblib matplotlib seaborn shap
```

#### 4. 실행 중 오류

**문제**: 중간에 실행이 멈춤
```bash
해결책:
1. 마스터 파이프라인 대신 개별 스크립트 실행
2. 로그 확인하여 어느 단계에서 멈췄는지 파악
3. 해당 단계부터 재실행
```

**문제**: `Permission denied` 오류
```bash
해결책:
1. 관리자 권한으로 실행
2. 파일/폴더 권한 확인
3. 바이러스 백신 실시간 검사 일시 중지
```

#### 5. SHAP 분석 오류

**문제**: `SHAP 패키지가 설치되지 않았습니다`
```bash
pip install shap

# M1 Mac의 경우:
conda install -c conda-forge shap
```

## 📈 예상 성능

- **ROC-AUC**: 0.75-0.85 범위 예상
- **F1-Score**: 0.3-0.5 범위 예상 (클래스 불균형으로 인해)
- **실행 시간**: 전체 파이프라인 약 2-6시간 (데이터 크기에 따라)

## 📊 생성되는 결과물

### 데이터 파일
- `dataset/0_raw/mimic_mortality_raw.csv` - 추출된 원본 데이터
- `dataset/5_final_models/*.pkl` - 최적화된 모델 파일들
- `results/07_shap_analysis/*.csv` - SHAP 분석 결과

### 시각화 파일 (figures/ 폴더)
1. `01_data_distribution.png` - 데이터 분포 분석
2. `02_missing_data_analysis.png` - 결측치 히트맵
3. `02_missing_data_impact.png` - 결측치 제거 영향 분석
4. `03_resampling_comparison.png` - 리샘플링 효과 비교
5. `03_data_pipeline.png` - 데이터 처리 파이프라인
6. `04_model_performance.png` - 모델 성능 비교
7. `05_final_dashboard.png` - 최종 결과 대시보드
8. `05_shap_analysis.png` - SHAP 분석 종합
9. `06_shap_beeplot_*.png` - SHAP Beeplot (상위 모델별)

### 보고서 파일
- `pipeline_summary.txt` - 전체 실행 로그
- `dataset/*/summary.txt` - 각 단계별 요약
- `results/07_shap_analysis/shap_analysis_summary.txt` - SHAP 분석 요약

## 🔬 연구 활용 가이드

### 논문 작성시 참고사항
1. **데이터**: MIMIC-IV v3.1 중환자실 데이터 사용
2. **환자 선정**: 첫 번째 ICU 입원만 포함 (재입원 제외)
3. **시간 범위**: ICU 입실 후 24시간 내 데이터로 48시간 사망 예측
4. **특성**: 생체징후, 검사수치, 인구학적 정보, 동반질환 포함
5. **모델**: 6개 알고리즘 + 베이지안 최적화
6. **검증**: 층화 분할 (Train:Val:Test = 6:2:2)

### 확장 연구 아이디어
- **시간 범위 변경**: 12시간 → 24시간 예측, 72시간 → 48시간 예측
- **추가 특성**: 약물 정보, 처치 정보, 간호 기록
- **딥러닝**: LSTM, Transformer 등 시계열 모델 적용
- **앙상블**: 여러 모델의 스태킹/블렌딩
- **해석성**: LIME, Grad-CAM 등 추가 해석 기법

## ⚖️ 윤리적 고려사항

- **데이터 보안**: MIMIC-IV 데이터는 민감한 의료 정보를 포함합니다
- **사용 승인**: 반드시 PhysioNet의 사용 승인을 받은 후 사용하세요
- **연구 목적**: 상업적 이용 금지, 연구 및 교육 목적으로만 사용
- **개인정보**: 모든 환자 식별 정보는 익명화되어 있습니다

## 🤝 기여 및 문의

이 프로젝트는 MIMIC-IV 데이터를 활용한 의료 AI 연구를 위한 표준화된 베이스라인입니다.

### 기여 방법
- Issue 제기: 버그 신고, 기능 요청
- Pull Request: 코드 개선, 문서 업데이트
- 연구 결과 공유: 이 코드를 사용한 연구 결과

### 라이선스
- **코드**: MIT License (자유로운 사용 및 수정 가능)
- **데이터**: MIMIC-IV License (별도 승인 필요)

---

**📧 문의**: 이 코드를 사용한 연구 결과나 개선 사항이 있으시면 언제든 공유해 주세요!
