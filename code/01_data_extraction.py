"""
MIMIC IV 중환자실 사망률 예측 데이터 추출 스크립트

작성자: AI Assistant  
작성일: 2024-01-20
목적: MIMIC IV 데이터에서 ICU 환자의 24시간 기반 48시간 사망률 예측에 필요한 데이터 추출

주요 추출 데이터:
- 기본 의료 정보: 키, 몸무게, BMI, 혈압, 혈당, SpO2
- 대사질환 정보: 당뇨병, 고혈압, 비만, 이상지질혈증 등
- 라이프스타일 정보: 흡연, 음주, 임신 여부
- 수면 관련 정보
- 48시간 사망률 라벨
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class MIMICMortalityDataExtractor:
    def __init__(self, data_root: str, output_dir: str = None):
        """
        MIMIC IV 사망률 예측 데이터 추출기 초기화
        
        Args:
            data_root: MIMIC IV 데이터 루트 경로
            output_dir: 결과 저장 경로 (기본값: 현재 디렉토리/dataset/0_raw)
        """
        self.data_root = data_root
        
        # 출력 디렉토리 설정 (기본값 제공)
        if output_dir is None:
            self.output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "0_raw")
        else:
            self.output_dir = output_dir
        
        # 출력 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 데이터 경로 설정
        self.hosp_path = os.path.join(data_root, "hosp")
        self.icu_path = os.path.join(data_root, "icu")
        
        # 주요 테이블들 로드
        self.load_base_tables()
        
        # 생체징후 관련 아이템 ID 설정
        self.setup_vital_signs_items()
        
        # 검사 관련 아이템 ID 설정
        self.setup_lab_items()
        
        # 대사질환 ICD 코드 설정
        self.setup_disease_codes()
        
        # OMR 신체측정 매핑 설정
        self.setup_anthropometric_measures()
        
        # 수면/진정 관련 아이템 설정
        self.setup_sleep_sedation_items()
        
    def load_base_tables(self):
        """기본 테이블들 로드"""
        print("기본 테이블 로딩 중...")
        
        # 환자 기본 정보
        self.patients = pd.read_csv(os.path.join(self.hosp_path, "patients.csv"))
        print(f"환자 수: {len(self.patients):,}명")
        
        # 입원 정보
        self.admissions = pd.read_csv(os.path.join(self.hosp_path, "admissions.csv"))
        print(f"입원 기록: {len(self.admissions):,}건")
        
        # ICU 재원 정보
        self.icustays = pd.read_csv(os.path.join(self.icu_path, "icustays.csv"))
        print(f"ICU 재원 기록: {len(self.icustays):,}건")
        
        # 진단 정보
        self.diagnoses_icd = pd.read_csv(os.path.join(self.hosp_path, "diagnoses_icd.csv"))
        print(f"진단 기록: {len(self.diagnoses_icd):,}건")
        
        # ICD 진단 사전
        self.d_icd_diagnoses = pd.read_csv(os.path.join(self.hosp_path, "d_icd_diagnoses.csv"))
        
        # 차트 아이템 사전
        self.d_items = pd.read_csv(os.path.join(self.icu_path, "d_items.csv"))
        
        # 검사 아이템 사전
        self.d_labitems = pd.read_csv(os.path.join(self.hosp_path, "d_labitems.csv"))
        
        # OMR 테이블 로드
        omr_path = os.path.join(self.hosp_path, "omr.csv")
        if os.path.exists(omr_path):
            self.omr = pd.read_csv(omr_path)
            print(f"OMR 기록: {len(self.omr):,}건")
        else:
            print("⚠️ OMR 테이블을 찾을 수 없습니다.")
            self.omr = pd.DataFrame()
    
    def setup_vital_signs_items(self):
        """생체징후 관련 아이템 ID 설정"""
        # 혈압 관련
        self.bp_items = {
            'sbp': [220050, 220179, 225309],  # 수축기 혈압
            'dbp': [220051, 220180, 225310],  # 이완기 혈압
            'mbp': [220052, 220181, 225312]   # 평균 혈압
        }
        
        # 기타 생체징후
        self.vital_items = {
            'heart_rate': [220045, 211],      # 심박수
            'respiratory_rate': [220210, 618, 615, 220339, 224690],  # 호흡수
            'temperature': [223762, 676, 223761, 678],  # 체온
            'spo2': [220277, 646]             # 산소포화도
        }
        
    def setup_lab_items(self):
        """검사 관련 아이템 ID 설정"""
        self.lab_items = {
            'glucose': [50809, 50931, 225664],    # 혈당
            'creatinine': [50912],                # 크레아티닌  
            'bun': [51006],                       # BUN
            'sodium': [50983],                    # 나트륨
            'potassium': [50971],                 # 칼륨
            'chloride': [50902],                  # 염소
            'hematocrit': [51221],                # 헤마토크리트
            'hemoglobin': [51222],                # 헤모글로빈
            'platelet': [51265],                  # 혈소판
            'wbc': [51301],                       # 백혈구
            'lactate': [50813],                   # 젖산
        }
    
    def setup_disease_codes(self):
        """대사질환 ICD 코드 설정"""
        # 당뇨병
        self.diabetes_codes = [
            'E10', 'E11', 'E12', 'E13', 'E14',  # ICD-10
            '250'  # ICD-9
        ]
        
        # 고혈압
        self.hypertension_codes = [
            'I10', 'I11', 'I12', 'I13', 'I15',  # ICD-10
            '401', '402', '403', '404', '405'    # ICD-9
        ]
        
        # 비만
        self.obesity_codes = [
            'E66',    # ICD-10
            '278.0'   # ICD-9
        ]
        
        # 이상지질혈증
        self.dyslipidemia_codes = [
            'E78.0', 'E78.1', 'E78.2', 'E78.3', 'E78.4', 'E78.5',  # ICD-10
            '272.0', '272.1', '272.2', '272.3', '272.4'             # ICD-9
        ]
        
        # 갑상선 질환
        self.thyroid_disease_codes = [
            'E00', 'E01', 'E02', 'E03', 'E04', 'E05', 'E06', 'E07',  # ICD-10
            '240', '241', '242', '243', '244', '245', '246'          # ICD-9
        ]
        
        # 신장 질환
        self.kidney_disease_codes = [
            'N18', 'N19', 'I12', 'I13',          # ICD-10
            '585', '586', '403', '404'           # ICD-9
        ]
        
        # 심혈관 질환
        self.cardiovascular_disease_codes = [
            'I20', 'I21', 'I22', 'I23', 'I24', 'I25',  # ICD-10
            '410', '411', '412', '413', '414'          # ICD-9
        ]
        
        # 뇌혈관 질환
        self.cerebrovascular_disease_codes = [
            'I60', 'I61', 'I62', 'I63', 'I64', 'I65', 'I66', 'I67', 'I68', 'I69',  # ICD-10
            '430', '431', '432', '433', '434', '435', '436', '437', '438'           # ICD-9
        ]
        
        # 흡연 관련 코드
        self.smoking_codes = [
            'Z87.891', 'F17', 'Z72.0',          # ICD-10
            'V15.82', '305.1', 'V15.82'        # ICD-9
        ]
        
        # 알코올 관련 코드
        self.alcohol_codes = [
            'F10', 'Z72.1', 'K70',             # ICD-10
            '305.0', '303', '571.0', '571.1', '571.2', '571.3'  # ICD-9
        ]
        
        # 임신 관련 코드
        self.pregnancy_codes = [
            'O00', 'O01', 'O02', 'O03', 'O04', 'O05', 'O06', 'O07', 'O08', 'O09',  # ICD-10 O00-O99
            'O10', 'O11', 'O12', 'O13', 'O14', 'O15', 'O16', 'O20', 'O21', 'O22',
            'O23', 'O24', 'O25', 'O26', 'O28', 'O29', 'O30', 'O31', 'O32', 'O33',
            'O34', 'O35', 'O36', 'O40', 'O41', 'O42', 'O43', 'O44', 'O45', 'O46',
            'O47', 'O48', 'O60', 'O61', 'O62', 'O63', 'O64', 'O65', 'O66', 'O67',
            'O68', 'O69', 'O70', 'O71', 'O72', 'O73', 'O74', 'O75', 'O80', 'O81',
            'O82', 'O83', 'O84', 'O85', 'O86', 'O87', 'O88', 'O89', 'O90', 'O91',
            'O92', 'O94', 'O95', 'O96', 'O97', 'O98', 'O99', 'Z34', 'Z35', 'Z36',
            'Z37', 'Z38', 'Z39',
            '630', '631', '632', '633', '634', '635', '636', '637', '638', '639',    # ICD-9
            '640', '641', '642', '643', '644', '645', '646', '647', '648', '649',
            '650', '651', '652', 'V22', 'V23', 'V24', 'V27', 'V28'
        ]
        
        # 수면 장애 코드
        self.sleep_disorder_codes = [
            'G47', 'F51',                       # ICD-10
            '327', '307.4'                      # ICD-9
        ]
    
    def setup_anthropometric_measures(self):
        """OMR 신체측정 매핑 설정"""
        self.anthropometric_measures = {
            'height': ['Height (Inches)', 'Height'],
            'weight': ['Weight (Lbs)', 'Weight'],
            'bmi': ['BMI (kg/m2)', 'BMI']
        }
    
    def setup_sleep_sedation_items(self):
        """수면/진정 관련 아이템 설정"""
        self.sleep_sedation_items = {
            'rass': [228096, 228333],           # Richmond Agitation-Sedation Scale
            'gcs': [220739, 223900, 223901],    # Glasgow Coma Scale
            'sedation_level': [228329, 228330], # Sedation Level
            'consciousness': [226242],           # Level of Consciousness
        }

    def get_first_icu_stays(self):
        """각 환자의 첫 번째 ICU 재원만 선택"""
        print("첫 번째 ICU 재원 정보 추출 중...")
        
        # ICU 입실 시간 기준으로 정렬
        icustays_sorted = self.icustays.copy()
        icustays_sorted['intime'] = pd.to_datetime(icustays_sorted['intime'])
        icustays_sorted = icustays_sorted.sort_values(['subject_id', 'intime'])
        
        # 각 환자의 첫 번째 ICU 재원만 선택
        first_icu_stays = icustays_sorted.groupby('subject_id').first().reset_index()
        
        print(f"첫 번째 ICU 재원: {len(first_icu_stays):,}건 (전체 환자: {len(first_icu_stays):,}명)")
        
        return first_icu_stays
    
    def calculate_mortality_labels(self, icu_stays):
        """사망률 라벨 계산 (48시간)"""
        print("사망률 라벨 계산 중...")
        
        # 환자 정보와 입원 정보 병합
        mortality_data = icu_stays.copy()
        mortality_data = mortality_data.merge(self.patients[['subject_id', 'gender', 'anchor_age', 'dod']], on='subject_id', how='left')
        mortality_data = mortality_data.merge(
            self.admissions[['hadm_id', 'deathtime', 'hospital_expire_flag']], 
            on='hadm_id', how='left'
        )
        
        # 시간 변환
        mortality_data['intime'] = pd.to_datetime(mortality_data['intime'])
        mortality_data['outtime'] = pd.to_datetime(mortality_data['outtime'])
        mortality_data['dod'] = pd.to_datetime(mortality_data['dod'])
        mortality_data['deathtime'] = pd.to_datetime(mortality_data['deathtime'])
        
        # 48시간 후 시점 계산
        mortality_data['time_48h'] = mortality_data['intime'] + timedelta(hours=48)
        
        # 48시간 이내 사망 여부
        mortality_data['mortality_48h'] = 0
        death_within_48h = (
            (mortality_data['dod'].notna() & (mortality_data['dod'] <= mortality_data['time_48h'])) |
            (mortality_data['deathtime'].notna() & (mortality_data['deathtime'] <= mortality_data['time_48h']))
        )
        mortality_data.loc[death_within_48h, 'mortality_48h'] = 1
        
        # 결과 출력
        mortality_48h_count = mortality_data['mortality_48h'].sum()
        
        print(f"48시간 이내 사망: {mortality_48h_count:,}명 ({mortality_48h_count/len(mortality_data)*100:.2f}%)")
        print("✅ 7일 사망률은 제외됨 (48시간 예측에만 집중)")
        
        return mortality_data
    
    def extract_comorbidities(self, subject_ids):
        """동반질환 정보 추출"""
        print("동반질환 정보 추출 중...")
        
        # 관심 있는 환자들의 진단 정보만 필터링
        diagnoses_subset = self.diagnoses_icd[
            self.diagnoses_icd['subject_id'].isin(subject_ids)
        ].copy()
        
        print(f"대상 환자 진단 기록: {len(diagnoses_subset):,}건")
        
        # 동반질환 여부 확인을 위한 함수
        def has_disease(icd_code, disease_codes):
            if pd.isna(icd_code):
                return False
            icd_str = str(icd_code)
            return any(icd_str.startswith(code) for code in disease_codes)
        
        # 각 환자별 동반질환 여부 계산
        comorbidities_data = []
        
        for subject_id in subject_ids:
            patient_diagnoses = diagnoses_subset[diagnoses_subset['subject_id'] == subject_id]
            
            # 각 질환별 여부 확인
            diabetes = patient_diagnoses['icd_code'].apply(lambda x: has_disease(x, self.diabetes_codes)).any()
            hypertension = patient_diagnoses['icd_code'].apply(lambda x: has_disease(x, self.hypertension_codes)).any()
            obesity = patient_diagnoses['icd_code'].apply(lambda x: has_disease(x, self.obesity_codes)).any()
            dyslipidemia = patient_diagnoses['icd_code'].apply(lambda x: has_disease(x, self.dyslipidemia_codes)).any()
            thyroid_disease = patient_diagnoses['icd_code'].apply(lambda x: has_disease(x, self.thyroid_disease_codes)).any()
            kidney_disease = patient_diagnoses['icd_code'].apply(lambda x: has_disease(x, self.kidney_disease_codes)).any()
            cardiovascular_disease = patient_diagnoses['icd_code'].apply(lambda x: has_disease(x, self.cardiovascular_disease_codes)).any()
            cerebrovascular_disease = patient_diagnoses['icd_code'].apply(lambda x: has_disease(x, self.cerebrovascular_disease_codes)).any()
            smoking_icd = patient_diagnoses['icd_code'].apply(lambda x: has_disease(x, self.smoking_codes)).any()
            alcohol_icd = patient_diagnoses['icd_code'].apply(lambda x: has_disease(x, self.alcohol_codes)).any()
            pregnancy = patient_diagnoses['icd_code'].apply(lambda x: has_disease(x, self.pregnancy_codes)).any()
            sleep_disorder = patient_diagnoses['icd_code'].apply(lambda x: has_disease(x, self.sleep_disorder_codes)).any()
            
            comorbidities_data.append({
                'subject_id': subject_id,
                'diabetes': int(diabetes),
                'hypertension': int(hypertension),
                'obesity': int(obesity),
                'dyslipidemia': int(dyslipidemia),
                'thyroid_disease': int(thyroid_disease),
                'kidney_disease': int(kidney_disease),
                'cardiovascular_disease': int(cardiovascular_disease),
                'cerebrovascular_disease': int(cerebrovascular_disease),
                'smoking_icd': int(smoking_icd),
                'alcohol_icd': int(alcohol_icd),
                'pregnancy': int(pregnancy),
                'sleep_disorder': int(sleep_disorder)
            })
        
        comorbidities_df = pd.DataFrame(comorbidities_data)
        
        # 결과 출력
        disease_counts = {
            '당뇨병': comorbidities_df['diabetes'].sum(),
            '고혈압': comorbidities_df['hypertension'].sum(), 
            '비만': comorbidities_df['obesity'].sum(),
            '이상지질혈증': comorbidities_df['dyslipidemia'].sum(),
            '갑상선질환': comorbidities_df['thyroid_disease'].sum(),
            '신장질환': comorbidities_df['kidney_disease'].sum(),
            '심혈관질환': comorbidities_df['cardiovascular_disease'].sum(),
            '뇌혈관질환': comorbidities_df['cerebrovascular_disease'].sum(),
            '흡연(ICD)': comorbidities_df['smoking_icd'].sum(),
            '알코올(ICD)': comorbidities_df['alcohol_icd'].sum(),
            '임신': comorbidities_df['pregnancy'].sum(),
            '수면장애': comorbidities_df['sleep_disorder'].sum()
        }
        
        print("동반질환 보유 환자 수:")
        for disease, count in disease_counts.items():
            print(f"  {disease}: {count:,}명 ({count/len(comorbidities_df)*100:.1f}%)")
        
        return comorbidities_df
    
    def extract_omr_lifestyle(self, subject_ids):
        """OMR에서 라이프스타일 정보 추출"""
        if self.omr.empty:
            print("OMR 테이블이 없어 라이프스타일 정보 추출 건너뜀")
            return pd.DataFrame()
        
        print("OMR에서 라이프스타일 정보 추출 중...")
        
        # 흡연, 알코올 관련 키워드
        smoking_keywords = ['Smoking', 'Tobacco', 'Cigarette', 'smoke', 'smoking']
        alcohol_keywords = ['Alcohol', 'Drinking', 'alcohol', 'drinking', 'drink']
        
        # OMR에서 라이프스타일 정보 추출
        lifestyle_data = []
        
        for subject_id in subject_ids:
            patient_omr = self.omr[self.omr['subject_id'] == subject_id]
            
            if len(patient_omr) == 0:
                lifestyle_data.append({
                    'subject_id': subject_id,
                    'smoking_omr': 0,
                    'alcohol_omr': 0
                })
                continue
            
            # 흡연 정보 확인
            smoking_omr = 0
            for keyword in smoking_keywords:
                if patient_omr['result_name'].str.contains(keyword, case=False, na=False).any():
                    result_values = patient_omr[
                        patient_omr['result_name'].str.contains(keyword, case=False, na=False)
                    ]['result_value'].str.lower()
                    
                    if result_values.str.contains('yes|current|former|quit|pack', na=False).any():
                        smoking_omr = 1
                        break
            
            # 알코올 정보 확인
            alcohol_omr = 0
            for keyword in alcohol_keywords:
                if patient_omr['result_name'].str.contains(keyword, case=False, na=False).any():
                    result_values = patient_omr[
                        patient_omr['result_name'].str.contains(keyword, case=False, na=False)
                    ]['result_value'].str.lower()
                    
                    if result_values.str.contains('yes|daily|weekly|occasional|social', na=False).any():
                        alcohol_omr = 1
                        break
            
            lifestyle_data.append({
                'subject_id': subject_id,
                'smoking_omr': smoking_omr,
                'alcohol_omr': alcohol_omr
            })
        
        lifestyle_df = pd.DataFrame(lifestyle_data)
        
        # 결과 출력
        print(f"OMR 흡연 정보: {lifestyle_df['smoking_omr'].sum():,}명")
        print(f"OMR 알코올 정보: {lifestyle_df['alcohol_omr'].sum():,}명")
        
        return lifestyle_df
    
    def extract_omr_anthropometric(self, subject_ids):
        """OMR 테이블에서 신체측정 정보 추출 (키, 몸무게, BMI)"""
        if self.omr.empty:
            print("OMR 테이블이 없어 신체측정 정보 추출 건너뜀")
            return pd.DataFrame()
        
        print("OMR 테이블에서 신체측정 정보 추출 중...")
        
        # 모든 신체측정 항목 수집
        anthropometric_items = []
        for measure_list in self.anthropometric_measures.values():
            anthropometric_items.extend(measure_list)
        
        # OMR에서 신체측정 데이터 필터링
        omr_anthro = self.omr[
            (self.omr['subject_id'].isin(subject_ids)) & 
            (self.omr['result_name'].isin(anthropometric_items))
        ].copy()
        
        if len(omr_anthro) == 0:
            print("OMR에서 신체측정 데이터를 찾을 수 없음")
            return pd.DataFrame()
        
        print(f"OMR에서 {len(omr_anthro):,}건의 신체측정 데이터 발견")
        
        # 날짜 변환 후 환자별 최초 측정값 사용
        omr_anthro['chartdate'] = pd.to_datetime(omr_anthro['chartdate'])
        omr_processed = omr_anthro.sort_values(['subject_id', 'result_name', 'chartdate']).groupby(['subject_id', 'result_name']).first().reset_index()
        
        # 피벗 테이블 생성
        omr_pivot = omr_processed.pivot(index='subject_id', columns='result_name', values='result_value').reset_index()
        omr_pivot.columns.name = None
        
        # 수치 변환
        for col in omr_pivot.columns:
            if col != 'subject_id':
                omr_pivot[col] = pd.to_numeric(omr_pivot[col], errors='coerce')
        
        # 키 데이터 처리 (인치 → cm)
        if 'Height (Inches)' in omr_pivot.columns:
            omr_pivot['height_cm'] = omr_pivot['Height (Inches)'] * 2.54
        elif 'Height' in omr_pivot.columns:
            omr_pivot['height_cm'] = omr_pivot['Height'] * 2.54
        
        # 몸무게 데이터 처리 (파운드 → kg)
        if 'Weight (Lbs)' in omr_pivot.columns:
            omr_pivot['weight_kg'] = omr_pivot['Weight (Lbs)'] * 0.453592
        elif 'Weight' in omr_pivot.columns:
            omr_pivot['weight_kg'] = omr_pivot['Weight'] * 0.453592
        
        # BMI 데이터 처리 (직접 추출 우선, 없으면 계산)
        if 'BMI (kg/m2)' in omr_pivot.columns:
            omr_pivot['bmi'] = omr_pivot['BMI (kg/m2)']
        elif 'BMI' in omr_pivot.columns:
            omr_pivot['bmi'] = omr_pivot['BMI']
        elif 'height_cm' in omr_pivot.columns and 'weight_kg' in omr_pivot.columns:
            # BMI 계산 (키와 몸무게가 모두 있는 경우만)
            valid_mask = (
                (omr_pivot['height_cm'].notna()) & 
                (omr_pivot['weight_kg'].notna())
            )
            omr_pivot['bmi'] = np.nan
            omr_pivot.loc[valid_mask, 'bmi'] = (
                omr_pivot.loc[valid_mask, 'weight_kg'] / 
                (omr_pivot.loc[valid_mask, 'height_cm'] / 100) ** 2
            )
        
        # 필요한 칼럼만 선택
        anthro_columns = ['subject_id', 'height_cm', 'weight_kg', 'bmi']
        available_columns = [col for col in anthro_columns if col in omr_pivot.columns]
        result_df = omr_pivot[available_columns]
        
        # 결과 통계
        height_count = result_df['height_cm'].notna().sum() if 'height_cm' in result_df.columns else 0
        weight_count = result_df['weight_kg'].notna().sum() if 'weight_kg' in result_df.columns else 0
        bmi_count = result_df['bmi'].notna().sum() if 'bmi' in result_df.columns else 0
        
        print(f"OMR 신체측정 데이터 추출 완료:")
        print(f"  - 키 데이터: {height_count:,}명")
        print(f"  - 몸무게 데이터: {weight_count:,}명") 
        print(f"  - BMI 데이터: {bmi_count:,}명")
        
        return result_df
    
    def extract_vital_signs(self, icu_stays):
        """생체징후 정보 추출 (청킹 사용)"""
        print("생체징후 정보 추출 중 (청킹 사용)...")
        
        # 관심 대상 ICU stay들
        target_stay_ids = set(icu_stays['stay_id'].unique())
        
        # 모든 관심 itemid들 수집
        all_itemids = []
        
        # 혈압 관련
        for bp_type, itemids in self.bp_items.items():
            all_itemids.extend(itemids)
        
        # 기타 생체징후
        for vital_type, itemids in self.vital_items.items():
            all_itemids.extend(itemids)
        
        # 수면/진정 관련
        for sleep_type, itemids in self.sleep_sedation_items.items():
            all_itemids.extend(itemids)
        
        all_itemids = list(set(all_itemids))
        print(f"추출 대상 itemid: {len(all_itemids)}개")
        
        # chartevents.csv 청킹으로 읽기
        chartevents_path = os.path.join(self.icu_path, "chartevents.csv")
        chunk_size = 500000
        
        vital_data_list = []
        chunk_count = 0
        
        try:
            for chunk in pd.read_csv(chartevents_path, chunksize=chunk_size):
                chunk_count += 1
                if chunk_count % 10 == 0:
                    print(f"  청크 {chunk_count} 처리 중...")
                
                # 관심 대상 필터링
                chunk_filtered = chunk[
                    (chunk['stay_id'].isin(target_stay_ids)) &
                    (chunk['itemid'].isin(all_itemids))
                ].copy()
                
                if len(chunk_filtered) > 0:
                    vital_data_list.append(chunk_filtered)
        
        except Exception as e:
            print(f"chartevents 읽기 중 오류: {e}")
            return pd.DataFrame()
        
        if not vital_data_list:
            print("생체징후 데이터를 찾을 수 없습니다.")
            return pd.DataFrame()
        
        # 모든 청크 합치기
        vital_signs = pd.concat(vital_data_list, ignore_index=True)
        print(f"생체징후 데이터: {len(vital_signs):,}건")
        
        return vital_signs
    
    def extract_lab_results(self, icu_stays):
        """검사 결과 추출 (청킹 사용)"""
        print("검사 결과 추출 중 (청킹 사용)...")
        
        # 관심 대상 ICU stay들의 hadm_id
        target_hadm_ids = set(icu_stays['hadm_id'].unique())
        
        # 관심 대상 itemid들
        all_lab_itemids = []
        for lab_type, itemids in self.lab_items.items():
            all_lab_itemids.extend(itemids)
        
        all_lab_itemids = list(set(all_lab_itemids))
        print(f"추출 대상 lab itemid: {len(all_lab_itemids)}개")
        
        # labevents.csv 청킹으로 읽기
        labevents_path = os.path.join(self.hosp_path, "labevents.csv")
        chunk_size = 500000
        
        lab_data_list = []
        chunk_count = 0
        
        try:
            for chunk in pd.read_csv(labevents_path, chunksize=chunk_size):
                chunk_count += 1
                if chunk_count % 10 == 0:
                    print(f"  청크 {chunk_count} 처리 중...")
                
                # 관심 대상 필터링
                chunk_filtered = chunk[
                    (chunk['hadm_id'].isin(target_hadm_ids)) &
                    (chunk['itemid'].isin(all_lab_itemids))
                ].copy()
                
                if len(chunk_filtered) > 0:
                    lab_data_list.append(chunk_filtered)
        
        except Exception as e:
            print(f"labevents 읽기 중 오류: {e}")
            return pd.DataFrame()
        
        if not lab_data_list:
            print("검사 결과 데이터를 찾을 수 없습니다.")
            return pd.DataFrame()
        
        # 모든 청크 합치기
        lab_results = pd.concat(lab_data_list, ignore_index=True)
        print(f"검사 결과 데이터: {len(lab_results):,}건")
        
        return lab_results
    
    def aggregate_vital_signs(self, vital_signs, icu_stays):
        """생체징후 데이터를 환자별로 집계"""
        print("생체징후 데이터 집계 중...")
        
        if vital_signs.empty:
            return pd.DataFrame()
        
        # 시간 정보 추가 및 24시간 내 데이터 필터링
        vital_signs['charttime'] = pd.to_datetime(vital_signs['charttime'])
        
        # ICU 입실 시간 정보 병합
        icu_times = icu_stays[['stay_id', 'intime']].copy()
        icu_times['intime'] = pd.to_datetime(icu_times['intime'])
        
        vital_signs = vital_signs.merge(icu_times, on='stay_id', how='left')
        vital_signs['hours_from_admission'] = (vital_signs['charttime'] - vital_signs['intime']).dt.total_seconds() / 3600
        
        # 24시간 내 데이터만 사용
        vital_signs_24h = vital_signs[
            (vital_signs['hours_from_admission'] >= 0) & 
            (vital_signs['hours_from_admission'] <= 24)
        ].copy()
        
        print(f"24시간 내 생체징후 데이터: {len(vital_signs_24h):,}건")
        
        # 수치형으로 변환
        vital_signs_24h['valuenum'] = pd.to_numeric(vital_signs_24h['valuenum'], errors='coerce')
        
        # 각 생체징후별 집계를 위한 itemid 매핑
        itemid_mapping = {}
        
        # 혈압 매핑
        for bp_type, itemids in self.bp_items.items():
            for itemid in itemids:
                itemid_mapping[itemid] = bp_type
        
        # 기타 생체징후 매핑
        for vital_type, itemids in self.vital_items.items():
            for itemid in itemids:
                itemid_mapping[itemid] = vital_type
        
        # 수면/진정 매핑
        for sleep_type, itemids in self.sleep_sedation_items.items():
            for itemid in itemids:
                itemid_mapping[itemid] = sleep_type
        
        # itemid로 vital_type 매핑
        vital_signs_24h['vital_type'] = vital_signs_24h['itemid'].map(itemid_mapping)
        
        # 환자별, 생체징후별 집계
        aggregated_vitals = []
        
        for stay_id in vital_signs_24h['stay_id'].unique():
            patient_vitals = vital_signs_24h[vital_signs_24h['stay_id'] == stay_id]
            
            vital_stats = {'stay_id': stay_id}
            
            for vital_type in patient_vitals['vital_type'].unique():
                if pd.isna(vital_type):
                    continue
                
                vital_values = patient_vitals[patient_vitals['vital_type'] == vital_type]['valuenum']
                valid_values = vital_values.dropna()
                
                if len(valid_values) > 0:
                    vital_stats[f'{vital_type}_mean'] = valid_values.mean()
                    vital_stats[f'{vital_type}_max'] = valid_values.max()
                    vital_stats[f'{vital_type}_min'] = valid_values.min()
                    vital_stats[f'{vital_type}_std'] = valid_values.std()
                    vital_stats[f'{vital_type}_count'] = len(valid_values)
            
            aggregated_vitals.append(vital_stats)
        
        aggregated_df = pd.DataFrame(aggregated_vitals)
        print(f"집계된 생체징후 데이터: {len(aggregated_df):,}명")
        
        return aggregated_df
    
    def aggregate_lab_results(self, lab_results, icu_stays):
        """검사 결과를 환자별로 집계"""
        print("검사 결과 집계 중...")
        
        if lab_results.empty:
            return pd.DataFrame()
        
        # 시간 정보 추가 및 24시간 내 데이터 필터링
        lab_results['charttime'] = pd.to_datetime(lab_results['charttime'])
        
        # ICU 입실 시간과 hadm_id로 연결
        icu_times = icu_stays[['stay_id', 'hadm_id', 'intime']].copy()
        icu_times['intime'] = pd.to_datetime(icu_times['intime'])
        
        lab_results = lab_results.merge(icu_times, on='hadm_id', how='left')
        lab_results['hours_from_admission'] = (lab_results['charttime'] - lab_results['intime']).dt.total_seconds() / 3600
        
        # 24시간 내 데이터만 사용
        lab_results_24h = lab_results[
            (lab_results['hours_from_admission'] >= 0) & 
            (lab_results['hours_from_admission'] <= 24)
        ].copy()
        
        print(f"24시간 내 검사 결과: {len(lab_results_24h):,}건")
        
        # itemid로 lab_type 매핑
        itemid_mapping = {}
        for lab_type, itemids in self.lab_items.items():
            for itemid in itemids:
                itemid_mapping[itemid] = lab_type
        
        lab_results_24h['lab_type'] = lab_results_24h['itemid'].map(itemid_mapping)
        
        # 환자별, 검사별 집계
        aggregated_labs = []
        
        for stay_id in lab_results_24h['stay_id'].unique():
            if pd.isna(stay_id):
                continue
                
            patient_labs = lab_results_24h[lab_results_24h['stay_id'] == stay_id]
            
            lab_stats = {'stay_id': stay_id}
            
            for lab_type in patient_labs['lab_type'].unique():
                if pd.isna(lab_type):
                    continue
                
                lab_values = pd.to_numeric(patient_labs[patient_labs['lab_type'] == lab_type]['valuenum'], errors='coerce')
                valid_values = lab_values.dropna()
                
                if len(valid_values) > 0:
                    lab_stats[f'{lab_type}_mean'] = valid_values.mean()
                    lab_stats[f'{lab_type}_std'] = valid_values.std()
                    lab_stats[f'{lab_type}_max'] = valid_values.max()
                    lab_stats[f'{lab_type}_min'] = valid_values.min()
                    lab_stats[f'{lab_type}_count'] = len(valid_values)
            
            aggregated_labs.append(lab_stats)
        
        aggregated_df = pd.DataFrame(aggregated_labs)
        print(f"집계된 검사 결과: {len(aggregated_df):,}명")
        
        return aggregated_df
    
    def create_final_dataset(self, icu_stays, mortality_data, comorbidities, lifestyle_omr, anthropometric_omr, vital_signs_agg, lab_results_agg):
        """최종 데이터셋 생성"""
        print("최종 데이터셋 생성 중...")
        
        # 기본 정보로 시작 (hospital_expire_flag, los 제거 - Data Leakage 방지)
        final_dataset = mortality_data[['subject_id', 'hadm_id', 'stay_id', 'gender', 'anchor_age', 
                                       'first_careunit', 'mortality_48h']].copy()
        
        print(f"기본 정보: {len(final_dataset)}명")
        
        # 동반질환 정보 병합
        if not comorbidities.empty:
            final_dataset = final_dataset.merge(comorbidities, on='subject_id', how='left')
            print("동반질환 정보 병합 완료")
        
        # OMR 라이프스타일 정보 병합
        if not lifestyle_omr.empty:
            final_dataset = final_dataset.merge(lifestyle_omr, on='subject_id', how='left')
            print("OMR 라이프스타일 정보 병합 완료")
        
        # OMR 신체측정 정보 병합
        if not anthropometric_omr.empty:
            final_dataset = final_dataset.merge(anthropometric_omr, on='subject_id', how='left')
            print("OMR 신체측정 정보 병합 완료")
        
        # 생체징후 정보 병합
        if not vital_signs_agg.empty:
            final_dataset = final_dataset.merge(vital_signs_agg, on='stay_id', how='left')
            print("생체징후 정보 병합 완료")
        
        # 검사 결과 병합
        if not lab_results_agg.empty:
            final_dataset = final_dataset.merge(lab_results_agg, on='stay_id', how='left')
            print("검사 결과 병합 완료")
        
        # 성별을 더미 변수로 변환
        final_dataset['gender_male'] = (final_dataset['gender'] == 'M').astype(int)
        
        # ICU 유형을 더미 변수로 변환 (주요 유형들만)
        major_units = ['Medical Intensive Care Unit (MICU)', 'Surgical Intensive Care Unit (SICU)', 
                       'Cardiac Vascular Intensive Care Unit (CVICU)', 'Coronary Care Unit (CCU)']
        
        for unit in major_units:
            unit_name = unit.split('(')[0].strip().replace(' ', '_').lower()
            final_dataset[f'icu_{unit_name}'] = (final_dataset['first_careunit'] == unit).astype(int)
        
        # 라이프스타일 통합 변수 생성 (ICD + OMR)
        if 'smoking_icd' in final_dataset.columns and 'smoking_omr' in final_dataset.columns:
            final_dataset['smoking'] = ((final_dataset['smoking_icd'] == 1) | (final_dataset['smoking_omr'] == 1)).astype(int)
        elif 'smoking_icd' in final_dataset.columns:
            final_dataset['smoking'] = final_dataset['smoking_icd']
        elif 'smoking_omr' in final_dataset.columns:
            final_dataset['smoking'] = final_dataset['smoking_omr']
        
        if 'alcohol_icd' in final_dataset.columns and 'alcohol_omr' in final_dataset.columns:
            final_dataset['alcohol'] = ((final_dataset['alcohol_icd'] == 1) | (final_dataset['alcohol_omr'] == 1)).astype(int)
        elif 'alcohol_icd' in final_dataset.columns:
            final_dataset['alcohol'] = final_dataset['alcohol_icd']
        elif 'alcohol_omr' in final_dataset.columns:
            final_dataset['alcohol'] = final_dataset['alcohol_omr']
        
        print(f"최종 데이터셋: {len(final_dataset):,}명, {len(final_dataset.columns)}개 변수")
        
        return final_dataset
    
    def save_extraction_summary(self, final_dataset):
        """데이터 추출 요약 정보 저장"""
        summary_path = os.path.join(self.output_dir, "extraction_summary.txt")
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("MIMIC IV 중환자실 사망률 예측 데이터 추출 요약\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"추출 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 1. 환자 선정 기준
            f.write("1. 환자 선정 기준\n")
            f.write("-" * 30 + "\n")
            f.write("- 대상: 중환자실(ICU)에 입원한 모든 환자\n")
            f.write("- 기준: 각 환자의 첫 번째 ICU 입원만 포함\n")
            f.write("- 이유: 동일 환자의 재입원으로 인한 데이터 중복 방지\n")
            f.write(f"- 최종 선정 환자 수: {len(final_dataset):,}명\n\n")
            
            # 2. 데이터 수집 범위
            f.write("2. 데이터 수집 범위\n")
            f.write("-" * 30 + "\n")
            f.write("- 시간 범위: ICU 입실 후 24시간 내 데이터\n")
            f.write("- 예측 대상: 48시간 이내 사망률 (7일 사망률 제외)\n")
            f.write("- 데이터 소스: MIMIC IV v3.1\n\n")
            
            # 3. 48시간 사망률 통계만
            if 'mortality_48h' in final_dataset.columns:
                mort_48h = final_dataset['mortality_48h'].sum()
                mort_48h_rate = mort_48h / len(final_dataset) * 100
                f.write("3. 48시간 사망률 통계\n")
                f.write("-" * 30 + "\n")
                f.write(f"- 48시간 이내 사망: {mort_48h:,}명 ({mort_48h_rate:.2f}%)\n")
                f.write(f"- 클래스 불균형 비율: {(len(final_dataset) - mort_48h)/mort_48h:.1f}:1 (생존:사망)\n\n")
            
            # 4. 대사질환 보유 현황
            f.write("4. 대사질환 보유 현황\n")
            f.write("-" * 30 + "\n")
            disease_columns = ['diabetes', 'hypertension', 'obesity', 'dyslipidemia', 'thyroid_disease', 
                               'kidney_disease', 'cardiovascular_disease', 'cerebrovascular_disease', 'sleep_disorder']
            disease_names = ['당뇨병', '고혈압', '비만', '이상지질혈증', '갑상선질환', '신장질환', '심혈관질환', '뇌혈관질환', '수면장애']
            
            for col, name in zip(disease_columns, disease_names):
                if col in final_dataset.columns:
                    count = final_dataset[col].sum()
                    rate = count / len(final_dataset) * 100
                    f.write(f"- {name}: {count:,}명 ({rate:.1f}%)\n")
            f.write("\n")
            
            # 5. 라이프스타일 정보
            f.write("5. 라이프스타일 정보\n")
            f.write("-" * 30 + "\n")
            if 'smoking' in final_dataset.columns:
                smoking_count = final_dataset['smoking'].sum()
                smoking_rate = smoking_count / len(final_dataset) * 100
                f.write(f"- 흡연력: {smoking_count:,}명 ({smoking_rate:.1f}%)\n")
            
            if 'alcohol' in final_dataset.columns:
                alcohol_count = final_dataset['alcohol'].sum()
                alcohol_rate = alcohol_count / len(final_dataset) * 100
                f.write(f"- 알코올 사용력: {alcohol_count:,}명 ({alcohol_rate:.1f}%)\n")
            
            if 'pregnancy' in final_dataset.columns:
                pregnancy_count = final_dataset['pregnancy'].sum()
                pregnancy_rate = pregnancy_count / len(final_dataset) * 100
                f.write(f"- 임신: {pregnancy_count:,}명 ({pregnancy_rate:.1f}%)\n")
            f.write("\n")
            
            # 6. 신체측정 정보 (OMR 기반)
            f.write("6. 신체측정 정보 추출 방법\n")
            f.write("-" * 30 + "\n")
            f.write("- 데이터 소스: OMR (Outpatient Medical Records) 테이블\n")
            f.write("- 추출 방법:\n")
            f.write("  - 키: 'Height (Inches)', 'Height' → cm 변환\n")
            f.write("  - 몸무게: 'Weight (Lbs)', 'Weight' → kg 변환\n")
            f.write("  - BMI: 'BMI (kg/m2)', 'BMI' 직접 추출 (우선)\n")
            f.write("  - BMI 없으면 키/몸무게로 자동 계산\n")
            f.write("  - 환자별 최초 측정값 사용 (가장 오래된 데이터)\n")
            f.write("  - 모든 측정값 보존 (이상치 필터링 없음)\n\n")
            
            # 7. 변수별 한글명 및 설명
            f.write("7. 주요 변수별 한글명 및 설명\n")
            f.write("-" * 30 + "\n")
            f.write("기본 정보:\n")
            f.write("- subject_id: 환자 고유번호\n")
            f.write("- gender_male: 성별 (남성=1, 여성=0)\n")
            f.write("- anchor_age: 나이\n\n")
            
            f.write("신체측정:\n")
            f.write("- height_cm: 키 (cm)\n")
            f.write("- weight_kg: 몸무게 (kg)\n")
            f.write("- bmi: 체질량지수 (kg/m²)\n\n")
            
            f.write("생체징후 (24시간 내 평균, 최대, 최소, 표준편차):\n")
            f.write("- sbp_*: 수축기혈압 (mmHg)\n")
            f.write("- dbp_*: 이완기혈압 (mmHg)\n")
            f.write("- mbp_*: 평균혈압 (mmHg)\n")
            f.write("- heart_rate_*: 심박수 (bpm)\n")
            f.write("- respiratory_rate_*: 호흡수 (/min)\n")
            f.write("- temperature_*: 체온 (°C)\n")
            f.write("- spo2_*: 산소포화도 (%)\n")
            f.write("- rass_*: Richmond 진정-초조 척도\n")
            f.write("- gcs_*: Glasgow 혼수 척도\n\n")
            
            f.write("검사수치 (24시간 내 평균, 표준편차, 최대, 최소):\n")
            f.write("- glucose_*: 혈당 (mg/dL)\n")
            f.write("- creatinine_*: 크레아티닌 (mg/dL)\n")
            f.write("- bun_*: 혈중요소질소 (mg/dL)\n")
            f.write("- sodium_*: 나트륨 (mEq/L)\n")
            f.write("- potassium_*: 칼륨 (mEq/L)\n")
            f.write("- hematocrit_*: 헤마토크리트 (%)\n")
            f.write("- hemoglobin_*: 헤모글로빈 (g/dL)\n")
            f.write("- wbc_*: 백혈구 (K/uL)\n\n")
            
            f.write("종속변수:\n")
            f.write("- mortality_48h: 48시간 이내 사망 여부 (1=사망, 0=생존)\n\n")
            
            # 8. 결측치 분석
            f.write("8. 결측치 분석\n")
            f.write("-" * 30 + "\n")
            
            missing_analysis = final_dataset.isnull().sum().sort_values(ascending=False)
            missing_percent = (missing_analysis / len(final_dataset) * 100).round(1)
            
            high_missing = missing_percent[missing_percent >= 10]  # 10% 이상 결측
            
            f.write("결측율 10% 이상인 변수들:\n")
            for var, pct in high_missing.items():
                f.write(f"- {var}: {pct}%\n")
            
            f.write(f"\n총 변수 수: {len(final_dataset.columns)}개\n")
            f.write(f"결측율 10% 미만 변수: {len(missing_percent[missing_percent < 10])}개\n")
            f.write(f"결측율 50% 미만 변수: {len(missing_percent[missing_percent < 50])}개\n")
        
        print(f"추출 요약 저장 완료: {summary_path}")

    def run_extraction(self):
        """전체 데이터 추출 프로세스 실행"""
        print("MIMIC IV 사망률 예측 데이터 추출 시작")
        print("=" * 60)
        
        # 1. 첫 번째 ICU 재원 정보 추출
        first_icu_stays = self.get_first_icu_stays()
        subject_ids = first_icu_stays['subject_id'].unique()
        
        # 2. 사망률 라벨 계산
        mortality_data = self.calculate_mortality_labels(first_icu_stays)
        
        # 3. 동반질환 정보 추출
        comorbidities = self.extract_comorbidities(subject_ids)
        
        # 4. OMR 라이프스타일 정보 추출
        lifestyle_omr = self.extract_omr_lifestyle(subject_ids)
        
        # 5. OMR 신체측정 정보 추출
        anthropometric_omr = self.extract_omr_anthropometric(subject_ids)
        
        # 6. 생체징후 추출
        vital_signs = self.extract_vital_signs(first_icu_stays)
        vital_signs_agg = self.aggregate_vital_signs(vital_signs, first_icu_stays)
        
        # 7. 검사 결과 추출
        lab_results = self.extract_lab_results(first_icu_stays)
        lab_results_agg = self.aggregate_lab_results(lab_results, first_icu_stays)
        
        # 8. 최종 데이터셋 생성
        final_dataset = self.create_final_dataset(
            first_icu_stays, mortality_data, comorbidities, lifestyle_omr, 
            anthropometric_omr, vital_signs_agg, lab_results_agg
        )
        
        # 9. 데이터셋 저장
        output_path = os.path.join(self.output_dir, "mimic_mortality_raw.csv")
        final_dataset.to_csv(output_path, index=False)
        print(f"최종 데이터셋 저장 완료: {output_path}")
        
        # 10. 추출 요약 저장
        self.save_extraction_summary(final_dataset)
        
        print("=" * 60)
        print("데이터 추출 완료!")
        print(f"최종 결과: {len(final_dataset):,}명, {len(final_dataset.columns)}개 변수")
        print(f"출력 경로: {output_path}")


def main():
    """메인 실행 함수"""
    # MIMIC IV 데이터 경로 설정 - 사용자가 수정해야 함
    # TODO: 아래 경로를 실제 MIMIC-IV 데이터가 위치한 경로로 변경하세요
    data_root = "C:/path/to/your/mimic-iv/3.1"  # 예: "/path/to/mimic-iv/3.1" 또는 "C:/data/mimic-iv/3.1"
    
    # 출력 디렉토리 설정 (선택사항 - 기본값 사용하려면 None)
    output_dir = None  # 예: "/path/to/output/dataset/0_raw" 또는 기본값 사용시 None
    
    print("MIMIC IV 중환자실 사망률 예측 데이터 추출")
    print(f"데이터 경로: {data_root}")
    print("-" * 60)
    
    # 경로 유효성 확인
    if not os.path.exists(data_root):
        print("❌ 오류: MIMIC-IV 데이터 경로가 존재하지 않습니다!")
        print(f"   입력한 경로: {data_root}")
        print("   해결 방법:")
        print("   1. main() 함수에서 data_root 변수를 올바른 경로로 수정하세요")
        print("   2. MIMIC-IV 데이터가 올바르게 압축 해제되었는지 확인하세요")
        print("   3. PhysioNet에서 승인받은 정식 MIMIC-IV v3.1 데이터인지 확인하세요")
        return
    
    try:
        # 데이터 추출기 초기화 및 실행
        extractor = MIMICMortalityDataExtractor(data_root, output_dir)
        extractor.run_extraction()
        
    except Exception as e:
        print(f"데이터 추출 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()