#!/usr/bin/env python3
"""
Figure Generator
전체 분석 과정의 모든 시각화를 한 번에 생성하는 스크립트

생성 대상:
1. 데이터 분포 시각화 
2. 결측치 분석 히트맵
3. 클래스 불균형 비교
4. 리샘플링 효과 비교
5. 모델 성능 비교
6. 최종 결과 대시보드
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import json
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

class FigureGenerator:
    def __init__(self):
        # 프로젝트 루트 디렉토리 찾기
        self.base_path = Path(__file__).parent.parent
        self.dataset_path = self.base_path / "dataset"
        self.output_path = self.base_path / "figures"
        self.output_path.mkdir(exist_ok=True)
        
        # 색상 팔레트 설정
        self.colors = {
            'primary': '#2E86C1',
            'secondary': '#F39C12', 
            'success': '#27AE60',
            'danger': '#E74C3C',
            'warning': '#F1C40F',
            'info': '#8E44AD'
        }
        
    def load_data_safely(self, file_path, description="데이터"):
        """안전한 데이터 로드"""
        try:
            if Path(file_path).exists():
                df = pd.read_csv(file_path)
                print(f"✅ {description} 로드: {df.shape}")
                return df
            else:
                print(f"⚠️ {description} 파일이 없습니다: {file_path}")
                return None
        except Exception as e:
            print(f"❌ {description} 로드 실패: {e}")
            return None
    
    def create_data_distribution_plots(self):
        """데이터 분포 시각화"""
        print("\n📊 데이터 분포 시각화 생성 중...")
        
        # 원본 데이터 로드
        raw_data = self.load_data_safely(self.dataset_path / "0_raw/mimic_mortality_raw.csv", "원본 데이터")
        cleaned_data = self.load_data_safely(self.dataset_path / "1_cleaned/mimic_mortality_cleaned.csv", "정제 데이터")
        
        if raw_data is None and cleaned_data is None:
            print("❌ 시각화할 데이터가 없습니다.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('MIMIC-IV 48-hour Mortality Prediction - Data Distribution', fontsize=16, fontweight='bold')
        
        # 1. 사망률 분포 (원본 vs 정제 후)
        if raw_data is not None and 'mortality_48h' in raw_data.columns:
            raw_mortality = raw_data['mortality_48h'].mean()
            axes[0,0].bar(['Raw Data'], [raw_mortality], color=self.colors['primary'], alpha=0.7, label='Raw')
        
        if cleaned_data is not None and 'mortality_48h' in cleaned_data.columns:
            cleaned_mortality = cleaned_data['mortality_48h'].mean() 
            axes[0,0].bar(['After Cleaning'], [cleaned_mortality], color=self.colors['success'], alpha=0.7, label='After Cleaning')
        
        axes[0,0].set_ylabel('48-hour Mortality Rate')
        axes[0,0].set_title('Mortality Rate Comparison Before/After Cleaning')
        axes[0,0].legend()
        
        # 2. 연령 분포
        data_for_age = cleaned_data if cleaned_data is not None else raw_data
        if data_for_age is not None:
            age_col = 'anchor_age' if 'anchor_age' in data_for_age.columns else 'age'
            if age_col in data_for_age.columns:
                axes[0,1].hist(data_for_age[age_col].dropna(), bins=30, color=self.colors['secondary'], alpha=0.7, edgecolor='black')
                axes[0,1].set_xlabel('Age')
                axes[0,1].set_ylabel('Frequency')
                axes[0,1].set_title('Patient Age Distribution')
        
        # 3. 성별 분포
        if data_for_age is not None and 'gender' in data_for_age.columns:
            gender_counts = data_for_age['gender'].value_counts()
            colors = [self.colors['info'], self.colors['warning']]
            axes[1,0].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', 
                         colors=colors, startangle=90)
            axes[1,0].set_title('Gender Distribution')
        
        # 4. ICU 유형 분포
        if data_for_age is not None and 'first_careunit' in data_for_age.columns:
            icu_counts = data_for_age['first_careunit'].value_counts().head(6)
            axes[1,1].barh(range(len(icu_counts)), icu_counts.values, color=self.colors['primary'])
            axes[1,1].set_yticks(range(len(icu_counts)))
            axes[1,1].set_yticklabels([label[:15] + '...' if len(label) > 15 else label for label in icu_counts.index])
            axes[1,1].set_xlabel('Number of Patients')
            axes[1,1].set_title('ICU Type Distribution (Top 6)')
        
        plt.tight_layout()
        plt.savefig(self.output_path / "01_data_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 데이터 분포 시각화 저장")
    
    def create_missing_data_heatmap(self):
        """결측치 분석 히트맵"""
        print("\n📊 결측치 분석 히트맵 생성 중...")
        
        raw_data = self.load_data_safely(self.dataset_path / "0_raw/mimic_mortality_raw.csv", "원본 데이터")
        
        if raw_data is None:
            print("❌ 결측치 분석을 위한 데이터가 없습니다.")
            return
        
        # 결측치 비율 계산
        missing_ratios = raw_data.isnull().sum() / len(raw_data)
        missing_ratios = missing_ratios[missing_ratios > 0].sort_values(ascending=False)
        
        if len(missing_ratios) == 0:
            print("✅ 결측치가 없습니다.")
            return
        
        # 상위 30개 변수만 표시
        top_missing = missing_ratios.head(30)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Missing Values Analysis', fontsize=16, fontweight='bold')
        
        # 1. 결측치 비율 막대그래프
        y_pos = np.arange(len(top_missing))
        bars = ax1.barh(y_pos, top_missing.values, color=self.colors['danger'], alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(top_missing.index, fontsize=8)
        ax1.set_xlabel('Missing Value Ratio')
        ax1.set_title('Missing Value Ratio by Variable (Top 30)')
        ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.8, label='50% Threshold')
        ax1.legend()
        
        # 막대에 값 표시
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.1%}', ha='left', va='center', fontsize=7)
        
        # 2. 결측치 분포 히스토그램
        ax2.hist(missing_ratios.values, bins=20, color=self.colors['warning'], alpha=0.7, edgecolor='black')
        ax2.axvline(x=0.5, color='red', linestyle='--', alpha=0.8, label='50% Threshold')
        ax2.set_xlabel('Missing Value Ratio')
        ax2.set_ylabel('Number of Variables')
        ax2.set_title('Missing Value Ratio Distribution')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_path / "02_missing_data_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 결측치 분석 히트맵 저장")
        
        # 추가: 결측치 제거 전후 비교 시각화
        self.create_missing_data_impact_visualization()
    
    def create_missing_data_impact_visualization(self):
        """결측치 제거 전후 데이터 건수 및 클래스 불균형 비교"""
        print("\n📊 결측치 제거 영향 분석 시각화 생성 중...")
        
        # 원본 데이터와 정제 데이터 로드
        raw_data = self.load_data_safely(self.dataset_path / "0_raw/mimic_mortality_raw.csv", "원본 데이터")
        cleaned_data = self.load_data_safely(self.dataset_path / "1_cleaned/mimic_mortality_cleaned.csv", "정제 데이터")
        
        if raw_data is None or cleaned_data is None:
            print("❌ 결측치 영향 분석을 위한 데이터가 없습니다.")
            return
        
        # 2x2 레이아웃으로 시각화
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Missing Data Removal Impact Analysis', fontsize=16, fontweight='bold')
        
        # 데이터 통계 수집
        impact_stats = self.collect_missing_data_impact_stats(raw_data, cleaned_data)
        
        # 1. 데이터 건수 변화 (Sankey diagram 스타일)
        self.plot_data_reduction_flow(axes[0, 0], impact_stats)
        
        # 2. 클래스 불균형 변화
        self.plot_class_imbalance_change(axes[0, 1], impact_stats)
        
        # 3. 사망률 변화 및 통계
        self.plot_mortality_rate_change(axes[1, 0], impact_stats)
        
        # 4. 제거된 데이터의 특성 분석
        self.plot_removed_data_analysis(axes[1, 1], impact_stats)
        
        plt.tight_layout()
        plt.savefig(self.output_path / "02_missing_data_impact.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 결측치 제거 영향 분석 시각화 저장")
    
    def collect_missing_data_impact_stats(self, raw_data, cleaned_data):
        """결측치 제거 영향 통계 수집"""
        stats = {}
        
        # 기본 통계
        stats['raw_count'] = len(raw_data)
        stats['cleaned_count'] = len(cleaned_data)
        stats['removed_count'] = stats['raw_count'] - stats['cleaned_count']
        stats['removal_rate'] = stats['removed_count'] / stats['raw_count']
        
        # 사망률 계산
        if 'mortality_48h' in raw_data.columns:
            stats['raw_mortality_rate'] = raw_data['mortality_48h'].mean()
            stats['raw_mortality_count'] = raw_data['mortality_48h'].sum()
        else:
            stats['raw_mortality_rate'] = 0
            stats['raw_mortality_count'] = 0
            
        if 'mortality_48h' in cleaned_data.columns:
            stats['cleaned_mortality_rate'] = cleaned_data['mortality_48h'].mean()
            stats['cleaned_mortality_count'] = cleaned_data['mortality_48h'].sum()
        else:
            stats['cleaned_mortality_rate'] = 0
            stats['cleaned_mortality_count'] = 0
        
        # 클래스 불균형 비율 계산
        if stats['raw_mortality_count'] > 0:
            stats['raw_imbalance_ratio'] = (stats['raw_count'] - stats['raw_mortality_count']) / stats['raw_mortality_count']
        else:
            stats['raw_imbalance_ratio'] = 0
            
        if stats['cleaned_mortality_count'] > 0:
            stats['cleaned_imbalance_ratio'] = (stats['cleaned_count'] - stats['cleaned_mortality_count']) / stats['cleaned_mortality_count']
        else:
            stats['cleaned_imbalance_ratio'] = 0
        
        # 로그 출력
        print(f"  원본 데이터: {stats['raw_count']:,}명 (사망률: {stats['raw_mortality_rate']:.1%})")
        print(f"  정제 데이터: {stats['cleaned_count']:,}명 (사망률: {stats['cleaned_mortality_rate']:.1%})")
        print(f"  제거된 데이터: {stats['removed_count']:,}명 ({stats['removal_rate']:.1%})")
        
        return stats
    
    def plot_data_reduction_flow(self, ax, stats):
        """데이터 감소 플로우 차트"""
        # 단계별 데이터
        stages = ['Before\nMissing Removal', 'After\nMissing Removal', 'Removed\nData']
        values = [stats['raw_count'], stats['cleaned_count'], stats['removed_count']]
        colors = [self.colors['primary'], self.colors['success'], self.colors['danger']]
        
        bars = ax.bar(stages, values, color=colors, alpha=0.7)
        
        # 값 표시
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + max(values)*0.01,
                   f'{value:,}', ha='center', va='bottom', fontweight='bold')
            
            # 비율 표시
            if bar == bars[1]:  # After
                retention_rate = value / stats['raw_count'] * 100
                ax.text(bar.get_x() + bar.get_width()/2, height/2,
                       f'{retention_rate:.1f}%\nretained', ha='center', va='center', 
                       fontweight='bold', color='white', fontsize=10)
            elif bar == bars[2]:  # Removed
                removal_rate = value / stats['raw_count'] * 100
                ax.text(bar.get_x() + bar.get_width()/2, height/2,
                       f'{removal_rate:.1f}%\nremoved', ha='center', va='center', 
                       fontweight='bold', color='white', fontsize=10)
        
        ax.set_ylabel('Number of Patients')
        ax.set_title('Data Reduction by Missing Value Removal', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    def plot_class_imbalance_change(self, ax, stats):
        """클래스 불균형 변화"""
        categories = ['Before\nRemoval', 'After\nRemoval']
        imbalance_ratios = [stats['raw_imbalance_ratio'], stats['cleaned_imbalance_ratio']]
        
        bars = ax.bar(categories, imbalance_ratios, 
                     color=[self.colors['warning'], self.colors['info']], alpha=0.7)
        
        # 값 표시
        for bar, ratio in zip(bars, imbalance_ratios):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(imbalance_ratios)*0.01,
                   f'{ratio:.1f}:1', ha='center', va='bottom', fontweight='bold')
        
        # 개선/악화 표시
        if len(imbalance_ratios) >= 2:
            change = imbalance_ratios[1] - imbalance_ratios[0]
            change_text = "Better" if change < 0 else "Worse"
            change_color = "green" if change < 0 else "red"
            ax.text(0.5, max(imbalance_ratios) * 0.8, f'{change_text}\n({change:+.1f})', 
                   ha='center', va='center', fontweight='bold', color=change_color,
                   transform=ax.transData)
        
        ax.set_ylabel('Imbalance Ratio (Survived:Died)')
        ax.set_title('Class Imbalance Change', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    def plot_mortality_rate_change(self, ax, stats):
        """사망률 변화"""
        categories = ['Before Removal', 'After Removal']
        mortality_rates = [stats['raw_mortality_rate'] * 100, stats['cleaned_mortality_rate'] * 100]
        
        bars = ax.bar(categories, mortality_rates,
                     color=[self.colors['secondary'], self.colors['primary']], alpha=0.7)
        
        # 값 표시
        for bar, rate in zip(bars, mortality_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mortality_rates)*0.01,
                   f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 변화량 표시
        if len(mortality_rates) >= 2:
            change = mortality_rates[1] - mortality_rates[0]
            ax.text(0.5, max(mortality_rates) * 0.5, f'{change:+.1f}%p', 
                   ha='center', va='center', fontweight='bold', 
                   color='red' if change > 0 else 'green', fontsize=12)
        
        ax.set_ylabel('Mortality Rate (%)')
        ax.set_title('48-hour Mortality Rate Change', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    def plot_removed_data_analysis(self, ax, stats):
        """제거된 데이터 분석 (파이 차트)"""
        # 제거된 데이터와 보존된 데이터 비율
        labels = ['Retained Data', 'Removed Data']
        sizes = [stats['cleaned_count'], stats['removed_count']]
        colors = [self.colors['success'], self.colors['danger']]
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                         colors=colors, startangle=90, 
                                         explode=(0, 0.1))  # 제거된 부분 강조
        
        # 개수도 표시
        for autotext, size in zip(autotexts, sizes):
            autotext.set_text(f'{size:,}\n({autotext.get_text()})')
            autotext.set_fontsize(9)
            autotext.set_fontweight('bold')
        
        # 제목과 추가 정보
        ax.set_title('Data Retention vs Removal', fontweight='bold')
        
        # 텍스트 박스로 추가 정보
        info_text = f"Total Original: {stats['raw_count']:,}\nRemoval Rate: {stats['removal_rate']:.1%}"
        ax.text(1.2, 0.5, info_text, transform=ax.transAxes, 
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5),
               fontsize=10, ha='left', va='center')
    
    def create_resampling_comparison(self):
        """리샘플링 효과 비교"""
        print("\n📊 리샘플링 효과 비교 시각화 생성 중...")
        
        # 분할된 원본 데이터와 리샘플링 데이터 로드
        original_train = self.load_data_safely(self.dataset_path / "2_split/mimic_mortality_train.csv", "원본 Train")
        smote_train = self.load_data_safely(self.dataset_path / "3_resampled/smote/mimic_mortality_train.csv", "SMOTE Train")
        down_train = self.load_data_safely(self.dataset_path / "3_resampled/downsampling/mimic_mortality_train.csv", "Downsampling Train")
        
        datasets = [
            ("Original", original_train, self.colors['primary']),
            ("SMOTE", smote_train, self.colors['success']),
            ("Downsampling", down_train, self.colors['warning'])
        ]
        
        available_datasets = [(name, data, color) for name, data, color in datasets if data is not None]
        
        if len(available_datasets) == 0:
            print("❌ 리샘플링 비교를 위한 데이터가 없습니다.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Resampling Effect Comparison', fontsize=16, fontweight='bold')
        
        # 1. 데이터 크기 비교
        sizes = []
        names = []
        colors = []
        
        for name, data, color in available_datasets:
            if data is not None:
                sizes.append(len(data))
                names.append(name)
                colors.append(color)
        
        bars = axes[0,0].bar(names, sizes, color=colors, alpha=0.7)
        axes[0,0].set_ylabel('Number of Samples')
        axes[0,0].set_title('Dataset Size Comparison')
        
        # 막대에 값 표시
        for bar, size in zip(bars, sizes):
            axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sizes)*0.01,
                          f'{size:,}', ha='center', va='bottom', fontweight='bold')
        
        # 2. 클래스 분포 비교 (비율)
        ratios = []
        for name, data, color in available_datasets:
            if data is not None and 'mortality_48h' in data.columns:
                mortality_rate = data['mortality_48h'].mean()
                ratios.append([1-mortality_rate, mortality_rate])
            else:
                ratios.append([0, 0])
        
        if ratios:
            ratios = np.array(ratios)
            x = np.arange(len(names))
            width = 0.35
            
            axes[0,1].bar(x, ratios[:, 0], width, label='Survived (0)', color=self.colors['info'], alpha=0.7)
            axes[0,1].bar(x, ratios[:, 1], width, bottom=ratios[:, 0], label='Died (1)', color=self.colors['danger'], alpha=0.7)
            
            axes[0,1].set_ylabel('Ratio')
            axes[0,1].set_title('Class Distribution Comparison')
            axes[0,1].set_xticks(x)
            axes[0,1].set_xticklabels(names)
            axes[0,1].legend()
        
        # 3. 불균형 비율 비교
        imbalance_ratios = []
        for name, data, color in available_datasets:
            if data is not None and 'mortality_48h' in data.columns:
                counts = data['mortality_48h'].value_counts()
                if len(counts) >= 2:
                    ratio = counts[0] / counts[1]
                    imbalance_ratios.append(ratio)
                else:
                    imbalance_ratios.append(0)
            else:
                imbalance_ratios.append(0)
        
        bars = axes[1,0].bar(names, imbalance_ratios, color=colors, alpha=0.7)
        axes[1,0].axhline(y=1, color='red', linestyle='--', alpha=0.8, label='Perfect Balance')
        axes[1,0].set_ylabel('Imbalance Ratio (Survived:Died)')
        axes[1,0].set_title('Class Imbalance Ratio')
        axes[1,0].legend()
        
        # 막대에 값 표시
        for bar, ratio in zip(bars, imbalance_ratios):
            axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(imbalance_ratios)*0.02,
                          f'{ratio:.1f}:1', ha='center', va='bottom', fontweight='bold')
        
        # 4. 사망률 비교
        mortality_rates = []
        for name, data, color in available_datasets:
            if data is not None and 'mortality_48h' in data.columns:
                mortality_rate = data['mortality_48h'].mean()
                mortality_rates.append(mortality_rate)
            else:
                mortality_rates.append(0)
        
        bars = axes[1,1].bar(names, mortality_rates, color=colors, alpha=0.7)
        axes[1,1].set_ylabel('Mortality Rate')
        axes[1,1].set_title('48-hour Mortality Rate Comparison')
        
        # 막대에 값 표시
        for bar, rate in zip(bars, mortality_rates):
            axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mortality_rates)*0.01,
                          f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_path / "03_resampling_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 리샘플링 효과 비교 저장")
        
        # 추가: 데이터 처리 파이프라인 크기 변화 시각화
        self.create_data_pipeline_visualization()
    
    def create_data_pipeline_visualization(self):
        """데이터 처리 파이프라인 단계별 크기 변화 시각화"""
        print("\n📊 데이터 처리 파이프라인 시각화 생성 중...")
        
        # 데이터 크기 수집
        pipeline_data = self.collect_pipeline_data()
        
        if not pipeline_data:
            print("❌ 파이프라인 데이터 수집 실패")
            return
        
        # 2x2 레이아웃으로 시각화
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Data Processing Pipeline - Dataset Size Changes', fontsize=16, fontweight='bold')
        
        # 1. 전체 파이프라인 플로우
        self.plot_pipeline_flow(axes[0, 0], pipeline_data)
        
        # 2. 분할 전후 비교
        self.plot_split_comparison(axes[0, 1], pipeline_data)
        
        # 3. 리샘플링 효과 (Train 세트)
        self.plot_resampling_effects(axes[1, 0], pipeline_data)
        
        # 4. 최종 데이터셋 분포 (파이 차트)
        self.plot_final_distribution(axes[1, 1], pipeline_data)
        
        plt.tight_layout()
        plt.savefig(self.output_path / "03_data_pipeline.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 데이터 처리 파이프라인 시각화 저장")
    
    def collect_pipeline_data(self):
        """파이프라인 각 단계의 데이터 크기 수집"""
        pipeline_data = {}
        
        # 데이터 파일 경로 정의
        data_files = {
            'raw': self.dataset_path / "0_raw/mimic_mortality_raw.csv",
            'cleaned': self.dataset_path / "1_cleaned/mimic_mortality_cleaned.csv", 
            'train_original': self.dataset_path / "2_split/mimic_mortality_train.csv",
            'val': self.dataset_path / "2_split/mimic_mortality_validation.csv",
            'test': self.dataset_path / "2_split/mimic_mortality_test.csv",
            'train_smote': self.dataset_path / "3_resampled/smote/mimic_mortality_train.csv",
            'train_downsampling': self.dataset_path / "3_resampled/downsampling/mimic_mortality_train.csv"
        }
        
        # 각 파일의 크기 및 사망률 수집
        for stage, file_path in data_files.items():
            data = self.load_data_safely(file_path, f"{stage} 데이터")
            if data is not None:
                size = len(data)
                mortality_rate = data['mortality_48h'].mean() if 'mortality_48h' in data.columns else 0
                mortality_count = data['mortality_48h'].sum() if 'mortality_48h' in data.columns else 0
                
                pipeline_data[stage] = {
                    'size': size,
                    'mortality_rate': mortality_rate,
                    'mortality_count': int(mortality_count),
                    'survival_count': int(size - mortality_count)
                }
                print(f"  {stage}: {size:,}명 (사망률: {mortality_rate:.1%})")
            else:
                pipeline_data[stage] = None
        
        return pipeline_data
    
    def plot_pipeline_flow(self, ax, pipeline_data):
        """전체 파이프라인 플로우 차트"""
        stages = ['raw', 'cleaned', 'train_original', 'train_smote', 'train_downsampling']
        stage_names = ['Raw Data', 'Cleaned', 'Train Split', 'SMOTE Train', 'Downsample Train']
        
        sizes = []
        colors = []
        
        for stage in stages:
            if pipeline_data.get(stage):
                sizes.append(pipeline_data[stage]['size'])
                if 'raw' in stage:
                    colors.append(self.colors['primary'])
                elif 'cleaned' in stage:
                    colors.append(self.colors['secondary'])
                elif 'original' in stage:
                    colors.append(self.colors['info'])
                elif 'smote' in stage:
                    colors.append(self.colors['success'])
                else:
                    colors.append(self.colors['warning'])
            else:
                sizes.append(0)
                colors.append('gray')
        
        # 막대 그래프
        bars = ax.bar(range(len(stage_names)), sizes, color=colors, alpha=0.7)
        
        # 값 표시
        for i, (bar, size) in enumerate(zip(bars, sizes)):
            if size > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sizes)*0.01,
                       f'{size:,}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        ax.set_xticks(range(len(stage_names)))
        ax.set_xticklabels(stage_names, rotation=45, ha='right')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Data Pipeline Flow', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    def plot_split_comparison(self, ax, pipeline_data):
        """분할 전후 비교"""
        # 분할 전 (cleaned) vs 분할 후 (train+val+test)
        datasets = {}
        
        if pipeline_data.get('cleaned'):
            datasets['Before Split\n(Cleaned)'] = pipeline_data['cleaned']['size']
        
        # 분할 후 총합
        split_total = 0
        for stage in ['train_original', 'val', 'test']:
            if pipeline_data.get(stage):
                split_total += pipeline_data[stage]['size']
        
        if split_total > 0:
            datasets['After Split\n(Train+Val+Test)'] = split_total
        
        if datasets:
            names = list(datasets.keys())
            sizes = list(datasets.values())
            colors = [self.colors['secondary'], self.colors['info']]
            
            bars = ax.bar(names, sizes, color=colors, alpha=0.7)
            
            # 값 표시
            for bar, size in zip(bars, sizes):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sizes)*0.01,
                       f'{size:,}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_ylabel('Number of Samples')
            ax.set_title('Before/After Data Split', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
    
    def plot_resampling_effects(self, ax, pipeline_data):
        """리샘플링 효과 (Train 세트 크기 변화)"""
        train_datasets = {}
        colors = []
        
        if pipeline_data.get('train_original'):
            train_datasets['Original\nTrain'] = pipeline_data['train_original']['size']
            colors.append(self.colors['primary'])
        
        if pipeline_data.get('train_smote'):
            train_datasets['SMOTE\nTrain'] = pipeline_data['train_smote']['size']
            colors.append(self.colors['success'])
        
        if pipeline_data.get('train_downsampling'):
            train_datasets['Downsampling\nTrain'] = pipeline_data['train_downsampling']['size']
            colors.append(self.colors['warning'])
        
        if train_datasets:
            names = list(train_datasets.keys())
            sizes = list(train_datasets.values())
            
            bars = ax.bar(names, sizes, color=colors, alpha=0.7)
            
            # 값 표시 및 변화율 계산
            original_size = train_datasets.get('Original\nTrain', 0)
            for i, (bar, size) in enumerate(zip(bars, sizes)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sizes)*0.01,
                       f'{size:,}', ha='center', va='bottom', fontweight='bold', fontsize=9)
                
                # 변화율 표시 (원본 대비)
                if original_size > 0 and i > 0:
                    change_pct = (size - original_size) / original_size * 100
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                           f'{change_pct:+.0f}%', ha='center', va='center', 
                           fontweight='bold', color='white', fontsize=8)
            
            ax.set_ylabel('Number of Training Samples')
            ax.set_title('Resampling Effects on Training Set', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
    
    def plot_final_distribution(self, ax, pipeline_data):
        """최종 데이터셋 분포 (Val, Test, SMOTE Train, Downsample Train)"""
        final_datasets = {}
        
        # Val, Test는 고정
        if pipeline_data.get('val'):
            final_datasets['Validation'] = pipeline_data['val']['size']
        if pipeline_data.get('test'):
            final_datasets['Test'] = pipeline_data['test']['size']
        
        # 가장 큰 Train 세트 선택 (보통 SMOTE)
        max_train_size = 0
        max_train_name = ''
        for train_type in ['train_smote', 'train_downsampling', 'train_original']:
            if pipeline_data.get(train_type):
                size = pipeline_data[train_type]['size']
                if size > max_train_size:
                    max_train_size = size
                    max_train_name = train_type.replace('train_', '').replace('_', ' ').title()
        
        if max_train_size > 0:
            final_datasets[f'{max_train_name} Train'] = max_train_size
        
        if final_datasets:
            labels = list(final_datasets.keys())
            sizes = list(final_datasets.values())
            colors = [self.colors['success'], self.colors['info'], self.colors['warning']][:len(sizes)]
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                             colors=colors, startangle=90)
            
            # 개수도 표시
            for i, (autotext, size) in enumerate(zip(autotexts, sizes)):
                autotext.set_text(f'{size:,}\n({autotext.get_text()})')
                autotext.set_fontsize(8)
            
            ax.set_title('Final Dataset Distribution', fontweight='bold')
    
    def create_model_performance_plots(self):
        """모델 성능 비교 시각화"""
        print("\n📊 모델 성능 비교 시각화 생성 중...")
        
        # 모델링 결과 로드
        modeling_results = self.load_data_safely(self.dataset_path / "4_modeling/modeling_results.csv", "모델링 결과")
        tuning_results = self.load_data_safely(self.dataset_path / "5_final_models/tuning_results.csv", "튜닝 결과")
        
        if modeling_results is None:
            print("❌ 모델 성능 비교를 위한 데이터가 없습니다.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. 모델별 ROC-AUC 비교
        model_auc = modeling_results.groupby('model_name')['roc_auc'].mean().sort_values(ascending=True)
        
        bars = axes[0,0].barh(range(len(model_auc)), model_auc.values, color=self.colors['primary'], alpha=0.7)
        axes[0,0].set_yticks(range(len(model_auc)))
        axes[0,0].set_yticklabels(model_auc.index)
        axes[0,0].set_xlabel('ROC-AUC')
        axes[0,0].set_title('Average ROC-AUC by Model')
        
        # 값 표시
        for i, (bar, value) in enumerate(zip(bars, model_auc.values)):
            axes[0,0].text(value + 0.005, bar.get_y() + bar.get_height()/2,
                          f'{value:.3f}', va='center', fontweight='bold')
        
        # 2. 리샘플링별 성능 비교
        resampling_auc = modeling_results.groupby('resampling_method')['roc_auc'].mean()
        
        colors = [self.colors['success'], self.colors['warning']]
        bars = axes[0,1].bar(resampling_auc.index, resampling_auc.values, color=colors, alpha=0.7)
        axes[0,1].set_ylabel('Average ROC-AUC')
        axes[0,1].set_title('Performance by Resampling Method')
        
        for bar, value in zip(bars, resampling_auc.values):
            axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                          f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. 상위 10개 모델 성능
        top_10 = modeling_results.nlargest(10, 'roc_auc')
        
        x_pos = np.arange(len(top_10))
        bars = axes[1,0].bar(x_pos, top_10['roc_auc'], color=self.colors['info'], alpha=0.7)
        axes[1,0].set_xticks(x_pos)
        axes[1,0].set_xticklabels([f"{row['resampling_method'][:4]}\n{row['model_name'][:8]}" 
                                  for _, row in top_10.iterrows()], rotation=45, fontsize=8)
        axes[1,0].set_ylabel('ROC-AUC')
        axes[1,0].set_title('Top 10 Model Performance')
        
        # 4. 튜닝 전후 비교 (있는 경우)
        if tuning_results is not None:
            # 튜닝된 모델들과 원래 모델 비교
            axes[1,1].set_title('Hyperparameter Tuning Effect')
            
            tuning_models = tuning_results['model_name'].unique()
            before_after = []
            
            for model in tuning_models:
                # 원래 성능
                original_perf = modeling_results[modeling_results['model_name'] == model]['roc_auc'].max()
                # 튜닝 후 성능
                tuned_perf = tuning_results[tuning_results['model_name'] == model]['roc_auc'].iloc[0]
                
                before_after.append({
                    'model': model,
                    'before': original_perf,
                    'after': tuned_perf,
                    'improvement': tuned_perf - original_perf
                })
            
            if before_after:
                models = [item['model'] for item in before_after]
                before_values = [item['before'] for item in before_after]
                after_values = [item['after'] for item in before_after]
                
                x = np.arange(len(models))
                width = 0.35
                
                axes[1,1].bar(x - width/2, before_values, width, label='Before Tuning', alpha=0.7)
                axes[1,1].bar(x + width/2, after_values, width, label='After Tuning', alpha=0.7)
                
                axes[1,1].set_ylabel('ROC-AUC')
                axes[1,1].set_xticks(x)
                axes[1,1].set_xticklabels(models, rotation=45)
                axes[1,1].legend()
        else:
            # 튜닝 결과가 없으면 F1-Score 비교
            model_f1 = modeling_results.groupby('model_name')['f1_score'].mean().sort_values(ascending=False)
            
            bars = axes[1,1].bar(range(len(model_f1)), model_f1.values, color=self.colors['secondary'], alpha=0.7)
            axes[1,1].set_xticks(range(len(model_f1)))
            axes[1,1].set_xticklabels(model_f1.index, rotation=45)
            axes[1,1].set_ylabel('F1-Score')
            axes[1,1].set_title('Average F1-Score by Model')
        
        plt.tight_layout()
        plt.savefig(self.output_path / "04_model_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 모델 성능 비교 시각화 저장")
    
    def create_final_dashboard(self):
        """최종 결과 대시보드"""
        print("\n📊 최종 결과 대시보드 생성 중...")
        
        # 모든 주요 결과 로드
        modeling_results = self.load_data_safely(self.dataset_path / "4_modeling/modeling_results.csv", "모델링 결과")
        tuning_results = self.load_data_safely(self.dataset_path / "5_final_models/tuning_results.csv", "튜닝 결과")
        
        # 대시보드 생성
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('MIMIC-IV 48-hour Mortality Prediction - Final Results Dashboard', fontsize=20, fontweight='bold')
        
        # GridSpec을 사용하여 레이아웃 설정
        gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1, 1])
        
        # 1. 프로젝트 개요 (텍스트)
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.axis('off')
        
        project_info = """
🏥 MIMIC-IV ICU 48시간 사망률 예측 프로젝트

📊 데이터: MIMIC-IV 중환자실 데이터
🎯 목표: ICU 입실 후 24시간 데이터로 48시간 사망 예측
⏰ 시간 범위: ICU 입실 후 24시간 내 생체징후/검사수치
🧬 특성: 생체징후, 검사수치, 인구학적 정보, 동반질환

🔬 모델: 6개 (Logistic Regression, SVC, Random Forest, XGBoost, LightGBM, Extra Trees)
⚖️ 리샘플링: SMOTE, Downsampling
🎯 최적화: Optuna 베이지안 최적화
        """
        
        ax1.text(0.05, 0.95, project_info, transform=ax1.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        # 2. 주요 지표 (텍스트)
        ax2 = fig.add_subplot(gs[0, 2:])
        ax2.axis('off')
        
        if modeling_results is not None:
            best_model = modeling_results.loc[modeling_results['roc_auc'].idxmax()]
            total_models = len(modeling_results)
            
            metrics_info = f"""
📈 주요 성과

🏆 최고 성능 모델: {best_model['resampling_method'].upper()}-{best_model['model_name']}
📊 ROC-AUC: {best_model['roc_auc']:.4f}
📏 F1-Score: {best_model['f1_score']:.4f}
🎯 Precision: {best_model['precision']:.4f}
🔍 Recall: {best_model['recall']:.4f}

🤖 총 실험 모델: {total_models}개
✅ 재현 가능성: 100% (시드 고정)
            """
            
            ax2.text(0.05, 0.95, metrics_info, transform=ax2.transAxes, fontsize=11,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        
        # 3. 상위 모델 성능 (막대 그래프)
        if modeling_results is not None:
            ax3 = fig.add_subplot(gs[1, :2])
            top_5 = modeling_results.nlargest(5, 'roc_auc')
            
            bars = ax3.barh(range(len(top_5)), top_5['roc_auc'], color=self.colors['primary'], alpha=0.8)
            ax3.set_yticks(range(len(top_5)))
            ax3.set_yticklabels([f"{row['resampling_method'][:6]}-{row['model_name'][:10]}" 
                               for _, row in top_5.iterrows()])
            ax3.set_xlabel('ROC-AUC')
            ax3.set_title('Top 5 Model Performance', fontweight='bold')
            
            # 값 표시
            for i, (bar, value) in enumerate(zip(bars, top_5['roc_auc'])):
                ax3.text(value + 0.005, bar.get_y() + bar.get_height()/2,
                        f'{value:.3f}', va='center', fontweight='bold')
        
        # 4. 리샘플링 효과 (파이 차트)
        if modeling_results is not None:
            ax4 = fig.add_subplot(gs[1, 2:])
            resampling_perf = modeling_results.groupby('resampling_method')['roc_auc'].mean()
            
            colors = [self.colors['success'], self.colors['warning']]
            wedges, texts, autotexts = ax4.pie(resampling_perf.values, 
                                              labels=resampling_perf.index,
                                              autopct='%1.3f', 
                                              colors=colors,
                                              startangle=90)
            ax4.set_title('Average Performance by Resampling Method', fontweight='bold')
        
        # 5. 최종 결과 요약 테이블
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        if tuning_results is not None and len(tuning_results) > 0:
            # 튜닝 결과 테이블
            table_data = []
            for _, row in tuning_results.iterrows():
                table_data.append([
                    row['model_name'],
                    f"{row['roc_auc']:.4f}",
                    f"{row['f1_score']:.4f}",
                    f"{row['precision']:.4f}",
                    f"{row['recall']:.4f}",
                    f"{row['accuracy']:.4f}"
                ])
            
            table = ax5.table(cellText=table_data,
                            colLabels=['모델', 'ROC-AUC', 'F1-Score', 'Precision', 'Recall', 'Accuracy'],
                            cellLoc='center',
                            loc='center',
                            cellColours=None)
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            # 헤더 스타일링
            for i in range(6):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            ax5.set_title('Final Model Performance (Test Set)', fontweight='bold', pad=20)
        
        elif modeling_results is not None:
            # 모델링 결과만 있는 경우
            table_data = []
            for _, row in modeling_results.head(3).iterrows():
                table_data.append([
                    f"{row['resampling_method']}-{row['model_name']}",
                    f"{row['roc_auc']:.4f}",
                    f"{row['f1_score']:.4f}",
                    f"{row['precision']:.4f}",
                    f"{row['recall']:.4f}",
                    f"{row['accuracy']:.4f}"
                ])
            
            table = ax5.table(cellText=table_data,
                            colLabels=['모델', 'ROC-AUC', 'F1-Score', 'Precision', 'Recall', 'Accuracy'],
                            cellLoc='center',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            ax5.set_title('Top 3 Model Performance (Validation Set)', fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(self.output_path / "05_final_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 최종 결과 대시보드 저장")
    
    def generate_shap_visualizations(self):
        """SHAP 분석 시각화"""
        print("\n📊 SHAP 분석 시각화 생성 중...")
        
        shap_path = self.base_path / "results" / "07_shap_analysis"
        
        if not shap_path.exists():
            print("⚠️ SHAP 분석 결과가 없습니다. 07_shap_analysis.py를 먼저 실행해주세요.")
            return
        
        # SHAP 결과 파일들 찾기
        feature_importance_files = list(shap_path.glob("*_feature_importance.csv"))
        shap_value_files = list(shap_path.glob("*_shap_values.json"))
        
        if not feature_importance_files:
            print("⚠️ SHAP feature importance 파일이 없습니다.")
            return
        
        # Figure 생성
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('MIMIC-IV SHAP Analysis - Model Interpretability', fontsize=20, fontweight='bold')
        
        # 2x2 레이아웃
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1], hspace=0.3, wspace=0.3)
        
        # 1. 모델별 상위 특성 비교
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_model_feature_comparison(ax1, feature_importance_files)
        
        # 2. 전체 상위 특성 (모든 모델 평균)
        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_global_feature_importance(ax2, feature_importance_files)
        
        # 3. 최고 성능 모델의 상위 20개 특성
        ax3 = fig.add_subplot(gs[1, 0])
        self.plot_best_model_features(ax3, feature_importance_files)
        
        # 4. 특성 카테고리별 중요도
        ax4 = fig.add_subplot(gs[1, 1])
        self.plot_feature_categories(ax4, feature_importance_files)
        
        # 저장
        shap_viz_path = self.output_path / "05_shap_analysis.png"
        plt.savefig(shap_viz_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("✅ SHAP 분석 시각화 저장")
        
        # 별도로 beeplot 생성
        self.generate_shap_beeplot()
    
    def plot_model_feature_comparison(self, ax, feature_importance_files):
        """모델별 상위 특성 비교"""
        top_features_by_model = {}
        
        # 각 모델의 상위 5개 특성 추출
        for file_path in feature_importance_files:
            model_name = file_path.stem.replace('_feature_importance', '')
            df = pd.read_csv(file_path)
            top_5 = df.head(5)
            top_features_by_model[model_name] = top_5
        
        # 전체 특성 집합
        all_features = set()
        for model_features in top_features_by_model.values():
            all_features.update(model_features['feature'].tolist())
        
        # 각 특성별로 모델별 중요도 매트릭스 생성
        feature_importance_matrix = []
        feature_names = []
        
        # 상위 빈도 특성들만 선택
        feature_counts = {}
        for model_features in top_features_by_model.values():
            for feature in model_features['feature']:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
        
        # 2개 이상 모델에서 나타나는 특성들만
        common_features = [f for f, c in feature_counts.items() if c >= 2][:10]
        
        for feature in common_features:
            importances = []
            for model_name, model_features in top_features_by_model.items():
                feature_row = model_features[model_features['feature'] == feature]
                if not feature_row.empty:
                    importances.append(feature_row['importance'].values[0])
                else:
                    importances.append(0)
            feature_importance_matrix.append(importances)
            feature_names.append(feature[:20] + '...' if len(feature) > 20 else feature)
        
        if not feature_importance_matrix:
            ax.text(0.5, 0.5, 'No common features found', ha='center', va='center')
            ax.set_title('Model Feature Comparison')
            return
        
        # 히트맵 그리기
        feature_importance_matrix = np.array(feature_importance_matrix)
        model_names = [name.replace('_', '\n')[:15] for name in top_features_by_model.keys()]
        
        im = ax.imshow(feature_importance_matrix, cmap='Reds', aspect='auto')
        
        # 축 설정
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names, fontsize=8)
        ax.set_title('Feature Importance by Model', fontweight='bold')
        
        # 컬러바
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    def plot_global_feature_importance(self, ax, feature_importance_files):
        """전체 상위 특성 (모든 모델 평균)"""
        all_features = {}
        
        # 모든 모델의 특성 중요도 수집
        for file_path in feature_importance_files:
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                feature = row['feature']
                importance = row['importance']
                if feature not in all_features:
                    all_features[feature] = []
                all_features[feature].append(importance)
        
        # 평균 중요도 계산
        avg_importance = {feature: np.mean(importances) 
                         for feature, importances in all_features.items()}
        
        # 상위 15개 선택
        top_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:15]
        
        features, importances = zip(*top_features)
        
        # 특성명 단축
        short_features = [f[:25] + '...' if len(f) > 25 else f for f in features]
        
        # 수평 막대 그래프
        bars = ax.barh(range(len(short_features)), importances, color=self.colors['primary'], alpha=0.7)
        
        ax.set_yticks(range(len(short_features)))
        ax.set_yticklabels(short_features, fontsize=9)
        ax.set_xlabel('Average SHAP Importance')
        ax.set_title('Top Features Across All Models', fontweight='bold')
        
        # 값 표시
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center', fontsize=8)
        
        ax.invert_yaxis()
    
    def plot_best_model_features(self, ax, feature_importance_files):
        """최고 성능 모델의 상위 특성"""
        # 최고 성능 모델 찾기 (파일명에서 ROC-AUC 정보 없으므로 첫 번째 파일 사용)
        best_model_file = feature_importance_files[0]  # 이미 성능순으로 정렬되어 있다고 가정
        
        df = pd.read_csv(best_model_file)
        top_20 = df.head(20)
        
        model_name = best_model_file.stem.replace('_feature_importance', '')
        
        # 특성명 단축
        short_features = [f[:30] + '...' if len(f) > 30 else f for f in top_20['feature']]
        
        bars = ax.barh(range(len(short_features)), top_20['importance'], 
                      color=self.colors['success'], alpha=0.7)
        
        ax.set_yticks(range(len(short_features)))
        ax.set_yticklabels(short_features, fontsize=8)
        ax.set_xlabel('SHAP Importance')
        ax.set_title(f'Top 20 Features - Best Model\n({model_name.replace("_", " ")})', fontweight='bold')
        
        # 값 표시 (상위 10개만)
        for i, bar in enumerate(bars[:10]):
            width = bar.get_width()
            ax.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center', fontsize=8)
        
        ax.invert_yaxis()
    
    def plot_feature_categories(self, ax, feature_importance_files):
        """특성 카테고리별 중요도"""
        # 특성 카테고리 정의
        categories = {
            'vital_signs': ['hr_', 'sbp_', 'dbp_', 'mbp_', 'temp_', 'resp_', 'spo2_'],
            'lab_results': ['glucose_', 'creatinine_', 'bun_', 'sodium_', 'potassium_', 
                           'hemoglobin_', 'hematocrit_', 'platelet_', 'wbc_'],
            'demographics': ['anchor_age', 'gender'],
            'comorbidities': ['diabetes', 'hypertension', 'obesity'],
            'icu_info': ['first_careunit', 'los'],
            'sleep_sedation': ['sleep_', 'sedation_']
        }
        
        # 모든 모델의 특성 중요도 수집
        all_importances = {}
        for file_path in feature_importance_files:
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                feature = row['feature']
                importance = row['importance']
                if feature not in all_importances:
                    all_importances[feature] = []
                all_importances[feature].append(importance)
        
        # 카테고리별 평균 중요도 계산
        category_importance = {}
        for category, keywords in categories.items():
            total_importance = 0
            count = 0
            for feature, importances in all_importances.items():
                if any(keyword in feature.lower() for keyword in keywords):
                    total_importance += np.mean(importances)
                    count += 1
            
            if count > 0:
                category_importance[category] = total_importance / count
            else:
                category_importance[category] = 0
        
        # 그래프 그리기
        categories = list(category_importance.keys())
        importances = list(category_importance.values())
        
        bars = ax.bar(categories, importances, color=[
            self.colors['primary'], self.colors['secondary'], self.colors['success'],
            self.colors['danger'], self.colors['warning'], self.colors['info']
        ][:len(categories)], alpha=0.7)
        
        ax.set_ylabel('Average SHAP Importance')
        ax.set_title('Feature Importance by Category', fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        
        # 값 표시
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    def generate_shap_beeplot(self):
        """SHAP Beeplot (SHAP 값 분포 시각화) - F1 최고 & AUROC 최고 모델 2개"""
        print("\n📊 SHAP Beeplot 생성 중...")
        
        # 모델링 결과에서 최고 성능 모델 찾기
        modeling_results_path = self.dataset_path / "4_modeling/modeling_results.csv"
        if not modeling_results_path.exists():
            print("⚠️ 모델링 결과 파일이 없습니다.")
            return
        
        results_df = pd.read_csv(modeling_results_path)
        
        # F1 Score 최고 모델
        best_f1_row = results_df.loc[results_df['f1_score'].idxmax()]
        best_f1_model = f"{best_f1_row['resampling_method']}_{best_f1_row['model_name']}"
        
        # ROC-AUC 최고 모델 
        best_auc_row = results_df.loc[results_df['roc_auc'].idxmax()]
        best_auc_model = f"{best_auc_row['resampling_method']}_{best_auc_row['model_name']}"
        
        print(f"🏆 F1 Score 최고 모델: {best_f1_model} (F1: {best_f1_row['f1_score']:.4f})")
        print(f"🏆 ROC-AUC 최고 모델: {best_auc_model} (AUC: {best_auc_row['roc_auc']:.4f})")
        
        # 대상 모델들
        target_models = [best_f1_model, best_auc_model]
        if best_f1_model == best_auc_model:
            target_models = [best_f1_model]  # 같은 모델이면 하나만
            print("💡 F1과 AUC 최고 모델이 동일합니다.")
        
        shap_path = self.base_path / "results" / "07_shap_analysis"
        
        if not shap_path.exists():
            print("⚠️ SHAP 분석 결과가 없습니다.")
            return
        
        # 각 모델에 대해 beeplot 생성
        for i, model_name in enumerate(target_models):
            try:
                self.generate_single_beeplot(shap_path, model_name, i, len(target_models))
            except Exception as e:
                print(f"❌ {model_name} beeplot 생성 실패: {e}")
                
        print("✅ 모든 SHAP Beeplot 저장 완료")
    
    def generate_single_beeplot(self, shap_path, model_name, index, total_models):
        """단일 모델에 대한 beeplot 생성"""
        
        # 파일 찾기
        shap_file = shap_path / f"{model_name}_shap_values.json"
        importance_file = shap_path / f"{model_name}_feature_importance.csv"
        
        if not shap_file.exists():
            print(f"⚠️ {model_name} SHAP 값 파일이 없습니다: {shap_file}")
            return
            
        if not importance_file.exists():
            print(f"⚠️ {model_name} 특성 중요도 파일이 없습니다: {importance_file}")
            return
        
        print(f"\n=== {model_name} SHAP Beeplot 생성 ===")
        print(f"SHAP 파일: {shap_file.name}")
        print(f"특성 중요도 파일: {importance_file.name}")
        
        # SHAP 값 로드
        with open(shap_file, 'r') as f:
            shap_data = json.load(f)
        
        # 특성 중요도 로드 (상위 15개 특성)
        importance_df = pd.read_csv(importance_file)
        top_features = importance_df.head(15)['feature'].tolist()
        
        print(f"상위 15개 특성: {top_features}")
        lactate_features = [f for f in top_features if 'lactate' in f.lower()]
        print(f"젖산 관련 특성: {lactate_features}")

        # Figure 생성
        fig, axes = plt.subplots(1, 2, figsize=(20, 12))
        fig.suptitle(f'SHAP Beeplot Analysis - {model_name.replace("_", " ").title()}', 
                    fontsize=16, fontweight='bold')
        
        # 1. SHAP 값 분포 (Beeswarm-style plot)
        self.plot_shap_beeswarm(axes[0], shap_data, top_features, "SHAP Value Distribution")
        
        # 2. 특성별 SHAP 값 요약 통계
        self.plot_shap_summary_stats(axes[1], shap_data, top_features, "SHAP Value Statistics")
        
        plt.tight_layout()
        
        # 저장 (여러 모델인 경우 번호 추가)
        if total_models > 1:
            beeplot_path = self.output_path / f"06_shap_beeplot_{index+1}_{model_name}.png"
        else:
            beeplot_path = self.output_path / "06_shap_beeplot.png"
            
        plt.savefig(beeplot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ {model_name} Beeplot 저장: {beeplot_path.name}")
    
    def plot_shap_beeswarm(self, ax, shap_data, features, title):
        """SHAP 값 beeswarm plot"""
        try:
            # SHAP 값과 특성 값 추출
            shap_values_raw = shap_data.get('shap_values', [])
            feature_values_raw = shap_data.get('feature_values', [])
            feature_names = shap_data.get('feature_names', [])
            
            print(f"SHAP values type: {type(shap_values_raw)}")
            print(f"Feature values type: {type(feature_values_raw)}")
            print(f"Feature names length: {len(feature_names)}")
            
            if not shap_values_raw or not feature_names:
                ax.text(0.5, 0.5, 'No SHAP values available', ha='center', va='center')
                ax.set_title(title)
                return
            
            # SHAP 값을 numpy 배열로 변환하되 차원 확인
            shap_values = np.array(shap_values_raw)
            feature_values = np.array(feature_values_raw) if feature_values_raw else None
            
            print(f"SHAP values shape: {shap_values.shape}")
            if feature_values is not None:
                print(f"Feature values shape: {feature_values.shape}")
            
            # 1차원인 경우 2차원으로 변환
            if shap_values.ndim == 1:
                print("SHAP values is 1D, reshaping...")
                # 특성 수만큼 분할
                n_features = len(feature_names)
                n_samples = len(shap_values) // n_features
                if len(shap_values) % n_features == 0:
                    shap_values = shap_values.reshape(n_samples, n_features)
                else:
                    ax.text(0.5, 0.5, 'SHAP values dimension mismatch', ha='center', va='center')
                    ax.set_title(title)
                    return
            
            # feature_values도 동일하게 처리
            if feature_values is not None and feature_values.ndim == 1:
                print("Feature values is 1D, reshaping...")
                n_features = len(feature_names)
                n_samples = len(feature_values) // n_features
                if len(feature_values) % n_features == 0:
                    feature_values = feature_values.reshape(n_samples, n_features)
                else:
                    feature_values = None
            
            print(f"Final SHAP values shape: {shap_values.shape}")
            
            # 상위 특성들의 인덱스 찾기
            feature_indices = []
            available_features = []
            lactate_found = []
            
            print(f"\n=== Feature Matching 디버깅 ===")
            print(f"요청된 features (상위 15개): {features}")
            print(f"SHAP feature_names: {feature_names}")
            
            for feature in features:
                if feature in feature_names:
                    idx = feature_names.index(feature)
                    if idx < shap_values.shape[1]:  # 범위 확인
                        feature_indices.append(idx)
                        available_features.append(feature)
                        if 'lactate' in feature.lower():
                            lactate_found.append(feature)
                        print(f"  ✅ {feature} -> index {idx}")
                    else:
                        print(f"  ❌ {feature} -> index {idx} (out of range)")
                else:
                    print(f"  ❌ {feature} -> not found in feature_names")
            
            print(f"발견된 젖산 특성들: {lactate_found}")
            
            if not feature_indices:
                ax.text(0.5, 0.5, 'No matching features found', ha='center', va='center')
                ax.set_title(title)
                return
            
            print(f"사용 가능한 features: {len(available_features)}")
            
            # 각 특성별로 SHAP 값 플롯
            y_positions = []
            colors = plt.cm.RdYlBu_r  # 빨강-노랑-파랑 색상맵
            
            for i, feature_idx in enumerate(feature_indices):
                try:
                    feature_name = feature_names[feature_idx]
                    shap_vals = shap_values[:, feature_idx]
                    
                    # 특성 값 처리
                    if feature_values is not None and feature_idx < feature_values.shape[1]:
                        feat_vals = feature_values[:, feature_idx]
                    else:
                        # 특성 값이 없으면 SHAP 값 자체를 사용
                        feat_vals = shap_vals
                    
                    # 샘플링 (너무 많은 점이 있을 경우)
                    if len(shap_vals) > 1000:
                        sample_idx = np.random.choice(len(shap_vals), 1000, replace=False)
                        shap_vals = shap_vals[sample_idx]
                        feat_vals = feat_vals[sample_idx]
                    
                    # 특성 값에 따른 색상 정규화
                    if np.std(feat_vals) > 0:
                        norm_feat_vals = (feat_vals - np.min(feat_vals)) / (np.max(feat_vals) - np.min(feat_vals))
                    else:
                        norm_feat_vals = np.zeros_like(feat_vals)
                    
                    # Y 위치에 약간의 jitter 추가
                    y_pos = np.full_like(shap_vals, i) + np.random.normal(0, 0.1, len(shap_vals))
                    y_positions.append(i)
                    
                    # 산점도 그리기
                    scatter = ax.scatter(shap_vals, y_pos, c=norm_feat_vals, cmap=colors, 
                                       alpha=0.6, s=20, edgecolors='none')
                    
                except Exception as e:
                    print(f"Error processing feature {feature_idx}: {e}")
                    continue
            
            if not y_positions:
                ax.text(0.5, 0.5, 'No features could be plotted', ha='center', va='center')
                ax.set_title(title)
                return
            
            # 축 설정
            ax.set_yticks(y_positions)
            ax.set_yticklabels([available_features[i][:30] + '...' if len(available_features[i]) > 30 else available_features[i] 
                               for i in range(len(y_positions))], fontsize=10)
            ax.set_xlabel('SHAP Value (Impact on Model Output)')
            ax.set_title(title, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3)
            
            # 컬러바 추가 (scatter가 정의된 경우에만)
            try:
                cbar = plt.colorbar(scatter, ax=ax, fraction=0.02, pad=0.01)
                cbar.set_label('Feature Value\n(Low → High)', rotation=270, labelpad=20)
            except:
                pass
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error creating beeswarm plot: {str(e)}', ha='center', va='center')
            ax.set_title(title)
            print(f"Beeswarm plot error: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_shap_summary_stats(self, ax, shap_data, features, title):
        """SHAP 값 요약 통계"""
        try:
            shap_values_raw = shap_data.get('shap_values', [])
            feature_names = shap_data.get('feature_names', [])
            
            if not shap_values_raw or not feature_names:
                ax.text(0.5, 0.5, 'No SHAP values available', ha='center', va='center')
                ax.set_title(title)
                return
            
            # SHAP 값을 numpy 배열로 변환하되 차원 확인
            shap_values = np.array(shap_values_raw)
            
            # 1차원인 경우 2차원으로 변환
            if shap_values.ndim == 1:
                n_features = len(feature_names)
                n_samples = len(shap_values) // n_features
                if len(shap_values) % n_features == 0:
                    shap_values = shap_values.reshape(n_samples, n_features)
                else:
                    ax.text(0.5, 0.5, 'SHAP values dimension mismatch', ha='center', va='center')
                    ax.set_title(title)
                    return
            
            # 특성별 통계 계산
            stats_data = []
            for feature in features:
                if feature in feature_names:
                    feature_idx = feature_names.index(feature)
                    if feature_idx < shap_values.shape[1]:  # 범위 확인
                        try:
                            shap_vals = shap_values[:, feature_idx]
                            
                            stats = {
                                'feature': feature[:25] + '...' if len(feature) > 25 else feature,
                                'mean_abs': np.mean(np.abs(shap_vals)),
                                'std': np.std(shap_vals),
                                'min': np.min(shap_vals),
                                'max': np.max(shap_vals)
                            }
                            stats_data.append(stats)
                        except Exception as e:
                            print(f"Error processing feature {feature}: {e}")
                            continue
            
            if not stats_data:
                ax.text(0.5, 0.5, 'No statistics available', ha='center', va='center')
                ax.set_title(title)
                return
            
            # 평균 절댓값으로 정렬
            stats_data.sort(key=lambda x: x['mean_abs'], reverse=True)
            
            # 박스플롯 스타일로 시각화
            y_positions = range(len(stats_data))
            feature_names_short = [item['feature'] for item in stats_data]
            
            # 평균 절댓값 막대그래프
            mean_abs_vals = [item['mean_abs'] for item in stats_data]
            bars = ax.barh(y_positions, mean_abs_vals, color=self.colors['primary'], alpha=0.7)
            
            # 최솟값, 최댓값 범위 표시
            for i, item in enumerate(stats_data):
                ax.plot([item['min'], item['max']], [i, i], 'k-', alpha=0.5, linewidth=2)
                ax.plot([item['min']], [i], 'k|', markersize=8, alpha=0.7)
                ax.plot([item['max']], [i], 'k|', markersize=8, alpha=0.7)
            
            ax.set_yticks(y_positions)
            ax.set_yticklabels(feature_names_short, fontsize=10)
            ax.set_xlabel('Mean |SHAP Value|')
            ax.set_title(title, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3, axis='x')
            
            # 값 표시
            for i, (bar, val) in enumerate(zip(bars, mean_abs_vals)):
                ax.text(val + val*0.01, bar.get_y() + bar.get_height()/2, 
                       f'{val:.3f}', ha='left', va='center', fontsize=8)
            
            ax.invert_yaxis()
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error creating summary stats: {str(e)}', ha='center', va='center')
            ax.set_title(title)
            print(f"Summary stats error: {e}")
            import traceback
            traceback.print_exc()
    
    def generate_all_figures(self):
        """모든 시각화 생성"""
        print("🎨 MIMIC-IV 48시간 사망률 예측 - 전체 시각화 생성 시작")
        print("=" * 70)
        
        # 출력 폴더 정보
        print(f"📁 시각화 저장 위치: {self.output_path}")
        
        # 각 시각화 생성
        self.create_data_distribution_plots()
        self.create_missing_data_heatmap()
        self.create_resampling_comparison()
        self.create_model_performance_plots()
        self.create_final_dashboard()
        self.generate_shap_visualizations()
        
        print("\n" + "=" * 70)
        print("🎉 모든 시각화 생성 완료!")
        print(f"📁 저장된 파일:")
        
        # 생성된 파일 목록 (동적으로 확인)
        expected_files = [
            "01_data_distribution.png",
            "02_missing_data_analysis.png",
            "02_missing_data_impact.png", 
            "03_resampling_comparison.png",
            "03_data_pipeline.png",
            "04_model_performance.png",
            "05_final_dashboard.png",
            "05_shap_analysis.png"
        ]
        
        # SHAP beeplot 파일들 동적 추가
        beeplot_files = list(self.output_path.glob("06_shap_beeplot*.png"))
        for beeplot_file in beeplot_files:
            expected_files.append(beeplot_file.name)
        
        for i, filename in enumerate(expected_files, 1):
            file_path = self.output_path / filename
            if file_path.exists():
                print(f"  ✅ {i}. {filename}")
            else:
                print(f"  ❌ {i}. {filename} (생성되지 않음)")
        
        print(f"\n📊 총 {len(expected_files)}개 시각화 파일이 저장되었습니다.")
        print("=" * 70)

def main():
    """메인 실행 함수"""
    generator = FigureGenerator()
    generator.generate_all_figures()

if __name__ == "__main__":
    main()
