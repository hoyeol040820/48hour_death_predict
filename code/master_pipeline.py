#!/usr/bin/env python3
"""
MIMIC-IV 48시간 사망률 예측 - 마스터 파이프라인
전체 분석 과정을 순서대로 실행하는 마스터 스크립트

실행 순서:
1. 01_data_extraction.py (별도 실행 - 파생변수 생성 포함)
2. 02_data_cleaning.py
3. 03_data_splitting.py
4. 04_resampling.py
5. 05_modeling_evaluation.py
6. 06_hyperparameter_tuning.py
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
import pandas as pd

class MIMICPipeline:
    def __init__(self):
        self.code_path = Path(__file__).parent
        self.scripts = [
            {
                'file': '02_data_cleaning.py',
                'name': '데이터 정제',
                'description': '24시간 내 사망자 제거, 이상치/결측치 처리'
            },
            {
                'file': '03_data_splitting.py',
                'name': '데이터 분할',
                'description': 'Train/Validation/Test 분할 (6:2:2)'
            },
            {
                'file': '04_resampling.py',
                'name': '리샘플링',
                'description': 'SMOTE 및 Downsampling 적용'
            },
            {
                'file': '05_modeling_evaluation.py',
                'name': '모델링 및 평가',
                'description': '6개 모델 학습 및 성능 평가'
            },
            {
                'file': '06_hyperparameter_tuning.py',
                'name': '하이퍼파라미터 튜닝',
                'description': '상위 모델들의 최적화 및 최종 평가'
            }
        ]
        
        self.start_time = datetime.now()
        self.results = []
    
    def check_prerequisites(self):
        """사전 요구사항 확인"""
        print("🔍 사전 요구사항 확인 중...")
        
        # 프로젝트 루트 디렉토리 찾기
        project_root = Path(__file__).parent.parent
        
        # 01_data_extraction.py 실행 확인
        raw_data_path = project_root / "dataset" / "0_raw" / "mimic_mortality_raw.csv"
        
        if not raw_data_path.exists():
            print("❌ 데이터 추출이 완료되지 않았습니다.")
            print("   먼저 01_data_extraction.py를 실행해주세요:")
            print(f"   python {self.code_path}/01_data_extraction.py")
            return False
        
        print("✅ 데이터 추출 완료 확인")
        
        # 필요한 라이브러리 확인
        required_packages = [
            'pandas', 'numpy', 'sklearn', 'xgboost', 'lightgbm', 
            'imblearn', 'optuna', 'joblib'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ 필요한 패키지가 설치되지 않았습니다: {missing_packages}")
            print("   다음 명령으로 설치해주세요:")
            print(f"   pip install {' '.join(missing_packages)}")
            return False
        
        print("✅ 필요한 패키지 설치 확인")
        return True
    
    def run_script(self, script_info):
        """개별 스크립트 실행"""
        script_file = script_info['file']
        script_name = script_info['name']
        description = script_info['description']
        
        print(f"\n{'='*80}")
        print(f"🚀 {script_name} 시작")
        print(f"📄 파일: {script_file}")
        print(f"📝 설명: {description}")
        print(f"⏰ 시작: {datetime.now().strftime('%H:%M:%S')}")
        print('='*80)
        
        script_path = self.code_path / script_file
        start_time = time.time()
        
        try:
            # 스크립트 실행
            result = subprocess.run(
                [sys.executable, str(script_path)],
                check=True,
                capture_output=True,
                text=True,
                timeout=3600  # 1시간 타임아웃
            )
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            print(f"\n✅ {script_name} 성공 완료!")
            print(f"⏱️  소요시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")
            print(f"⏰ 완료: {datetime.now().strftime('%H:%M:%S')}")
            
            # 결과 로그
            if result.stdout:
                print(f"\n📝 실행 로그:")
                print(result.stdout[-1000:])  # 마지막 1000자만 표시
            
            self.results.append({
                'script': script_file,
                'name': script_name,
                'status': 'SUCCESS',
                'elapsed_time': elapsed,
                'error': None
            })
            
            return True
            
        except subprocess.CalledProcessError as e:
            end_time = time.time()
            elapsed = end_time - start_time
            
            print(f"\n❌ {script_name} 실행 실패!")
            print(f"⏱️  소요시간: {elapsed:.1f}초")
            print(f"🚨 에러코드: {e.returncode}")
            
            if e.stdout:
                print(f"\n📝 표준출력:")
                print(e.stdout[-500:])
            
            if e.stderr:
                print(f"\n🚨 에러메시지:")
                print(e.stderr[-500:])
            
            self.results.append({
                'script': script_file,
                'name': script_name,
                'status': 'FAILED',
                'elapsed_time': elapsed,
                'error': str(e)
            })
            
            return False
            
        except subprocess.TimeoutExpired:
            print(f"\n⏰ {script_name} 타임아웃 (1시간 초과)")
            
            self.results.append({
                'script': script_file,
                'name': script_name,
                'status': 'TIMEOUT',
                'elapsed_time': 3600,
                'error': 'Timeout (1시간 초과)'
            })
            
            return False
    
    def generate_final_summary(self):
        """최종 요약 보고서 생성"""
        print(f"\n{'='*80}")
        print("📊 파이프라인 실행 완료 - 최종 요약")
        print('='*80)
        
        total_time = (datetime.now() - self.start_time).total_seconds()
        success_count = sum(1 for r in self.results if r['status'] == 'SUCCESS')
        
        print(f"🏁 전체 실행 시간: {total_time/60:.1f}분")
        print(f"✅ 성공한 단계: {success_count}/{len(self.results)}")
        print(f"🏆 완료율: {success_count/len(self.results)*100:.1f}%")
        
        print(f"\n📋 단계별 결과:")
        for result in self.results:
            status_emoji = "✅" if result['status'] == 'SUCCESS' else "❌" if result['status'] == 'FAILED' else "⏰"
            # 스크립트 파일명에서 숫자 추출
            script_num = int(result['script'].split('_')[0])
            print(f"  {script_num}. {result['name']}: {status_emoji} {result['status']} ({result['elapsed_time']:.1f}초)")
            if result['error']:
                print(f"     오류: {result['error'][:100]}...")
        
        # 최종 결과 파일 위치 안내
        if success_count == len(self.results):
            print(f"\n🎉 모든 단계가 성공적으로 완료되었습니다!")
            print(f"\n📁 주요 결과 파일 위치:")
            project_root = Path(__file__).parent.parent
            print(f"   • 최종 모델: {project_root}/dataset/5_final_models/")
            print(f"   • 모델링 결과: {project_root}/dataset/4_modeling/")
            print(f"   • 시각화 생성: python figure_generator.py")
        else:
            print(f"\n⚠️  일부 단계가 실패했습니다. 로그를 확인하여 문제를 해결해주세요.")
        
        # 요약 파일 저장
        project_root = Path(__file__).parent.parent
        summary_file = project_root / "pipeline_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("MIMIC-IV 48시간 사망률 예측 파이프라인 실행 요약\n")
            f.write("=" * 60 + "\n")
            f.write(f"실행 일시: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"완료 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"총 소요시간: {total_time/60:.1f}분\n\n")
            
            for result in self.results:
                # 스크립트 파일명에서 숫자 추출
                script_num = int(result['script'].split('_')[0])
                f.write(f"{script_num}. {result['name']}: {result['status']} ({result['elapsed_time']:.1f}초)\n")
                if result['error']:
                    f.write(f"   오류: {result['error']}\n")
            
            f.write(f"\n성공률: {success_count}/{len(self.results)} ({success_count/len(self.results)*100:.1f}%)\n")
        
        print(f"\n💾 실행 요약 저장: {summary_file}")
    
    def run_pipeline(self):
        """전체 파이프라인 실행"""
        print("🏥 MIMIC-IV 48시간 사망률 예측 파이프라인 시작")
        print(f"📅 시작 시간: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 총 {len(self.scripts)}단계 실행 예정")
        
        # 사전 요구사항 확인
        if not self.check_prerequisites():
            print("\n❌ 사전 요구사항을 만족하지 않아 파이프라인을 중단합니다.")
            return
        
        print(f"\n{'='*80}")
        print("📝 실행 예정 단계:")
        print('='*80)
        for i, script in enumerate(self.scripts):
            # 스크립트 파일명에서 숫자 추출
            script_num = int(script['file'].split('_')[0])
            print(f"{script_num}. {script['name']} ({script['file']})")
            print(f"   {script['description']}")
        
        input(f"\n계속 진행하려면 Enter를 누르세요... (Ctrl+C로 취소)")
        
        # 각 스크립트 순차 실행
        for script_info in self.scripts:
            success = self.run_script(script_info)
            
            if not success:
                user_input = input(f"\n❓ {script_info['name']}이 실패했습니다. 계속 진행하시겠습니까? (y/n): ").lower().strip()
                if user_input not in ['y', 'yes']:
                    print("❌ 사용자 요청으로 파이프라인을 중단합니다.")
                    break
        
        # 최종 요약
        self.generate_final_summary()

def main():
    """메인 실행 함수"""
    pipeline = MIMICPipeline()
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()
