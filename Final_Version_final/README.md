# 개인 농업인을 위한 온디바이스 토마토 생육 분석 AI 시스템

# 1. 프로젝트 개요 (Overview)
   
본 프로젝트는 개인 농업인이 네트워크 연결 없이 모바일 기기 환경에서 토마토의 생육 상태를 실시간으로 분석하고, 전문가 수준의 맞춤형 재배 조언을 얻을 수 있는 온디바이스 AI 시스템을 개발하는 것을 목표로 합니다.

이 시스템은 다음과 같은 3단계 하이브리드 파이프라인으로 구성됩니다:

1. 객체 탐지 (Object Detection): YOLOv11n 모델을 사용하여 이미지 내 토마토의 주요 부위(잎, 꽃, 과실 등)를 탐지합니다.

2. 생육 지표 예측 (Growth Indicator Regression): 탐지된 특정 객체의 이미지를 MobileNetV3-Small 기반의 회귀 모델에 입력하여, 식물의 키(plantHeight)와 같은 정량적 생육 지표를 예측합니다.

3. 종합 진단 및 조언 생성 (Diagnosis & Advising): 위 분석 결과(탐지된 객체, 예측된 키, 촬영 날짜)를 구조화된 텍스트로 만들고, 방대한 토마토 재배 지식이 담긴 프롬프트와 함께 Gemma-2B-IT 언어 모델에 전달하여 최종적으로 사용자 맞춤형 진단 및 관리 조언을 생성합니다.




# 2. 프로젝트 파일 구성
제출된 파일은 다음과 같이 구성되어 있습니다:

1. tomato.ipynb: 데이터셋 탐색, 모델 학습 및 실험을 진행한 주피터 노트북 파일입니다.

2. GrowthRegressor.py: MobileNetV3-Small 기반의 생육 지표 회귀 모델(GrowthRegressor)을 학습시키는 Python 스크립트입니다.

3. Final_pipe_Gemma2.py: 전체 파이프라인(YOLO 탐지, 회귀 모델 예측, Gemma 조언 생성)의 핵심 로직이 구현된 Python 스크립트입니다. Test_pipeline.py에서 이 파일을 모듈로 가져와 사용합니다.

4. Test_set_pipe.py: 검증(Validation) 데이터셋에서 임의의 샘플을 추출하여 전체 파이프라인의 성능을 테스트하고, 시각화 결과 및 텍스트 조언을 파일로 저장하는 실행 스크립트입니다.

5. test_pipeline.py: 위의 Test_set_pipe.py의 test셋 생성 기능을 제거하고 현재 폴더 내부의 Final_Version_final\data\test_samples에 있는 이미지, json파일로 전체 파이프라인의 성능을 테스트하고, 시각화 결과 및 텍스트 조언을 파일로 저장하는 메인 실행 스크립드입니다.

6. Final_Version_final/ (폴더):
   
7. Yolov11_nano/best.pt: 학습된 YOLOv11n 모델 가중치 파일이 위치해야 합니다.

8. Mobilenetv3_small/best_regressor_model_V3_plantHeight.pth: 학습된 GrowthRegressor 모델 가중치 파일이 위치해야 합니다.




# 3. 설치 및 환경 설정 (Setup & Installation)
본 프로젝트는 conda 가상 환경에서 개발되었습니다. 아래 절차에 따라 필요한 라이브러리를 설치해주시기 바랍니다.


Conda 가상 환경 생성 및 활성화:

## 'tomato_project'라는 이름의 새로운 Python 3.10 환경 생성
conda create -n tomato_project python=3.10

## 가상 환경 활성화
conda activate tomato_project
필수 라이브러리 설치:


## PyTorch (CUDA 12.1용) 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

## YOLO 및 기타 분석 도구 설치
pip install ultralytics scikit-learn pandas tqdm

## Hugging Face Transformers (Gemma용) 및 기타 LLM 관련 라이브러리 설치

pip install transformers accelerate bitsandbytes sentencepiece

Hugging Face Hub 로그인:
Gemma 모델을 다운로드하려면 Hugging Face 계정 로그인이 필요합니다. 터미널에서 아래 명령어를 실행하고, 웹사이트에서 발급받은 Access Token을 붙여넣어 주세요.


python -c "from huggingface_hub import login; login()"




# 4. 실행 방법 (How to Run)
프로젝트의 전체 파이프라인을 테스트하고 결과를 확인하려면 test_pipeline.py 스크립트를 실행합니다.
*Test_set_pipe.py는 전체 랜덤한 test셋을 생성하는 기능이 있어서 해당 기능을 제거

1. 모델 가중치 준비:
   Final_Version_final/Yolov11_nano/ 폴더에 학습된 YOLO 모델 파일(best.pt)을 위치시킵니다.
   Final_Version_final/Mobilenetv3_small/ 폴더에 학습된 회귀 모델 파일(best_regressor_model_V3_plantHeight.pth)을 위치시킵니다.

2. 테스트 스크립트 실행:
   터미널에서 (tomato_project) 가상 환경이 활성화된 상태인지 확인합니다.
   아래 명령어를 실행합니다.
   
   python Test_pipeline.py

3. 결과 확인:
   스크립트 실행이 완료되면, 터미널에 각 테스트 이미지에 대한 분석 결과와 AI 조언이 출력됩니다.
   
   Final_Version_final/data/test_samples 폴더에 테스트에 사용된 임의의 이미지와 JSON 파일이 생성됩니다.
   
   Final_Version_final/data/test_visual_outputs 폴더에 YOLO가 객체를 탐지한 바운딩 박스가 그려진 시각화 이미지가 저장됩니다.
   
   Final_Version_final/data/result_for_test 폴더에 각 테스트 이미지별 최종 조언이 담긴 .txt 파일이 개별적으로 저장됩니다.




# 5. 데이터셋

1. 데이터 출처: AI Hub - 지능형 스마트팜 통합 데이터(토마토)

2. 데이터 준비:
원본 데이터셋을 다운로드합니다.

3. GrowthRegressor.py 스크립트 내의 경로 변수를 실제 데이터셋 경로에 맞게 수정하여 회귀 모델을 학습시킵니다.

4. Test_pipeline.py 스크립트 내의 검증 데이터 경로(val_image_base_dir, val_json_base_dir)를 실제 경로에 맞게 수정합니다.

5. 기본적으로 테스트용 샘플 데이터가 data/test_samples에 포함되어 있습니다. (10쌍)