# test_pipeline.py (최종 제출용)

import json
import os
import torch
from ultralytics import YOLO
from transformers import AutoTokenizer, AutoModelForCausalLM

# 이전에 작성한 메인 파이프라인 파일에서 필요한 함수와 클래스를 가져옵니다.
# (파일 이름이 'Final_pipe_Gemma2.py'라고 가정)
from Final_pipe_Gemma2 import GrowthRegressor, analyze_and_advise

def main():
    # --- 1. 경로 설정 ---
    # 테스트할 이미지와 JSON 파일이 있는 고정된 폴더
    test_sample_dir = "Final_Version_final/data/test_samples"
    
    # 결과를 저장할 폴더
    visual_output_dir = "Final_Version_final/data/test_visual_outputs"
    result_output_dir = "Final_Version_final/data/result_for_test"
    
    # 결과 저장 폴더 생성 (없으면)
    os.makedirs(visual_output_dir, exist_ok=True)
    os.makedirs(result_output_dir, exist_ok=True)
    
    # --- 2. 모델 로드 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    print("Loading all models...")
    
    yolo_model_path = 'Final_Version/Yolov11_nano/best.pt'
    regressor_model_path = 'Final_Version/Mobilenetv3_small/best_regressor_model_V3_plantHeight.pth'
    llm_model_name = "google/gemma-2-2b-it"

    try:
        yolo_model = YOLO(yolo_model_path)
        regressor_model = GrowthRegressor(num_output_features=1)
        regressor_model.load_state_dict(torch.load(regressor_model_path, map_location=device))
        regressor_model.to(device)
        regressor_model.eval()
        llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("All models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # --- 3. 테스트 폴더의 모든 이미지에 대해 파이프라인 실행 ---
    # 테스트 폴더에 파일이 있는지 확인
    if not os.path.exists(test_sample_dir) or not os.listdir(test_sample_dir):
        print(f"Error: Test directory '{test_sample_dir}' is empty or does not exist.")
        print("Please place test images and their corresponding .json files in this directory.")
        return
        
    test_files = [f for f in os.listdir(test_sample_dir) if f.endswith(('.png', '.jpg'))]
    print(f"\nFound {len(test_files)} images to test in '{test_sample_dir}'. Starting analysis...")

    for test_image_filename in test_files:
        base_name = os.path.splitext(test_image_filename)[0]
        
        image_path = os.path.join(test_sample_dir, test_image_filename)
        json_path = os.path.join(test_sample_dir, base_name + ".json")
        
        # 각 결과 파일의 경로 설정
        visual_path = os.path.join(visual_output_dir, f"visual_{base_name}.png")
        result_path = os.path.join(result_output_dir, f"{base_name}_result.txt")
        
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            image_date = json_data.get("file_attributes", {}).get("date", "알 수 없음")
            
            # 메인 파이프라인 함수 호출
            analyze_and_advise(
                image_path=image_path,
                image_date=image_date,
                yolo_model=yolo_model,
                regressor_model=regressor_model,
                llm_tokenizer=llm_tokenizer,
                llm_model=llm_model,
                device=device,
                visual_output_path=visual_path,
                result_txt_path=result_path
            )
        else:
            print(f"Warning: JSON file for {test_image_filename} not found. Skipping analysis.")

# --- 스크립트 실행 ---
if __name__ == '__main__':
    main()