# test_pipeline.py
# test셋을 만드는 기능이 있어 최종 버전에 사용되지는 않음
# 테스트셋 생성 알고리즘 확인용

import json
import os
import random
import shutil
import torch
from ultralytics import YOLO
from transformers import AutoTokenizer, AutoModelForCausalLM


# 이전에 작성한 메인 파이프라인 파일에서 필요한 함수와 클래스를 가져옵니다.
from Final_pipe_Gemma2 import GrowthRegressor, analyze_and_advise

def create_test_samples_from_multiple_folders(
    image_base_dir, 
    json_base_dir, 
    dest_dir, 
    num_samples=10
):
    """
    여러 카테고리 폴더에서 임의의 샘플을 선택하고,
    이름을 변경하여 테스트 폴더를 생성합니다.
    """
    print(f"Creating a test set with {num_samples} samples from multiple categories...")
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir) # 기존 테스트 폴더가 있으면 삭제
    os.makedirs(dest_dir)

    all_image_paths = []
    # 모든 카테고리 폴더를 순회하며 이미지 파일 경로 수집
    for category in os.listdir(image_base_dir):
        category_path = os.path.join(image_base_dir, category)
        if os.path.isdir(category_path):
            for img_file in os.listdir(category_path):
                if img_file.endswith(('.png', '.jpg')):
                    all_image_paths.append(os.path.join(category_path, img_file))

    if not all_image_paths:
        print("No images found in any category.")
        return

    # 전체 이미지 목록에서 임의의 샘플 선택
    selected_image_paths = random.sample(all_image_paths, min(num_samples, len(all_image_paths)))

    # 선택된 파일을 새 이름으로 대상 폴더에 복사
    for i, src_img_path in enumerate(selected_image_paths):
        # 새 파일 이름 생성 (Test_img_1, Test_img_2, ...)
        new_base_name = f"Test_img_{i+1}"
        img_ext = os.path.splitext(src_img_path)[1] # 원본 확장자 유지 (예: .png)
        json_ext = ".json"
        
        # 원본 JSON 파일 경로 찾기
        original_base_name = os.path.splitext(os.path.basename(src_img_path))[0]
        original_category = os.path.basename(os.path.dirname(src_img_path))
        src_json_path = os.path.join(json_base_dir, original_category, original_base_name + json_ext)

        # 새 이름으로 대상 경로 설정
        dest_img_path = os.path.join(dest_dir, new_base_name + img_ext)
        dest_json_path = os.path.join(dest_dir, new_base_name + json_ext)

        # 이미지와 JSON 파일 복사 및 이름 변경
        if os.path.exists(src_json_path):
            shutil.copy2(src_img_path, dest_img_path)
            shutil.copy2(src_json_path, dest_json_path)
        else:
            print(f"Warning: JSON file for {src_img_path} not found. Skipping.")
    
    print(f"Test set with renamed files created at: {dest_dir}")


def main():
    # --- 1. 경로 설정 ---
    val_image_base_dir = "D:/지능형 스마트팜 통합 데이터 토마토/01.데이터/2.Validation/원천데이터"
    val_json_base_dir = "D:/지능형 스마트팜 통합 데이터 토마토/01.데이터/2.Validation/라벨링데이터"
    test_sample_dir = "Final_Version_final/data/test_samples"
    visual_output_dir = "Final_Version_final/data/test_visual_outputs"
    result_output_dir = "Final_Version_final/data/result_for_test" # 개별 결과 텍스트 저장 폴더
    
    os.makedirs(visual_output_dir, exist_ok=True)
    os.makedirs(result_output_dir, exist_ok=True)
    
    # --- 2. 테스트 샘플 생성 ---
    create_test_samples_from_multiple_folders(
        image_base_dir=val_image_base_dir,
        json_base_dir=val_json_base_dir,
        dest_dir=test_sample_dir,
        num_samples=10 # 테스트할 샘플 수
    )

    # --- 3. 모델 로드 ---
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

    # --- 4. 테스트 폴더의 모든 이미지에 대해 파이프라인 실행 ---
    test_files = [f for f in os.listdir(test_sample_dir) if f.endswith(('.png', '.jpg'))]
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
            
            analyze_and_advise(
                image_path=image_path,
                image_date=image_date,
                yolo_model=yolo_model,
                regressor_model=regressor_model,
                llm_tokenizer=llm_tokenizer,
                llm_model=llm_model,
                device=device,
                visual_output_path=visual_path,
                result_txt_path=result_path # 결과 텍스트 저장 경로 전달
            )
        else:
            print(f"JSON file not found for {test_image_filename}. Skipping analysis.")

# --- 스크립트 실행 ---
if __name__ == '__main__':
    main()