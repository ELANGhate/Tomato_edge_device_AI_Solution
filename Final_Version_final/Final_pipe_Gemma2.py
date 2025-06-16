# Final_pipe_Gemma2.py
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import json
import os
from ultralytics import YOLO
# Hugging Face 라이브러리를 다시 사용합니다.
from transformers import AutoTokenizer, AutoModelForCausalLM
import copy


# --- 1. 회귀 모델 클래스 정의 (변경 없음) ---
class GrowthRegressor(nn.Module):
    def __init__(self, num_output_features=1, dropout_rate=0.2):
        super(GrowthRegressor, self).__init__()
        mobilenet_v3 = models.mobilenet_v3_small(weights=None)
        self.features = nn.Sequential(
            *list(mobilenet_v3.features),
            mobilenet_v3.avgpool
        )
        self.regressor_head = nn.Sequential(
            nn.Linear(576, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_output_features)
        )
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        output = self.regressor_head(x)
        return output

# --- 2. 이미지 전처리 함수 (변경 없음) ---
def preprocess_image_for_regressor(image_crop_pil, target_size=(224, 224)):
    """크롭된 PIL 이미지를 회귀 모델 입력에 맞게 전처리합니다."""
    preprocess = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image_crop_pil).unsqueeze(0)

# --- 3. 메인 파이프라인 함수 (수정됨) ---
def analyze_and_advise(
    image_path,
    image_date,   #촬영 날짜 추가
    yolo_model, 
    regressor_model, 
    llm_tokenizer, # 토크나이저를 다시 인자로 받도록 수정
    llm_model, 
    device,
    visual_output_path=None, # <-- 시각화 결과를 저장할 경로 추가
    result_txt_path=None # <-- 결과 텍스트를 저장할 경로 추가
):
    """
    하나의 이미지에 대해 전체 분석 및 조언 파이프라인을 실행하고,
    YOLO 탐지 결과를 시각화하여 저장합니다
    """
    
    # --- 단계  1 & 2: YOLO 분석 및 크기 예측  ---
    print("--- 1. Running YOLO Detection & Regression ---")
    
    excluded_labels = [
        'tom_flower_center_bb', 'tom_flower_tip_bb', 'tom_post_flowering_poly',
        'tom_stem_dimeter_bb', 'tom_stem_dimeter_tilt_bb'
    ]
    excluded_class_ids = [k for k, v in yolo_model.names.items() if v in excluded_labels]

    yolo_results = yolo_model(image_path, verbose=False)
    pil_image = Image.open(image_path).convert("RGB")
    

    # --- 시각화 기능 추가 ---
    if visual_output_path and yolo_results:
        # yolo_results[0].plot()는 바운딩 박스가 그려진 이미지를 반환합니다.
        # labels=False, conf=False로 설정하면 클래스 이름과 신뢰도 점수를 숨길 수 있습니다.
        visualized_image_array = yolo_results[0].plot(labels=True, conf=True)
        # plot() 결과는 BGR 순서의 NumPy 배열이므로, PIL Image로 변환하려면 RGB 순서로 변경
        visualized_image_pil = Image.fromarray(visualized_image_array[..., ::-1])
        visualized_image_pil.save(visual_output_path)
        print(f"Visualized result saved to: {visual_output_path}")


    detected_objects = []
    predicted_height = None
    
    if yolo_results:
        for det in yolo_results[0].boxes.data:
            x1, y1, x2, y2, conf, cls_id_float = det.tolist()
            cls_id = int(cls_id_float)

            if cls_id not in excluded_class_ids:
                class_name = yolo_model.names[cls_id]
                detected_objects.append(class_name)

                if class_name == 'tom_growth_bb':
                    cropped_patch = pil_image.crop((x1, y1, x2, y2))
                    input_tensor = preprocess_image_for_regressor(cropped_patch).to(device)
                    
                    with torch.no_grad():
                        predicted_height = regressor_model(input_tensor).item()

    unique_detected_classes = sorted(list(set(detected_objects)))
    print("--- Analysis Complete ---")
    print(f"Detected Objects (filtered): {unique_detected_classes}")
    if predicted_height:
        print(f"Predicted Plant Height: {predicted_height:.2f} cm")

    # --- 단계 3: Gemma 입력용 텍스트 생성 (업데이트) ---
    # 3-1. 시스템 프롬프트: 모델에게 역할과 방대한 배경 지식을 제공
    SYSTEM_PROMPT = """
당신은 토마토 재배 전문가 AI 어시스턴트입니다. 당신의 임무는 아래의 '핵심 재배 지식'과 사용자가 제공하는 '현재 토마토 분석 결과'(특히 **예측된 식물 높이**)를 종합적으로 고려하여, 개인 농장주를 위한 맞춤형 진단과 다음 행동 계획을 구체적이고 친절하게 조언하는 것입니다.

### 핵심 재배 지식

**1. 키(cm)에 따른 생육 단계 진단:**
- **15cm ~ 30cm:** **모종 키우기 단계**. 줄기가 굵어지고 잎 수가 증가하는 시기입니다.
- **50cm ~ 70cm:** **아주심기 후 활착 단계**. 밭으로 옮겨 심은 후 뿌리가 자리 잡고 폭발적인 성장을 시작하는 시점입니다. 지지대를 준비해야 합니다.
- **150cm 이상:** **왕성한 생장 및 개화기**. 첫 꽃이 피는 시기이며, 곁순(측지) 제거가 매우 중요해집니다.
- **200cm ~ 300cm:** **결실 및 수확기**. 열매가 열리고 익기 시작하며, 성장, 개화, 결실, 수확이 동시에 일어나는 복합적인 관리기입니다.

**2. 라벨 의미 해석:**
- `tom_growth_bb`: 식물 전체 또는 일부의 생장 길이.
- `tom_flower_stem_bb`: 화방(꽃/열매 묶음)의 높이.
- `tom_leaf_bb`: 잎의 크기.
- `tom_flower_count_bb`: 개화한 꽃들의 군집.
- `tom_fruit_count_bb`: 열매가 달린 군집.
- `tom_pre_flowering_poly`: 꽃봉오리 (아직 피지 않음).
- `tom_flower_half_poly`: 반쯤 핀 꽃.
- `tom_flower_poly`: 활짝 핀 꽃.
- `tom_fruit_breaker_poly`: 착색 시작 단계의 초록색 열매.
- `tom_fruit_pink_poly`: 익어가는 핑크색 열매.
- `tom_fruit_red_poly`: 완전히 익은 붉은색 열매.

**3. 핵심 관리 지침:**
- **온도**: 최적 온도는 낮 25~27℃, 밤 18~20℃입니다. 야간 온도가 13℃ 이하로 떨어지면 수정 불량이나 기형 과일이 발생할 수 있습니다.
- **햇빛**: 강한 광선을 선호합니다. 햇빛이 부족하면 꽃이 떨어질 수 있습니다.
- **수정 및 착과**: 야간 온도가 13~24℃ 범위일 때 수정이 가장 잘 됩니다.
- **곁순 제거 (적화)**: 과일 품질과 식물 전체의 성장을 위해 한 화방 당 4~5개의 과일만 남기고 나머지는 꽃일 때 미리 제거해주는 것이 좋습니다. 곁순 제거는 오전에 손으로 하는 것이 좋습니다.
- **주요 병충해**: 잿빛곰팡이병, 잎곰팡이병, 역병 (주로 저온다습 환경 유의), 온실가루이, 파밤나방, 진딧물.

**4. 재배 일정:**
  - 3월 초: 씨뿌리기
  - 4월 말: 아주심기(옮겨 심기)
  - 6월 말 ~ 9월 중순: 수확

**5. 생장 정도:**
  3월	실내 파종, 발아	- 1주~2주: 1~2cm, 3주~4주: 5~10cm. 떡잎에서 본잎이 나옴. 햇빛 관리 중요.
  4월	모종 키우기	15~30cm. 줄기가 굵어지고 잎 수가 증가.
  5월	아주심기(밭으로 이동)	- 초~중순: 성장 더뎌 보임. 5월 말: 50~70cm	뿌리 활착 후 폭발적 성장 시작. 지지대 준비.
  6월	왕성한 생장, 개화	1.5m 이상	첫 꽃이 피고, 곁순 제거가 중요해짐.
  7월~	결실, 수확	2~3m까지 계속 성장.	열매가 열리고 익기 시작. 성장, 개화, 결실, 수확이 동시에 일어남.


이제 아래의 '현재 토마토 분석 결과'를 바탕으로 조언을 생성해주세요.
"""

    
     # 3-2. 사용자 프롬프트: 현재 분석 결과를 구조화하여 전달
    user_prompt_context = "다음은 제가 재배하는 토마토 사진에 대한 AI 분석 결과입니다.\n\n"
    user_prompt_context += f"### 이미지 분석 내용\n"
    user_prompt_context += f"- **촬영 시점**: {image_date}\n" # <-- 날짜 정보 추가!
    user_prompt_context += f"- 탐지된 객체: {', '.join(unique_detected_classes) if unique_detected_classes else '특정 객체 탐지 안됨'}\n"
    if predicted_height:
        user_prompt_context += f"- 예측된 식물 높이: 약 {predicted_height:.1f} cm\n"
    user_prompt_context += "\n분석 부탁드립니다."

    # 3-3. Gemma -it 모델에 맞는 최종 프롬프트 생성
    chat = [
        { "role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user_prompt_context}"},
    ]
    prompt = llm_tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)




    # --- 단계 4: Gemma로 조언 생성 (수정됨) ---
    print("\n--- 2. Generating Advice with Gemma... ---")
    
    # 별도로 전달받은 llm_tokenizer 객체를 사용
    inputs = llm_tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = llm_model.generate(input_ids=inputs, max_new_tokens=1024)
    
    # 별도로 전달받은 llm_tokenizer 객체로 결과 디코딩
    response_text = llm_tokenizer.decode(outputs[0])
    
    # 모델의 답변 부분만 깔끔하게 추출
    advice_only = response_text[len(llm_tokenizer.decode(inputs[0])):]
    advice_only = advice_only.replace("<eos>", "").strip()

    # 결과 출력 및 저장 (result를 저장하도록록수정)
    print("\n--- AI 기반 관리 조언 ---")
    print(advice_only)
    
    if result_txt_path: # 결과 저장 경로가 지정되었을 경우에만 파일로 저장
        with open(result_txt_path, "w", encoding="utf-8") as f:
            f.write("--- 토마토 생육 상태 분석 결과 ---\n\n")
            f.write( user_prompt_context)
            f.write("\n\n--- AI 기반 관리 조언 ---\n\n")
            f.write(advice_only)
        print(f"\n--- 결과가 '{result_txt_path}' 파일에 저장되었습니다. ---")


# --- 메인 실행 블록 (수정됨) ---
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 변수 초기화 ---
    yolo_model, regressor_model, llm_tokenizer, llm_model = None, None, None, None
    
    # 1. 학습된 모델들 로드
    print("Loading models...")
    yolo_model_path = 'Final_Version/Yolov11_nano/best.pt'
    regressor_model_path = 'Final_Version/Mobilenetv3_small/best_regressor_model_V3_plantHeight.pth'
    llm_model_name = "google/gemma-2-2b-it"  # Gemma 2B Instruction Tuned 모델 사용
    
    try:
        yolo_model = YOLO(yolo_model_path)
        
        regressor_model = GrowthRegressor(num_output_features=1)
        regressor_model.load_state_dict(torch.load(regressor_model_path, map_location=device))
        regressor_model.to(device)
        regressor_model.eval()

        # --- 토크나이저와 모델을 각각 로드 ---
        llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype=torch.float16,
            device_map="auto" # GPU에 자동으로 모델 할당
        )
        
    except Exception as e:
        print(f"Error loading models: {e}")
        exit()
        
    print("All models loaded successfully.")

    # 2. 분석할 이미지 지정
    target_image = "D:/지능형 스마트팜 통합 데이터 토마토/01.데이터/2.Validation/원천데이터/a1.생장길이/V001_tom1_42_021_a1_00_20211019_17_01082317_49122255.png"
    target_json = "D:/지능형 스마트팜 통합 데이터 토마토/01.데이터/2.Validation/라벨링데이터/a1.생장길이/V001_tom1_42_021_a1_00_20211019_17_01082317_49122255.json"

    if not os.path.exists(target_image):
        print(f"Error: Target image not found at {target_image}")
    elif not os.path.exists(target_json):
        print(f"Error: Target JSON file not found at {target_json}")
    else:
        # JSON 파일에서 촬영 날짜 읽어오기
        with open(target_json, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        image_date = json_data.get("file_attributes", {}).get("date", "알 수 없음")

        # 3. 파이프라인 실행
        if all([yolo_model, regressor_model, llm_tokenizer, llm_model]):
             analyze_and_advise(target_image, image_date, yolo_model, regressor_model, llm_tokenizer, llm_model, device)
        else:
            print("One or more models failed to load. Exiting.")