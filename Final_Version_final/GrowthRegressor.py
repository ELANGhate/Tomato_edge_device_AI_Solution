import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

class GrowthRegressor(nn.Module):
    def __init__(self, num_output_features=1, dropout_rate=0.2):
        super(GrowthRegressor, self).__init__()
        # MobileNetV3-Small 사전 학습 모델 로드
        mobilenet_v3 = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        
        # MobileNetV3-Small의 특징 추출기 부분만 사용
        # features 다음에는 AdaptiveAvgPool2d(output_size=1)이 오고, 
        # 그 다음 classifier가 (Linear -> Hardswish -> Dropout -> Linear) 구조임.
        # 마지막 Linear 계층의 입력 특징 수는 576 (features의 출력 채널 수) * avgpool 후.
        # avgpool까지 포함하여 특징을 추출합니다.
        self.features = nn.Sequential(
            *list(mobilenet_v3.features),
            mobilenet_v3.avgpool 
        )
        
        # 회귀 헤드: 입력 특징 576개 (MobileNetV3-Small의 avgpool 출력 채널 수)
        self.regressor_head = nn.Sequential(
            nn.Linear(576, 128),  # 입력 특징 수
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_output_features) # 최종 출력 (예: 길이 1개, 또는 여러 지표 동시 예측 시 >1)
        )

    def forward(self, x):
        # x: [batch_size, 3, input_height, input_width] (예: 224x224)
        x = self.features(x)       # 특징 추출
        x = torch.flatten(x, 1)    # [batch_size, 576] 형태로 flatten
        output = self.regressor_head(x) # 회귀 예측
        return output

# --- 모델 학습 및 사용을 위한 추가 유틸리티 (예시) ---
def preprocess_image_for_regressor(image_crop_pil, target_size=(224, 224)):
    """
    YOLO로 크롭된 PIL 이미지를 MobileNetV3 입력에 맞게 전처리합니다.
    """
    preprocess = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet 표준 정규화
    ])
    return preprocess(image_crop_pil).unsqueeze(0) # 배치 차원 추가

# --- 회귀 모델 학습 (개념적인 설명) ---
# 1. 데이터 준비:
#    - 각 JSON 파일과 해당 이미지에 대해:
#        - `imagePath`로 이미지 로드.
#        - `shapes`에서 특정 라벨(예: "tom_stem_dimeter_tilt_bb")의 바운딩 박스 정보로 이미지 크롭.
#        - `growth_indicators`에서 해당 라벨에 대한 정답 값(예: "stemDiameter") 가져오기.
#    - (크롭된 이미지 패치, 정답 값) 쌍으로 데이터셋 구성.
# 2. 모델 인스턴스 생성:
#    stem_diameter_model = GrowthRegressor(num_output_features=1)
# 3. 손실 함수 및 옵티마이저 정의:
#    criterion = nn.MSELoss() # 평균 제곱 오차 손실
#    optimizer = torch.optim.Adam(stem_diameter_model.parameters(), lr=0.001)
# 4. 학습 루프 실행:
#    - 데이터로더로 배치 단위 학습.
#    - 예측값과 정답 값으로 손실 계산 후 역전파.
# 5. 학습된 모델 가중치 저장:
#    torch.save(stem_diameter_model.state_dict(), "stem_diameter_regressor.pth")

# plantHeight 예측 모델도 위와 유사하게 별도의 데이터로 학습합니다.



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms
from PIL import Image # 이미지 로딩 및 크롭을 위해 PIL 사용
import json
import os
from sklearn.model_selection import train_test_split # 데이터 분할용
import copy # 모델 가중치 복사용
from tqdm import tqdm # 작업 확인용

# --- 1. GrowthRegressor 모델 정의 (이전과 동일) ---
class GrowthRegressor(nn.Module):
    def __init__(self, num_output_features=1, dropout_rate=0.2):
        super(GrowthRegressor, self).__init__()
        mobilenet_v3 = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
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

# --- 2. 커스텀 데이터셋 클래스 ---
class TomatoGrowthDataset(Dataset):
    def __init__(self, json_files, base_image_dir, class_mapping, target_indicator_key, relevant_bbox_labels, transform=None):
        """
        Args:
            json_files (list): 사용할 JSON 파일 경로 목록.
            base_image_dir (str): 원본 이미지가 있는 기본 디렉토리 경로.
                                  (JSON의 imagePath와 조합하여 실제 이미지 경로 구성)
            class_mapping (dict): 라벨 문자열을 클래스 ID로 매핑하는 딕셔너리 (여기서는 사용 안함, YOLO용)
            target_indicator_key (str): JSON의 growth_indicators에서 가져올 목표 값의 키 (예: "plantHeight", "stemDiameter").
            relevant_bbox_labels (list): 이 target_indicator를 예측하기 위해 크롭할 바운딩 박스의 라벨 목록.
                                         (예: plantHeight 예측 시 ["tom_growth_bb"])
            transform (callable, optional): 이미지에 적용할 전처리 변환.
        """
        self.json_files = json_files
        self.base_image_dir = base_image_dir
        self.target_indicator_key = target_indicator_key
        self.relevant_bbox_labels = relevant_bbox_labels # 리스트 형태로 여러 라벨 지정 가능
        self.transform = transform
        print("Loading and preparing data samples... This might take a while.")
        self.data_samples = self._load_data()
        print(f"Finished loading. Found {len(self.data_samples)} valid samples.")

    def _load_data(self):
        samples = []
        for json_filepath in self.json_files:
            try:
                with open(json_filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                image_filename_from_json = data.get("imagePath")
                img_width = data.get("imageWidth")
                img_height = data.get("imageHeight")
                
                # JSON 파일이 속한 카테고리 폴더명 (예: a1.생장길이)을 알아내기 위해
                category_folder_name = os.path.basename(os.path.dirname(json_filepath))
                # 원본 이미지 경로 구성
                original_image_filepath = os.path.join(self.base_image_dir, category_folder_name, image_filename_from_json)

                if not image_filename_from_json or not img_width or not img_height or not os.path.exists(original_image_filepath):
                    # print(f"Skipping {json_filepath} due to missing info or image file not found at {original_image_filepath}")
                    continue

                # growth_indicators에서 목표 값 가져오기
                indicator_value = data.get("growth_indicators", {}).get(self.target_indicator_key)
                if indicator_value is None: # 또는 유효하지 않은 값일 경우 건너뛰기
                    # print(f"Target indicator '{self.target_indicator_key}' not found or invalid in {json_filepath}. Skipping.")
                    continue
                
                try:
                    target_value = float(indicator_value) # 수치형으로 변환
                except ValueError:
                    # print(f"Could not convert indicator value '{indicator_value}' to float for {json_filepath}. Skipping.")
                    continue

                # relevant_bbox_labels에 해당하는 바운딩 박스 정보 추출
                for shape in data.get("shapes", []):
                    label_name = shape.get("label")
                    if label_name in self.relevant_bbox_labels and shape.get("shape_type") == "rectangle":
                        points = shape.get("points")
                        x1, y1 = points[0]
                        x2, y2 = points[1]
                        bbox = (int(min(x1,x2)), int(min(y1,y2)), int(max(x1,x2)), int(max(y1,y2))) # (xmin, ymin, xmax, ymax)
                        samples.append({
                            "image_path": original_image_filepath,
                            "bbox": bbox,
                            "target": target_value
                        })
                        # 하나의 JSON에서 relevant_bbox_labels에 해당하는 첫 번째 bbox만 사용할 수도 있고,
                        # 여러 개라면 모두 샘플로 추가할 수도 있습니다. 여기서는 모두 추가합니다.
            except Exception as e:
                print(f"Error processing file {json_filepath}: {e}")
        return samples

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
         # --- 디버깅을 위해 어떤 샘플에서 멈추는지 확인 ---
        #if idx % 100 == 0: # 100개 샘플마다 로그 출력
         #    print(f"DataLoader is getting item index: {idx}")
        sample = self.data_samples[idx]
        image_path = sample["image_path"]
        bbox = sample["bbox"]
        target = sample["target"]

        try:
            image = Image.open(image_path).convert("RGB")
            cropped_image = image.crop(bbox) # 바운딩 박스 영역 크롭

            if self.transform:
                cropped_image = self.transform(cropped_image)
            
            return cropped_image, torch.tensor(target, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading or cropping image {image_path} with bbox {bbox}: {e}")
            # 오류 발생 시 None을 반환하거나, 기본 이미지를 반환할 수 있습니다.
            # 여기서는 간단히 다음 샘플로 넘어가도록 DataLoader에서 처리될 수 있게 예외를 다시 발생시키거나,
            # 유효한 빈 텐서를 반환할 수 있습니다. (DataLoader의 collate_fn에서 None 처리 필요)
            # 지금은 일단 빈 텐서 반환
            return torch.zeros((3, 224, 224)), torch.tensor(0.0, dtype=torch.float32)


# --- 3. 학습 루프 함수 ---
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, device="cpu"):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            
            if phase not in dataloaders or dataloaders[phase] is None:
                print(f"No dataloader found for phase: {phase}. Skipping this phase for epoch {epoch+1}.")
                continue
            if len(dataloaders[phase].dataset) == 0:
                print(f"Dataset for phase {phase} is empty. Skipping this phase for epoch {epoch+1}.")
                continue

            for inputs, targets in dataloaders[phase]:
                inputs = inputs.to(device)
                targets = targets.to(device).unsqueeze(1) # [batch_size, 1] 형태로 변경

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print(f'{phase} Loss: {epoch_loss:.4f}')

            if phase == 'val' and epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f'New best val loss: {best_val_loss:.4f}. Saving model...')
                torch.save(model.state_dict(), f'best_regressor_model_V3_{target_indicator_key}.pth')


    print(f'Best val Loss: {best_val_loss:4f}')
    model.load_state_dict(best_model_wts)
    return model

# --- 4. 메인 실행 부분 ---

if __name__ == '__main__':
    # --- 설정값 (사용자 정의 필요) ---
    # 'plantHeight' 예측 학습인지, 'stemDiameter' 예측 학습인지 등을 설정
    TRAIN_FOR = "plantHeight" # 또는 "stemDiameter"

    if TRAIN_FOR == "plantHeight":
        target_indicator_key = "plantHeight"
        # plantHeight를 예측하기 위해 어떤 바운딩 박스 영역을 사용할지 결정
        # 예: 식물 전체를 나타내는 라벨 또는 '생장길이' 측정과 관련된 라벨
        relevant_bbox_labels_for_task = ["tom_growth_bb"] # 또는 ["tom_plant_bb"] 등 (데이터에 맞게)
        # `생장길이.json` 파일이 있는 카테고리 폴더만 지정
        target_category_folders = ["a1.생장길이"]
    elif TRAIN_FOR == "stemDiameter":
        target_indicator_key = "stemDiameter"
        relevant_bbox_labels_for_task = ["tom_stem_dimeter_bb", "tom_stem_dimeter_tilt_bb"]
        # `줄기두께.json` 파일이 있는 카테고리 폴더만 지정
        target_category_folders = ["a3.줄기두께"]
    else:
        raise ValueError("TRAIN_FOR must be 'plantHeight' or 'stemDiameter'")

    # 윈도우 경로 기준으로 수정
    base_label_dir = "D:/지능형 스마트팜 통합 데이터 토마토/01.데이터/1.Training/라벨링데이터"
    base_image_dir = "D:/지능형 스마트팜 통합 데이터 토마토/01.데이터/1.Training/원천데이터"
    # base_image_dir_add = "/mnt/d/지능형 스마트팜 통합 데이터 토마토/01.데이터/1.Training/원천데이터_230630_add" # 필요시 주석 해제 및 로직 추가

    # WSL 환경에서 실행할 경우
    #base_label_dir = "/mnt/d/지능형 스마트팜 통합 데이터 토마토/01.데이터/1.Training/라벨링데이터"
    #base_image_dir = "/mnt/d/지능형 스마트팜 통합 데이터 토마토/01.데이터/1.Training/원천데이터"



    # 이미지 전처리 정의
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)), # MobileNetV3 입력 크기
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }

    # 1. 학습에 사용할 JSON 파일 목록 수집
    all_json_files = []
    for cat_folder in target_category_folders:
        cat_path = os.path.join(base_label_dir, cat_folder)
        if os.path.isdir(cat_path):
            for fname in os.listdir(cat_path):
                if fname.endswith(".json"):
                    all_json_files.append(os.path.join(cat_path, fname))
    
    if not all_json_files:
        print(f"No JSON files found for categories: {target_category_folders}. Exiting.")
        exit()

    # 2. 전체 데이터를 학습 및 검증 세트로 분할 (파일 경로 기준)
    train_json_files, val_json_files = train_test_split(all_json_files, test_size=0.2, random_state=42)

    print(f"Total JSON files for {TRAIN_FOR}: {len(all_json_files)}")
    print(f"Training JSON files: {len(train_json_files)}")
    print(f"Validation JSON files: {len(val_json_files)}")

    # 3. 데이터셋 및 데이터로더 생성
    image_datasets = {
        'train': TomatoGrowthDataset(train_json_files, base_image_dir, {}, target_indicator_key, relevant_bbox_labels_for_task, transform=data_transforms['train']),
        'val': TomatoGrowthDataset(val_json_files, base_image_dir, {}, target_indicator_key, relevant_bbox_labels_for_task, transform=data_transforms['val'])
    }
    
    # DataLoader 생성 전 데이터셋 크기 확인
    if len(image_datasets['train']) == 0 or len(image_datasets['val']) == 0 :
        print("One of the datasets (train or val) is empty after loading. Check paths, JSON content, and relevant_bbox_labels. Exiting.")
        exit()

    dataloaders = {
        x: DataLoader(
            image_datasets[x], 
            batch_size=32, 
            shuffle=(x=='train'), 
            num_workers=4, 
            pin_memory=True,
            persistent_workers=True
            ) # num_workers는 환경에 맞게 조절, 오류발생, 0으로 수정정
        for x in ['train', 'val']
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print(f"Dataset sizes: {dataset_sizes}")


    # 4. 모델, 손실함수, 옵티마이저 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_reg = GrowthRegressor(num_output_features=1).to(device)
    criterion_reg = nn.MSELoss()
    optimizer_reg = optim.Adam(model_reg.parameters(), lr=0.0005)

    # 5. 모델 학습
    print(f"\n--- Starting training for: {TRAIN_FOR} ---")
    trained_model = train_model(model_reg, dataloaders, criterion_reg, optimizer_reg, num_epochs=50, device=device) # 에폭 수는 조절

    # 6. 최종 모델 저장 (선택 사항, train_model 함수 내에서 best 모델은 이미 저장됨)
    # torch.save(trained_model.state_dict(), f'final_regressor_model_{target_indicator_key}.pth')
    print(f"Training complete for {TRAIN_FOR}. Best model saved as best_regressor_model_V3_{target_indicator_key}.pth")

