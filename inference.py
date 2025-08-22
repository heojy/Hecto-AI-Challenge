import os
import torch
import pandas as pd
from PIL import Image
from glob import glob
from torchvision import models
from torchvision.models import EfficientNet_B3_Weights
import torch.nn as nn

def inference_softmax(
    model_path="efficientnet_b3_finetuned.pth",
    test_dir="/home/a/hecto_AI/data/test",
    sample_csv="sample_submission.csv",
    output_csv="submission.csv"
):
    # 클래스 정보 파악
    submission_df = pd.read_csv(sample_csv)
    class_names = list(submission_df.columns)[1:]
    num_classes = len(class_names)

    # 최신 weights 객체 불러오기 및 transforms 정의
    weights = EfficientNet_B3_Weights.DEFAULT
    transform = weights.transforms()

    # 모델 구성
    model = models.efficientnet_b3(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    # 가중치 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Model loaded with {num_classes} output classes.\n")

    # 테스트 이미지 로드
    test_image_paths = sorted(glob(os.path.join(test_dir, "*.jpg")) + glob(os.path.join(test_dir, "*.png")))
    print(f"Found {len(test_image_paths)} test images in {test_dir}")

    output_df = submission_df.copy()
    output_df[class_names] = 0

    # 추론 루프
    for idx, img_path in enumerate(test_image_paths):
        print(f"Predicting {idx+1}/{len(test_image_paths)}: {os.path.basename(img_path)}")
        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(image)
            probs = torch.softmax(logits, dim=1)

        output_df.iloc[idx, 1:] = probs.squeeze().cpu().numpy()

    output_df.to_csv(output_csv, index=False)
    print(f"\n★ Submission saved to {output_csv}")

if __name__ == "__main__":
    inference_softmax()