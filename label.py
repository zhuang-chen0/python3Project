from transformers import pipeline, AutoProcessor, AutoModelForZeroShotImageClassification, AutoModel
from PIL import Image
import accelerate
import requests
import torch
import json
import os
import io
from datetime import datetime
import cv2

# 输出当前时间
print("Current time:", datetime.now())
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

labels=[
  "Scenery and Landscapes",
  "Local Food and Cuisine",
  "Cultural Activities and Traditions",
  "Adventure and Sports",
  "City Life and Urban Exploration",
  "Wildlife and Nature",
  "History and Architecture",
  "Art and Music",
  "News and Current Affairs",
]

model = AutoModel.from_pretrained("laion/CLIP-ViT-g-14-laion2B-s12B-b42K", device_map="auto")
processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-g-14-laion2B-s12B-b42K", device_map="auto")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# for name, url in {
#     "mp4": "https://share.anylocal.cn/cn-north-1/anylocal-upload/uploaded/James/e7638927-3af8-4def-ba2f-401c9f1c2ffc.mp4"
#     # "food": "https://share.anylocal.cn/cn-north-1/anylocal-upload/uploaded/18179380542/1711880307122.jpeg",
#     # "nature": "https://share.anylocal.cn/cn-north-1/anylocal-upload/uploaded/w_1709346568/1711850715023.jpg",
#     # "city": "https://share.anylocal.cn/cn-north-1/anylocal-upload/uploaded/w_1706248655/1711506281244.jpeg",
#     }.items():
#     print(f"processing {name} ({url})")
#     if url.endswith("mp4") | url.endswith("mov"):
#         video_bytes = io.BytesIO()
#         response = requests.get(url, stream=True)
#         response.raise_for_status()
#         for chunk in response.iter_content(chunk_size=8192):
#             video_bytes.write(chunk)
#         # 使用 OpenCV 读取视频
#         cap = cv2.VideoCapture(url)
#         # 帧率
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         ret, frame = cap.read()
#         # 保存第一帧
#         image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     else:
#         image = Image.open(requests.get(url, stream=True).raw)
#     inputs = processor(images=image, text=labels, return_tensors="pt", padding=True)
#     inputs = {k: v.to(device) for k, v in inputs.items()}
#
#     with torch.no_grad():
#         outputs = model(**inputs)
#
#     logits = outputs.logits_per_image[0]
#     probs = logits.softmax(dim=-1).cpu().numpy()
#     scores = probs.tolist()
#
#     result = [
#         {"score": score, "label": label}
#         for score, label in sorted(zip(probs, labels), key=lambda x: -x[0])
#     ]
#     for re in result:
#         print(f"{name} - {re['score']:.1%} - {re['label']}")

# print("Current time:", datetime.now())


def label(image):
    print("label picture start.....")
    inputs = processor(images=image, text=labels, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits_per_image[0]
    probs = logits.softmax(dim=-1).cpu().numpy()
    scores = probs.tolist()

    result = [
        {"score": score, "label": label_name}
        for score, label_name in sorted(zip(probs, labels), key=lambda x: -x[0])
    ]
    print(f'result: {result}')
    return result
