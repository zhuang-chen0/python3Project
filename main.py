from PIL import Image
import requests
import io
import cv2
from label import label


def mail(url):
    print(f"processing ({url})")
    response = requests.get(url, stream=True)
    if url.endswith("mp4") | url.endswith("mov"):
        print("handle video start......")
        video_bytes = io.BytesIO()
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=8192):
            video_bytes.write(chunk)
        # 使用 OpenCV 读取视频
        cap = cv2.VideoCapture(url)
        # 帧率
        fps = cap.get(cv2.CAP_PROP_FPS)
        # 要提取的帧间隔（每三秒提取一帧）
        interval = int(fps * 3)
        result = []
        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            current_second = current_frame / fps
            if frame_index == 0 | frame_index % interval == 0:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result.append(label(image))
            frame_index += 1
        if len(result) == 0:
            return []
        elif len(result) == 1:
            return result[0]
        else:
            label_percentage = {}
            for frame_label in result:
                for label_data in frame_label:
                    label_name = label_data['label']
                    score = label_data['score']
                    label_percentage[label_name] = label_percentage.get(label_name, 0) + score
            total_frames = len(result)
            for label_data in label_percentage:
                label_percentage[label_data] /= total_frames
            average_label_list = [{'score': label_percentage[label_name], 'label': label_name} for label_name in label_percentage]
            return average_label_list
    else:
        print("handle image start......")
        return Image.open(response.raw)


res = mail("https://share.anylocal.cn/cn-north-1/anylocal-upload/uploaded/James/e7638927-3af8-4def-ba2f-401c9f1c2ffc.mp4")
for re in res:
    print(f"{re['score']:.1%} - {re['label']}")