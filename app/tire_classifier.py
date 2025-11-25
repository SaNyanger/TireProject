import os
import cv2
import torch
import numpy as np
from torch.nn import functional as F
import torchvision.transforms as transforms
import torch.nn as nn
import model.model as module_arch
from utils import get_testloader, get_attentionloader


def create_mask_for_single_image(input_image_path, temp_img_dir, temp_mask_dir):
    """
    한 장의 타이어/트레드 이미지를
    - temp_img_dir/dummy/ 아래로 복사
    - HSV 임계값으로 마스크 생성 후 temp_mask_dir/dummy/ 에 저장
    """
    dummy_class = "dummy"  # ImageFolder용 클래스 이름
    img_class_dir = os.path.join(temp_img_dir, dummy_class)
    mask_class_dir = os.path.join(temp_mask_dir, dummy_class)

    os.makedirs(img_class_dir, exist_ok=True)
    os.makedirs(mask_class_dir, exist_ok=True)

    filename = os.path.basename(input_image_path)
    output_image_path = os.path.join(img_class_dir, filename)
    output_mask_path  = os.path.join(mask_class_dir, filename)

    img = cv2.imread(input_image_path)  # BGR
    cv2.imwrite(output_image_path, img)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 90), (190, 255, 255))
    cv2.imwrite(output_mask_path, mask)

    print(f"[마스크 생성 완료] {output_mask_path}")
    print(f"[이미지 복사 완료] {output_image_path}")

    return output_image_path, output_mask_path


def create_cam_single(config):
    """
    팀원 app.py의 create_cam_single을 약간 수정:
    - CAM 저장 후
      pred_label_str, pred_prob_val, cam_save_path 를 return
    """
    from torch.nn import functional as F  # 안전하게 재-import

    print("이미지 로딩 중...")

    test_loader, num_class = get_testloader(
        config.dataset,
        config.dataset_path,
        config.img_size
    )

    attention_loader, _ = get_attentionloader(
        config.dataset,
        config.mask_path,
        7
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cnn = module_arch.resnet101()
    num_ftrs = cnn.fc.in_features
    cnn.fc = nn.Linear(num_ftrs + 1, 3)

    checkpoint = torch.load(
        os.path.join(config.model_path, config.model_name),
        weights_only=True
    )
    cnn.load_state_dict(checkpoint['state_dict'])
    cnn = cnn.to(device)
    cnn.eval()

    feature_blobs = []

    def hook_feature(module, input, output):
        feature_blobs.append(input[0].cpu().data.numpy())

    cnn._modules['avgpool'].register_forward_hook(hook_feature)

    params = list(cnn.parameters())
    weight_softmax = np.squeeze(params[-2].cpu().data.numpy())

    def returnCAM(feature_conv, weight_softmax, class_idx):
        size_upsample = (config.img_size, config.img_size)
        _, nc, h, w = feature_conv.shape
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        cam_img = cv2.resize(cam_img, size_upsample)
        return cam_img

    classtire = ['danger', 'safety', 'warning']

    pred_label_str = None
    pred_prob_val = None
    cam_save_path = None

    for i, (val, attention) in enumerate(zip(test_loader, attention_loader)):
        image_tensor, label = val[0].to(device), val[1].to(device)
        attention = attention[0].to(device)

        origin_image_path = os.path.join(config.result_path, "input.png")
        transforms.ToPILImage()(image_tensor[0]).save(origin_image_path)

        logit = cnn(image_tensor, attention)
        h_x = F.softmax(logit, dim=1).data.squeeze()
        probs, idx = h_x.sort(0, True)

        print(f"TRUE: {label.item()} → PRED: {idx[0].item()} ({probs[0].item():.2f})")

        CAM = returnCAM(feature_blobs[0], weight_softmax, label.item())
        img = cv2.imread(origin_image_path)
        h, w, _ = img.shape

        heatmap = cv2.applyColorMap(cv2.resize(CAM, (w, h)), cv2.COLORMAP_JET)
        result = heatmap * 0.3 + img * 0.5

        cam_save_path = os.path.join(
            config.result_path,
            f"cam_result_gt({classtire[label.item()]})_pred({classtire[idx[0].item()]})_{round(probs[0].item(),2)}.png"
        )

        pred_label_str = classtire[idx[0].item()]
        pred_prob_val = float(probs[0].item())

        print(f"결과예측: {pred_label_str}")
        cv2.imwrite(cam_save_path, result)
        print("[CAM 저장 완료] →", cam_save_path)

        break  # 이미지 1장만

    return pred_label_str, pred_prob_val, cam_save_path


def classify_tread_image(
    tread_image_path: str,
    model_path: str = "./models/",
    model_name: str = "model_best_weights.pth",
    img_size: int = 224,
    result_path: str = "./saved/cam_result/",
    temp_image_dir: str = "./temp_image/",
    temp_mask_dir: str = "./temp_mask/",
):
    """
    언랩된 타이어(트레드) 이미지 경로 하나를 받아서
    - temp 폴더/마스크 생성
    - 모델 분류 + CAM 생성
    - (라벨 문자열, 확률, CAM 이미지 경로) 를 반환.
    """
    os.makedirs(result_path, exist_ok=True)

    print("\n### STEP 1. 마스크 생성 ###")
    img_path, mask_path = create_mask_for_single_image(
        tread_image_path,
        temp_image_dir,
        temp_mask_dir
    )

    print("\n### STEP 2. 모델 분석 준비 ###")

    class Cfg:
        pass

    config = Cfg()
    config.input_image = tread_image_path
    config.model_path = model_path
    config.model_name = model_name
    config.img_size = img_size
    config.result_path = result_path
    config.temp_image_dir = temp_image_dir
    config.temp_mask_dir = temp_mask_dir

    config.dataset = "OWN"
    config.dataset_path = temp_image_dir
    config.mask_path = temp_mask_dir

    print("\n### STEP 3. CAM 생성 ###")
    pred_label_str, pred_prob_val, cam_save_path = create_cam_single(config)

    return pred_label_str, pred_prob_val, cam_save_path
