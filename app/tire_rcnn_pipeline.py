import os
import cv2
import numpy as np
import traceback
import csv
import json
from tire_classifier import classify_tread_image

# =========================
# 튜닝 파라미터 (숫자만 바꿔도 됨)
# =========================
IMG_NAME = "car1.jpg"      # 기본 입력 이미지 이름 (assets 폴더 기준)

MAX_SIDE = None            # 긴 변 리사이즈 상한(None이면 원본 유지)
CROP_PAD = 1.75            # 후보 박스 여유 배수
UPSCALE_BEFORE_UNWRAP = 1.8  # RCNN/언랩 전 업스케일 배수(1.6~2.0 권장)
STD_EXPORT_WIDTH = 256     # 언랩 결과의 표준 폭(후처리 리사이즈)
USE_CUDA = False           # GPU 사용 여부 (CUDA 있으면 True로 시도 가능)

# =========================
# IOU / NMS 유틸
# =========================
def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union

def nms(boxes, scores, thr=0.4):
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs):
        i = idxs[0]
        keep.append(i)
        rest = idxs[1:]
        remain = []
        for j in rest:
            if iou(boxes[i], boxes[j]) <= thr:
                remain.append(j)
        idxs = np.array(remain)
    return [boxes[i] for i in keep], [scores[i] for i in keep]

# =========================
# 후보 ROI 탐색 (타원 + 허프서클 → NMS)
# =========================
def find_candidates(gray):
    """
    입력: gray (uint8, 단일 채널)
    반환: (boxes, scores)
      - boxes: [x1,y1,x2,y2] 리스트
      - scores: 각 후보의 간단 점수(크게 의미두진 않아도 됨)
    """
    h, w = gray.shape
    boxes, scores = [], []

    # A) treadscan의 타원 검출 (휠 전체가 잘 보이는 경우에 강함)
    try:
        import treadscan
        seg = treadscan.Segmentor(gray)
        ell = seg.find_ellipse(threshold=135, min_area=0)
        if ell is not None:
            (xc, yc), (MA, ma), ang = ell
            r = max(MA, ma) * 0.55 * CROP_PAD
            x1, y1 = int(max(0, xc - r)), int(max(0, yc - r))
            x2, y2 = int(min(w, xc + r)), int(min(h, yc + r))
            if (x2 - x1) > 30 and (y2 - y1) > 30:
                boxes.append([x1, y1, x2, y2])
                circ = float(min(MA, ma) / (max(MA, ma) + 1e-6))  # 원형 비율
                scores.append(circ)
    except Exception:
        # 타원 검출 실패해도 전체 파이프라인은 계속 간다
        pass

    # B) 허프 서클 (림이 더 뚜렷한 경우에 강함)
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(
        g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min(h, w)//4,
        param1=120, param2=40, minRadius=40, maxRadius=0
    )
    if circles is not None:
        for c in circles[0, :]:
            cx, cy, r = int(c[0]), int(c[1]), int(c[2])
            r = int(r * CROP_PAD)
            x1, y1 = max(0, cx - r), max(0, cy - r)
            x2, y2 = min(w, cx + r), min(h, cy + r)
            if (x2 - x1) > 30 and (y2 - y1) > 30:
                boxes.append([x1, y1, x2, y2])
                roi = gray[y1:y2, x1:x2]
                edges = cv2.Canny(roi, 80, 160)
                scores.append(float(edges.mean()) / 255.0)

    # 후보가 하나도 없으면 중앙 크롭 폴백
    if not boxes:
        print("[INFO] no ellipse/circle found, fallback to center crop.")
        scale = 0.7
        nh, nw = int(h * scale), int(w * scale)
        y1, x1 = (h - nh) // 2, (w - nw) // 2
        boxes = [[x1, y1, x1 + nw, y1 + nh]]
        scores = [0.0]

    # 풀프레임도 후보에 추가 (항상 하나는 타겟이 되도록)
    full_box = [0, 0, w, h]
    boxes = [full_box] + boxes
    scores = [1.0] + scores

    # NMS로 중복 제거
    boxes, scores = nms(boxes, scores, thr=0.4)
    return boxes, scores

# =========================
# RCNN 실행 + 언랩 + 저장
# =========================
def run_rcnn_on_crops(gray, boxes, model_path, use_cuda=False, save_dir="outputs"):
    """
    gray: CLAHE 적용된 그레이스케일 이미지
    boxes: find_candidates에서 얻은 후보 박스 목록
    model_path: saved_model.pth
    반환:
      best: (score, tread_path, bbox) 또는 None
      all_results: 각 언랩 결과에 대한 리스트
                   (score, tread_path, bbox, metrics_dict)
    """
    import treadscan
    os.makedirs(save_dir, exist_ok=True)
    seg_rcnn = treadscan.SegmentorRCNN(model_path, use_cuda=use_cuda)

    best = None
    all_results = []

    dbg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    unwrap_fail_count = 0
    kps_fail_count = 0

    for i, (x1, y1, x2, y2) in enumerate(boxes, 1):
        crop = gray[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # 업스케일(필요시)
        if UPSCALE_BEFORE_UNWRAP and UPSCALE_BEFORE_UNWRAP != 1.0:
            crop = cv2.resize(
                crop, None,
                fx=UPSCALE_BEFORE_UNWRAP, fy=UPSCALE_BEFORE_UNWRAP,
                interpolation=cv2.INTER_CUBIC
            )

        # RCNN 키포인트
        kps_list = seg_rcnn.find_keypoints(crop)

        # 디버그 박스 색상(초록: 키포인트 있음, 빨강: 없음)
        color = (0, 255, 0) if kps_list else (0, 0, 255)
        cv2.rectangle(dbg, (x1, y1), (x2, y2), color, 2)

        if not kps_list:
            kps_fail_count += 1
            continue

        for j, kps in enumerate(kps_list, 1):
            tire = treadscan.TireModel(crop.shape)
            tire.from_keypoints(*kps)

            # 1️⃣ 기본 unwrap 먼저 시도 (start/end 없이)
            unwrap_cfgs = [
                dict(name="default", params={}),          # 기본각 (가장 안정적)
                dict(name="n20_80", params=dict(start=-20, end=80)),
                dict(name="n30_90", params=dict(start=-30, end=90)),
                dict(name="n40_110", params=dict(start=-40, end=110)),
                dict(name="n50_120", params=dict(start=-50, end=120)),
                dict(name="n60_130", params=dict(start=-60, end=130)),
                dict(name="n80_140", params=dict(start=-80, end=140)),
            ]

            for ui, cfg in enumerate(unwrap_cfgs, 1):
                cfg_name = cfg["name"]
                params = cfg["params"]

                try:
                    # 언랩 시도
                    if params:
                        tread = tire.unwrap(crop, **params)
                    else:
                        tread = tire.unwrap(crop)   # 기본각

                    if tread is None or tread.size == 0:
                        continue

                    # 후처리 표준화
                    if hasattr(treadscan, "remove_gradient"):
                        tread = treadscan.remove_gradient(tread)
                    if hasattr(treadscan, "clahe"):
                        tread = treadscan.clahe(tread)

                    # 표준 폭 리사이즈
                    th, tw = tread.shape[:2]
                    if STD_EXPORT_WIDTH and tw != STD_EXPORT_WIDTH:
                        scale = STD_EXPORT_WIDTH / float(tw)
                        tread = cv2.resize(
                            tread,
                            (STD_EXPORT_WIDTH, int(th * scale)),
                            interpolation=cv2.INTER_CUBIC
                        )

                    # 파일 저장
                    out_name = f"tread_{i:02d}_{j:02d}_{ui:02d}_{cfg_name}.png"
                    out_path = os.path.join(save_dir, out_name)
                    cv2.imwrite(out_path, tread)

                    # ===== 특징/점수 계산 (나중 마모도 코드에서 재활용 가능) =====
                    col_std = tread.std(axis=0)
                    if col_std.size == 0:
                        continue
                    active_cols = (col_std > (col_std.mean() * 0.6)).sum()
                    coverage = active_cols / float(tread.shape[1])  # 0~1, 트레드가 얼마나 꽉 찼는지
                    contrast = float(tread.std())                   # 전체 대비

                    score = 2.0 * coverage + (contrast / 50.0) + 0.25 * len(kps)

                    metrics = dict(
                        coverage=coverage,
                        contrast=contrast,
                        num_keypoints=len(kps),
                        cfg_name=cfg_name,
                        box=(x1, y1, x2, y2)
                    )

                    all_results.append((score, out_path, (x1, y1, x2, y2), metrics))

                    if (best is None) or (score > best[0]):
                        best = (score, out_path, (x1, y1, x2, y2))

                except ValueError as ve:
                    if "cannot unwrap tread out of view" in str(ve):
                        unwrap_fail_count += 1
                        continue
                    else:
                        raise
                except Exception as e:
                    print(f"[WARN] Unwrap failed for box {i}, kps {j}, cfg {cfg_name}: {e}")
                    unwrap_fail_count += 1
                    continue

    # 디버그 이미지 저장
    cv2.imwrite(os.path.join(save_dir, "debug_candidates.jpg"), dbg)

    # 결과 요약 CSV 저장 (팀원 코드에서 이 파일만 읽어도 됨)
    if all_results:
        csv_path = os.path.join(save_dir, "results_summary.csv")
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow([
                    "tread_path", "score",
                    "coverage", "contrast", "num_keypoints",
                    "cfg_name", "box_x1", "box_y1", "box_x2", "box_y2"
                ])
            for (score, path, (bx1, by1, bx2, by2), m) in all_results:
                w.writerow([
                    os.path.basename(path),
                    f"{score:.4f}",
                    f"{m['coverage']:.4f}",
                    f"{m['contrast']:.4f}",
                    m["num_keypoints"],
                    m["cfg_name"],
                    bx1, by1, bx2, by2,
                ])

    if kps_fail_count:
        print(f"[INFO] candidates with NO keypoints: {kps_fail_count}")
    if unwrap_fail_count:
        print(f"[INFO] unwrap failures (out-of-view or errors): {unwrap_fail_count}")

    return best, all_results

# 지피티가 추가하라고 한 분석용 함수

def analyze_car_image(car_image_path: str) -> dict:
    """
    서버에서 바로 호출할 함수.
    입력: 차량 전체 이미지 경로
    출력: {
      "result": {
        "predict_result": "danger" / "safety" / "warning" / None,
        "img_url": CAM_이미지_로컬경로 or None
      }
    }
    """
    # tire_rcnn_pipeline.py 가 있는 app 폴더
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # RCNN 모델 / 출력 폴더 (app 기준)
    rcnn_model_path = os.path.join(script_dir, "models", "RCNN", "saved_model.pth")
    out_dir = os.path.join(script_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # 1) 차량 이미지 로드 + 조명 보정
    img = cv2.imread(car_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {car_image_path}")

    print(f"[INFO] input car image: {car_image_path} shape={img.shape[::-1]}")

    h, w = img.shape
    if (MAX_SIDE is not None) and (max(h, w) > MAX_SIDE):
        s = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        print(f"[INFO] resized image -> {img.shape[::-1]}")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_n = clahe.apply(img)

    try:
        # 2) 타이어 후보 탐색
        boxes, scores = find_candidates(img_n)
        print(f"[INFO] candidate boxes: {len(boxes)}")

        # 3) RCNN 언랩
        best, all_results = run_rcnn_on_crops(
            img_n,
            boxes,
            rcnn_model_path,
            use_cuda=USE_CUDA,
            save_dir=out_dir
        )

        print(f"[INFO] total unwrap results: {len(all_results)}")

        if not best:
            print("[WARN] RCNN failed on all candidates.")
            return {
                "result": {
                    "predict_result": None,
                    "img_url": None
                }
            }

        best_score, best_tread_path, best_box = best
        print(f"[BEST] score={best_score:.3f} file={os.path.basename(best_tread_path)}")

        # 4) 언랩된 best 트레드 이미지로 팀원 모델 돌리기
        status, prob, cam_path = classify_tread_image(
            best_tread_path,
            model_path=os.path.join(script_dir, "models"),               # app/models
            model_name="model_best_weights.pth",
            img_size=224,
            result_path=os.path.join(script_dir, "saved", "cam_result"), # app/saved/cam_result
            temp_image_dir=os.path.join(script_dir, "temp_image"),       # app/temp_image
            temp_mask_dir=os.path.join(script_dir, "temp_mask"),         # app/temp_mask
        )

        print(f"[FINAL] status={status}, prob={prob:.3f}")
        print(f"[FINAL] cam_image={cam_path}")

        # 팀원이 원하는 JSON 형태
        return {
            "result": {
                "predict_result": status,
                "img_url": cam_path
            }
        }

    except Exception as e:
        print("[ERROR] Exception while running pipeline")
        print(e)
        traceback.print_exc()
        return {
            "result": {
                "predict_result": None,
                "img_url": None
            }
        }

# =========================
# 메인 : 로컬 테스트용으로 변경함
# =========================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))  # app 폴더
    img_path = os.path.join(script_dir, "assets", IMG_NAME)  # app/assets/IMG_NAME

    response = analyze_car_image(img_path)
    print("FINAL_RESPONSE_JSON:")
    print(json.dumps(response, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
