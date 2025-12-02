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

# IOU(intersection over union)함수 : 두 박스가 얼마나 겹치는지 비율
# 입력 a,b는 [x1, y1, x2, y2] 형태의 박스. (x1, y1): 왼쪽 위 좌표 (top-left), (x2, y2): 오른쪽 아래 좌표 (bottom-right)
# 출력은 두 박스가 얼마나 겹치는지 0~1 사이 값을 출력.

def iou(a, b):
    ax1, ay1, ax2, ay2 = a # a 박스의 네 좌표: ax1, ay1, ax2, ay2
    bx1, by1, bx2, by2 = b # b 박스의 네 좌표 → bx1, by1, bx2, by2
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1) # 교집합 영역의 좌표 계산, 겹치는 부분은 두 박스가 “공통으로 포함하는 구간”인데, 시작점은 둘 중 더 늦게 시작하는 지점인 max이고
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2) # 끝점은 둘 중 더 먼저 끝나는 지점인 min 이다.
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1) # 교집합의 가로, 세로 길이, 만약 안겹치는 경우 길이가 음수되니까, 0으로 처리, 면적도 0.
    inter = iw * ih # 두 박스가 실제로 겹치는 면적 값
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter + 1e-6 # 합집합 공식 적용. 아주작은 값인 EPS(1e-6) 더함. 만약 두 박스가 모두 면적이 0이거나, 둘이 완전히 겹쳐서 수치적으로 오차가 생기면 union이 0이 되어 0으로 나누기 에러가 날 수 있음. 이를 방지하기 위한 수치 안정성용 작은 상수.
    return inter / union

# NMS(Non-Maximum Suppression) 함수
# 서로 겹치는 후보 박스들 중에서 가장 점수 높은 것만 남기고, 나머지는 제거하는 알고리즘
# NMS 점수가 높은 박스부터 하나씩 고르고, IOU가 thr보다 큰(즉, 겹치는) 박스들은 삭제해서 중복 후보 제거.
# NMS를 거친 뒤에는 서로 과도하게 겹치는 박스가 없고 대체로 각 타이어/휠당 가장 좋은 박스 하나씩만 남게 됨.

def nms(boxes, scores, thr=0.4): # boxes: 후보 박스 리스트, scores: 각 박스의 스코어 (타원의 원형 비율, 엣지 강도 등), “이 박스가 진짜 타이어일 것 같다” 라는 정도. thr: 같은 타이어를 가리키는 중복 박스라고 보는 기준값.
    idxs = np.argsort(scores)[::-1] # 점수를 오름차순으로 정렬한걸 뒤집어서 내림차순 인덱스로 만든다.
    keep = [] # 최종적으로 NMS를 통과해서 살려둘 박스들의 인덱스를 넣기 위한 리스트.
    while len(idxs):
        i = idxs[0] # idxs[0] → 현재 남아 있는 후보들 중 가장 점수 높은 박스의 인덱스.
        keep.append(i) # 이건 무조건 살려야 하니까 keep에 추가.
        rest = idxs[1:] # 이번에 선택한 최고 박스 i를 제외한 후보들.
        remain = [] # 이번 턴에서 살아남아서 다음 while 루프까지 가져갈 애들을 담을 리스트.
        for j in rest:
            if iou(boxes[i], boxes[j]) <= thr: # 현재 선택된 최고점 박스 i와 후보 j가 얼마나 겹치는지 계산하기 위해 iou에 넣음. 겹침 정도가 임계값 이하라면,
                remain.append(j) # 서로 다른 물체일 가능성이 충분히 있다라고 보고 살려둔다. 그 외에는 겹친다고 판단하고 버림.
        idxs = np.array(remain) # 이제 다음 while 턴에는 remain만 후보로 다시 돌아감. 이 과정을 후보가 다 떨어질 때까지 반복.
    return [boxes[i] for i in keep], [scores[i] for i in keep] # keep에 들어있는 인덱스로 boxes, scores를 다시 구성해서 리턴.

# =========================
# 후보 ROI 탐색 (타원 + 허프서클 → NMS)
# =========================

# find_candidates(gray) 함수
# 입력은 CLAHE까지 적용된 그레이스케일 이미지(나중에 analyze_car_image에서 넘김).
# 최종적으로 타이어가 있을 법한 박스 후보들을 찾는 함수.
# CLAHE 등으로 전처리된 흑백 차량 이미지에서 타이어/휠이 있을 법한 후보 박스(ROI)들을 찾아서 boxes(좌표 리스트)와 scores(간단 점수 리스트)로 돌려주는 함수.
# 이 boxes가 바로 다음 단계 run_rcnn_on_crops(...)로 넘어가서 각 후보에 대해 RCNN 언랩을 시도하게 됨.
# 점수를 주는 기준은 A. 얼마나 원에 가까우냐, B. 얼마나 엣지가 많이 검출되냐(타이어 부분일수록 트레드 패턴때문에 경계면이 많이 나타나니까) C.  가장 가능성 낮으니까 점수 가장 낮게 줌

def find_candidates(gray):
    """
    입력: gray (uint8, 단일 채널)
    반환: (boxes, scores)
      - boxes: [x1,y1,x2,y2] 리스트
      - scores: 각 후보의 간단 점수(크게 의미두진 않아도 됨)
    """
    h, w = gray.shape # h: 이미지 높이(height), w: 이미지 너비(width) , 이후 박스 좌표를 이미지 범위 안으로 자를 때(0~w, 0~h) 쓰려고 바로 꺼내둔 것.
    boxes, scores = [], [] # boxes: ROI 후보들을 담는 리스트, scores: 각 박스에 대한 점수(이 후보가 타이어일 가능성 점수)를 담는 리스트. 두 리스트는 같은 인덱스끼리 짝이 됨.

    # A) treadscan의 타원 검출 (휠 전체가 잘 보이는 경우에 강함), 왜 타원을 찾나? 자동차 휠(림)을 측면에서 보면 거의 타원 모양이라 이걸 찾으려 함.
    try:
        import treadscan
        seg = treadscan.Segmentor(gray) #휠 전체의 타원을 찾는 모듈, 이 타원을 중심으로 주변을 잘라내면 타이어가 잘 포함된 박스를 만들 수 있음
        ell = seg.find_ellipse(threshold=135, min_area=0) # 휠 타원을 찾는 함수인 find_ellipse로 타원 파라미터 (center, axes, angle) 얻음. threshold: 내부에서 엣지 검출/이진화 시 밝기 기준으로 135값을 쓴다, min_area: 최소 면적 제한 없음. 작은 타원도 일단 허용.
        if ell is not None: # 타원 검출에 성공했을 때만 박스를 만든다.
            (xc, yc), (MA, ma), ang = ell # OpenCV fitEllipse 스타일의 반환값과 비슷한 형태, (xc, yc): 타원 중심 좌표, (MA, ma): 세로/가로 축 길이 (Major Axis, minor axis), ang: 타원의 회전 각도(사용은 안 하지만, 언랩 단계에서는 중요해질 수 있는 정보)
            r = max(MA, ma) * 0.55 * CROP_PAD # r: 타이어가 충분히 들어올 정도로 휠 주변을 감싼 “박스 반경” 역할. max(MA, ma): 타원 축 중 더 긴 쪽으로, 휠의 대략적인 크기. 휠 중심으로부터 타이어 영역까지가 휠 전체보다 조금 좁기 때문에 림과 일부 타이어 정도만 포함하도록 약간 줄이는 계수인 0.55 곱함. # CROP_PAD(1.75 같은 값)를 곱해서 여유를 조금 더 준다.
            x1, y1 = int(max(0, xc - r)), int(max(0, yc - r)) # 박스 좌표 계산 함. 이미지 범위를 벗어나지 않게 클리핑하는 작업.
            x2, y2 = int(min(w, xc + r)), int(min(h, yc + r))
            if (x2 - x1) > 30 and (y2 - y1) > 30: # 노이즈성 작은 타원 검출이나 이상한 상황을 방지하기 위한 간단한 필터. 가로/세로 길이가 둘 다 30픽셀 이상일 때만 후보로 인정. 너무 작은 박스 걸러내기.
                boxes.append([x1, y1, x2, y2]) # 타원 기반 후보 박스 하나 추가.
                circ = float(min(MA, ma) / (max(MA, ma) + 1e-6))  # 단축/장축 비율이 1에 가까울수록 정원에 가깝다는 의미. 휠이 정면에 가까울수록 원형에 가까울 테니, 그걸 점수로 사용. 1e-6: 아까 설명한 0으로 나누기 방지.
                scores.append(circ) #scores는 “얼마나 원에 가까운지” (circ = 단축/장축 비율)로 간단히 점수 부여.
    except Exception:
        # treadscan import 안되거나, Segmentor 내부에서 에러 나도, 전체 파이프라인이 멈추지 않게 하려고 예외를 무시, 타원 검출이 안 되면 뒤에 허프 서클, 폴백 등을 믿고 밀고간다 라는 설계
        pass

    # B) 허프 서클 (림이 더 뚜렷한 경우에 강함)
    g = cv2.GaussianBlur(gray, (5, 5), 0) # 허프변환은 노이즈에 민감하므로, 먼저 가우시안 블러로 노이즈를 부드럽게 만들어서 안정적인 원 검출을 유도. (5,5) 커널, sigmaX=0은 OpenCV가 적당한 σ를 자동 계산.
    circles = cv2.HoughCircles(
        g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min(h, w)//4, # g: 입력 이미지(블러된 그레이스케일). cv2.HOUGH_GRADIENT: 일반적인 그래디언트 기반 허프 원 검출 방식. dp=1.2: accumulator 해상도. 약간 줄어든 해상도로 연산해서 속도/안정성 트레이드오프. minDist=min(h, w)//4: 검출된 원들끼리 최소 거리. 한 이미지에 원(휠)이 여러 개 있어도 너무 가까이 겹쳐 있지 않게, 중복 검출을 줄이려는 튜닝값.
        param1=120, param2=40, minRadius=40, maxRadius=0 # param1=120, param2=40: 내부적으로 Canny 엣지 또는 허프 누산기 threshold로 사용됨. 120는 Canny 상한 threshold. 40는 원으로 인정할 허프 accumulator threshold. minRadius=40: 반지름 최소값 40픽셀 → 너무 작은 동그라미(노이즈) 무시. maxRadius=0: 상한 없음이니 자동으로 설정.
    )
    if circles is not None: # HoughCircles가 원을 하나 이상 찾았을 때만 루프.
        for c in circles[0, :]:
            cx, cy, r = int(c[0]), int(c[1]), int(c[2]) # 각각 원 중심과 반지름을 정수로 변환.
            r = int(r * CROP_PAD) # 마찬가지로 여유를 주기 위해 pad 배수(CROP_PAD) 곱함. 타이어 전체 영역을 포함하게 반지름을 키우는 효과.
            x1, y1 = max(0, cx - r), max(0, cy - r) # 원 중심+-반지름으로 사각형 박스 만듬. max(0, ...), min(w/h, ...)로 이미지 범위 제한.
            x2, y2 = min(w, cx + r), min(h, cy + r)
            if (x2 - x1) > 30 and (y2 - y1) > 30: # 크기가 30×30보다 작은 후보는 버린다 (노이즈 후보 제거).
                boxes.append([x1, y1, x2, y2]) # 허프 서클 기반 후보 박스 추가.
                roi = gray[y1:y2, x1:x2] # 해당 후보 영역을 잘라서 ROI로 사용.
                edges = cv2.Canny(roi, 80, 160) # Canny 엣지 검출로, 이 영역에 선/엣지가 얼마나 많은지 확인. 휠/타이어 부분은 보통 스포크, 패턴 때문에 엣지(경계선/패턴)가 풍부하다고 가정. 엣지가 많은 후보 박스 -> “휠/타이어일 확률이 좀 더 높다”
                scores.append(float(edges.mean()) / 255.0) #  엣지 영상의 평균 밝기를 계산한다. 이는 0~255 사이인데 나누기 255.0으로 0~1 사이로 정규화. 엣지 많을수록 평균이 높아지므로, 엣지가 많은 후보일수록 scores 높게 설정.

    # C) 후보가 하나도 없으면 중앙 크롭 폴백(최후의 최후), 최악의 상황(타원도, 원도 못 찾는 경우)에도 RCNN이 어딘가에서는 타이어를 찾아낼 기회를 주자는 취지.
    if not boxes:
        print("[INFO] no ellipse/circle found, fallback to center crop.") # 어떤 방법으로도 후보가 안 나오면 이미지 중앙 70% 정도를 강제로 후보로 추가.
        scale = 0.7
        nh, nw = int(h * scale), int(w * scale) # 새 박스의 높이·너비.
        y1, x1 = (h - nh) // 2, (w - nw) // 2 # 이미지 중앙에 위치하도록 시작점을 잡는 계산: 전체에서 nh만큼 빼고 반 나누면 중앙 정렬.
        boxes = [[x1, y1, x1 + nw, y1 + nh]] # 중앙 박스 하나만 후보로 강제 생성.
        scores = [0.0] # 뒤에서 full_box에 1.0을 줄 거라 상대적으로 낮게 설정

    # 풀프레임도 후보에 추가. 어떤 기법도 제대로 후보를 못 잡았다고 하더라도, 이미지 전체를 넣고 RCNN이 알아서 찾도록 하는 fallback을 항상 확보.
    full_box = [0, 0, w, h] # 이미지 전체 영역.
    boxes = [full_box] + boxes # 맨 앞에 풀프레임을 추가.
    scores = [1.0] + scores # 풀프레임의 스코어를 1.0으로 둔다. 앞에서 타원/허프 후보에 준 점수들이 0~1 사이였으니 풀프레임이 꽤 높은 점수를 가지도록 했다.

    # NMS로 중복 제거, 앞에서 타원, 허프 서클, 중앙 fallback 으로 생성 된거 상당수는 서로 많이 겹치는 박스일 수 있기 때문에 nms사용, IOU 0.4 이상 겹치는 박스들 중에서는 가장 점수 높은 것만 남기고 제거.
    boxes, scores = nms(boxes, scores, thr=0.4)
    return boxes, scores

# =========================
# RCNN 실행 + 언랩 + 저장
# =========================

# run_rcnn_on_crops 함수
# CLAHE까지 끝난 그레이스케일 이미지 + 타이어 후보 박스들을 받아서 각 후보별로 RCNN으로 키포인트를 찾고 여러 각도로 unwrap 시도한다.
# 언랩된 트레드 이미지들을 저장하고 점수를 매겨서 가장 좋은 하나(best)와 전체 리스트(all_results)를 돌려주는 함수.
# 이렇게 만들어진 “베스트 트레드 이미지 경로”가 classify_tread_image로 해서 tire_classifier.py로 넘어감.

def run_rcnn_on_crops(gray, boxes, model_path, use_cuda=False, save_dir="outputs"):
    """
    gray: CLAHE 적용된 그레이스케일 이미지
    boxes: find_candidates에서 얻은 후보 박스 목록
    model_path: saved_model.pth
    use_cuda=False: GPU 사용할지 여부.
    save_dir="outputs": 언랩된 트레드 이미지, 디버그 이미지, CSV 등을 저장할 디렉토리.
    반환:
      best: (score, tread_path, bbox) 또는 None
      all_results: 각 언랩 결과에 대한 리스트
                   (score, tread_path, bbox, metrics_dict)
    """
    import treadscan
    os.makedirs(save_dir, exist_ok=True) # 출력 폴더가 없다면 생성. exist_ok=True로 이미 폴더 있어도 에러 없이 넘어감.
    seg_rcnn = treadscan.SegmentorRCNN(model_path, use_cuda=use_cuda) # RCNN 모델 로드. RCNN 기반 키포인트 검출기 객체 생성. model_path: 학습된 가중치 로드. use_cuda: GPU 사용 여부 전달.

    best = None # 지금까지 나온 언랩 결과들 중 가장 좋은 하나를 저장할 변수. 형식: (score, tread_path, bbox)
    all_results = [] # 모든 성공한 언랩 결과를 저장하는 리스트. 각 원소: (score, tread_path, bbox, metrics_dict).

    # 디버그 이미지/카운터 준비 부분
    dbg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) # 디버그용 컬러 이미지 dbg에 후보 박스 색깔 입혀서 저장도 함.
    unwrap_fail_count = 0 # 언랩 도중 실패(“화면 밖이라 펴기 불가” 등)한 횟수 카운트.
    kps_fail_count = 0 # RCNN이 키포인트를 하나도 못 찾은 후보 박스 개수 카운트.

    # 후보 박스별로 RCNN, 언랩 시도
    for i, (x1, y1, x2, y2) in enumerate(boxes, 1): # boxes에 들어있는 모든 후보 박스를 하나씩 꺼내 처리. i는 1부터 시작하는 인덱스로, 파일 이름에 넣을 때 사용함.
        crop = gray[y1:y2, x1:x2] # 원본 그레이 이미지에서 해당 박스 영역만 잘라낸 부분. crop이 RCNN에 들어가는 입력 이미지.
        if crop.size == 0: # 혹시라도 좌표가 이상해서 잘라낸 영역이 비어 있으면(너비나 높이가 0 이거나 하면) 그냥 스킵
            continue

        # 업스케일(후보 박스 이미지를 필요한 경우에 크게 키워서 RCNN이 보기 편하게 만들어 줌)
        if UPSCALE_BEFORE_UNWRAP and UPSCALE_BEFORE_UNWRAP != 1.0: # UPSCALE_BEFORE_UNWRAP: 전역 튜닝 파라미터. 타이어가 사진 안에서 작게 찍혔을 때, 업스케일해서 디테일을 살리기 위함
            crop = cv2.resize( # 비율 기반 리사이즈.
                crop, None,
                fx=UPSCALE_BEFORE_UNWRAP, fy=UPSCALE_BEFORE_UNWRAP,
                interpolation=cv2.INTER_CUBIC # INTER_CUBIC은 확대 시 부드러운 결과를 주는 보간법.
            )

        # RCNN 키포인트 검출, 디버그 박스 색칠
        kps_list = seg_rcnn.find_keypoints(crop) # RCNN이 찾아낸 키포인트 세트 목록. 각 kps에는 트레드 언랩에 필요한 점들(림/타이어 경계 등)이 들어 있다고 보면 됨.

        # 디버그 박스 색상(초록: 키포인트 있음, 빨강: 없음)
        color = (0, 255, 0) if kps_list else (0, 0, 255) # 키포인트를 하나라도 찾았다면 초록색 박스. 아무 것도 못 찾았다면 빨간색 박스.
        cv2.rectangle(dbg, (x1, y1), (x2, y2), color, 2) # 디버그 이미지(dbg) 위에 해당 후보 박스를 그려 넣는다. 나중에 debug_candidates.jpg로 저장해서 “어떤 후보에서 키포인트가 잘 잡혔는지” 눈으로 확인할 수 있음.

        if not kps_list: # 키포인트가 하나도 없으면 이 후보는 언랩 시도 자체가 불가능하므로: kps_fail_count 올리고 continue로 다음 후보 박스로 넘어감.
            kps_fail_count += 1
            continue

        for j, kps in enumerate(kps_list, 1): # 한 박스 내에서도 여러 개의 키포인트 세트가 나올 수 있으므로, 각 세트별로 언랩을 시도.
            tire = treadscan.TireModel(crop.shape) # 현재 crop의 해상도(이미지 크기)를 기반으로 한 타이어 모델 객체 생성. 내부적으로 타이어의 위치, 반지름, 방향 등을 표현할 구조.
            tire.from_keypoints(*kps) # RCNN이 뽑은 키포인트(kps)를 타이어 모델에 주입. 이걸 통해 타이어가 이미지 안에서 어떻게 놓여 있는지를 모델이 알게 됨. 이후 tire.unwrap()에서 이 정보를 이용해 트레드를 펼침.

            # 기본 unwrap 먼저 시도 (start/end 없이)
            unwrap_cfgs = [ # unwrap_cfgs: 언랩할 때 사용할 여러가지 start/end 각도 조합 리스트. name: 나중에 CSV·파일명에 기록하기 위한 라벨. params: 실제 언랩 함수에 넘길 인자 집합.
                dict(name="default", params={}),          # 기본각 (가장 안정적)
                dict(name="n20_80", params=dict(start=-20, end=80)), # 이 아래는 타이어가 기울어져 있거나 부분적으로만 보이는 상황을 위한 각도 범위 튜닝 값들. start=-20, end=80 이런 식으로 이 각도 구간만 펴줘라는 요청.
                dict(name="n30_90", params=dict(start=-30, end=90)),
                dict(name="n40_110", params=dict(start=-40, end=110)),
                dict(name="n50_120", params=dict(start=-50, end=120)),
                dict(name="n60_130", params=dict(start=-60, end=130)),
                dict(name="n80_140", params=dict(start=-80, end=140)),
            ]
            # unwrap 시도와 후처리
            for ui, cfg in enumerate(unwrap_cfgs, 1): # 각 언랩 설정마다 실제 tire.unwrap를 호출.
                cfg_name = cfg["name"] # 이름/파라미터를 각각 꺼내서 사용.
                params = cfg["params"]

                try:
                    # 언랩 시도
                    if params: # 비어 있지 않으면
                        tread = tire.unwrap(crop, **params)
                    else: # 비어 있으면
                        tread = tire.unwrap(crop)   # 기본각

                    if tread is None or tread.size == 0: # 언랩이 실패했거나, 결과가 빈 이미지인 경우 이 시도는 버리고 다음 설정으로.
                        continue

                    # 후처리 표준화
                    if hasattr(treadscan, "remove_gradient"): #treadscan 버전에 따라 hasattr()가 없을 수도 있으니 있는 경우에만 호출해서 호환성을 유지하려고 if 씀.
                        tread = treadscan.remove_gradient(tread) # remove_gradient:  트레드 이미지에 있는 밝기 기울기(조명 변화)를 제거해서 패턴만 더 잘 보이게 하는 함수(있을 수도 있고, 없을 수도 있는 옵션 함수).
                    if hasattr(treadscan, "clahe"):
                        tread = treadscan.clahe(tread) # clahe: 트레드 이미지에 CLAHE를 다시 적용해서 대비를 올리는 후처리.

                    # 표준 폭 리사이즈
                    th, tw = tread.shape[:2] # th: 언랩된 트레드 이미지 높이. tw: 언랩된 트레드 이미지 너비.
                    if STD_EXPORT_WIDTH and tw != STD_EXPORT_WIDTH: # 상단에 정의된 STD_EXPORT_WIDTH (예: 256)가 있고, 현재 폭 tw가 그 값과 다르면, 폭을 기준으로 리사이즈.
                        scale = STD_EXPORT_WIDTH / float(tw) # 가로 길이를 STD_EXPORT_WIDTH로 맞출 비율.
                        tread = cv2.resize( # 가로는 고정, 세로는 비율에 맞게 줄이거나 늘림. 이후 분류 모델에서도 손쉽게 224/256 등의 입력 크기로 재가공하기 쉬워짐.
                            tread,
                            (STD_EXPORT_WIDTH, int(th * scale)),
                            interpolation=cv2.INTER_CUBIC
                        )

                    # 파일 저장
                    out_name = f"tread_{i:02d}_{j:02d}_{ui:02d}_{cfg_name}.png" # tread_후보인덱스_키포인트세트인덱스_언랩설정인덱스_설정이름.png
                    out_path = os.path.join(save_dir, out_name) # save_dir 안에 이 파일 이름으로 저장.
                    cv2.imwrite(out_path, tread) # 언랩 결과를 PNG 파일로 디스크에 기록.

                    # ===== 특징/점수 계산 (나중 마모도 코드에서 재활용 가능) =====
                    col_std = tread.std(axis=0) # column별 표준편차. 한 열 안에 밝기가 많이 변하면 패턴이 있다(트레드가 보인다)는 의미. 반대로 평평한 회색이면 표준편차가 작다.
                    if col_std.size == 0:
                        continue
                    active_cols = (col_std > (col_std.mean() * 0.6)).sum() # active_cols: 적당히 세게 패턴이 보이는 열의 수. col_std.mean(): 전체 열 표준편차의 평균.  col_std > (평균 * 0.6): 평균의 60% 이상 되는 열들을 활성된 열이라고 잡는다. .sum(): 이런 활성 열의 개수.
                    coverage = active_cols / float(tread.shape[1])  # tread.shape[1] : 전체 열 개수(=언랩 결과의 가로 픽셀 수).coverage 0~1, 1에 가까울수록 전체 가로폭에 걸쳐 패턴이 고르게 잘 보임. 0에 가까울수록 패턴이 일부만 보이거나 거의 없음.
                    contrast = float(tread.std())                   # 전체 이미지의 표준편차. 값이 클수록 전반적인 대비(밝기 변화)가 크다. 너무 낮으면 전체가 회색톤으로 뭉개진 상태일 가능성이 크다.

                    score = 2.0 * coverage + (contrast / 50.0) + 0.25 * len(kps) # 2.0 * coverage: 트레드가 가로로 얼마나 꽉 차 있는지에 가장 큰 비중. contrast / 50.0: 전체 대비도 점수에 조금 더 보탬. 0.25 * len(kps): 키포인트가 많을수록 모델이 이 영역을 타이어라고 강하게 인식했다는 뜻이니, 보너스 점수. (2.0, 50.0, 0.25)값은 결과 보고 내가 임의로 튜닝 해도 됨.

                    metrics = dict( # 나중에 CSV에 기록하기 좋은 형태로 숫자들을 모아둔 딕셔너리.
                        coverage=coverage,
                        contrast=contrast,
                        num_keypoints=len(kps),
                        cfg_name=cfg_name,
                        box=(x1, y1, x2, y2)
                    )

                    all_results.append((score, out_path, (x1, y1, x2, y2), metrics)) # 모든 성공한 언랩 결과들을 리스트에 저장 (score, 이미지경로, 박스좌표, metrics_dict)

                    if (best is None) or (score > best[0]): # 아직 best가 없거나, 현재 점수가 더 크면 갱신. best는 가장 높은 점수 하나만 유지.
                        best = (score, out_path, (x1, y1, x2, y2))

                #예외 처리(언랩 실패)
                except ValueError as ve:
                    if "cannot unwrap tread out of view" in str(ve): # 이건 타이어가 이미지 밖으로 너무 나가서 펴기 어렵다같은 예상 가능한 실패 상황.
                        unwrap_fail_count += 1
                        continue
                    else:
                        raise  #그 외 ValueError는 raise로 다시 던짐. 진짜 이상한 상황(버그)일 수도 있으니 숨기지 않음.
                except Exception as e:
                    print(f"[WARN] Unwrap failed for box {i}, kps {j}, cfg {cfg_name}: {e}") # 예상 외의 에러는 경고 메시지 출력 후, 역시 unwrap_fail_count 올리고 다음 시도로 넘어감.
                    unwrap_fail_count += 1
                    continue

    # 디버그 이미지 저장. 앞에서 초록/빨강 박스를 그려 넣었던 dbg 이미지를 파일로 저장.
    cv2.imwrite(os.path.join(save_dir, "debug_candidates.jpg"), dbg)

    # 결과 요약 CSV 저장 (팀원 코드에서 이 파일만 읽어도 되는 부분)
    if all_results: # 하나라도 언랩에 성공했을 때만 CSV를 쓴다.
        csv_path = os.path.join(save_dir, "results_summary.csv")
        write_header = not os.path.exists(csv_path) # 파일이 없으면(처음 쓸 때만) 헤더 행을 추가.
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow([ # 컬럼 이름들
                    "tread_path", "score",
                    "coverage", "contrast", "num_keypoints",
                    "cfg_name", "box_x1", "box_y1", "box_x2", "box_y2"
                ])
            for (score, path, (bx1, by1, bx2, by2), m) in all_results: # 앞에서 쌓아둔 결과들을 줄 단위로 CSV에 기록.
                w.writerow([
                    os.path.basename(path),
                    f"{score:.4f}",
                    f"{m['coverage']:.4f}",
                    f"{m['contrast']:.4f}",
                    m["num_keypoints"],
                    m["cfg_name"],
                    bx1, by1, bx2, by2,
                ])
    # 실패 카운트 로그, 반환. 키포인트 실패/언랩 실패 횟수를 출력해서 디버깅에 도움.
    if kps_fail_count:
        print(f"[INFO] candidates with NO keypoints: {kps_fail_count}")
    if unwrap_fail_count:
        print(f"[INFO] unwrap failures (out-of-view or errors): {unwrap_fail_count}")

    return best, all_results # best: (score, best_tread_path, best_bbox) 또는 언랩 전부 실패하면 None. all_results: 모든 성공한 언랩 결과 리스트.

# analyze_car_image 함수 (얘가 중요. 엔트리 함수임)
# 차량 전체 사진 1장을 입력으로 받아서 2)타이어 후보 찾기 → 3)RCNN + 언랩 → 4)베스트 트레드 선택 후 분류 모델 호출까지를 한 번에 수행하고, 최종 JSON 형식 결과만 돌려주는 엔트리 함수.
# FastAPI에서 이 함수만 호출하면, 바깥에서는 이미지 경로 ↔ JSON 결과만 신경 쓰면 되게 설계한 것.
def analyze_car_image(car_image_path: str) -> dict:
    """
    서버에서 바로 호출할 함수.
    car_image_path: str : 차량 전체가 찍힌 이미지의 파일 경로 (문자열). FastAPI 쪽에서는 업로드 받은 이미지를 서버의 특정 위치에 저장한 뒤, 그 저장 경로를 이 함수에 넘겨주게 될 것.
    -> dict :  반환 타입을 명시함. 최종적으로 파이썬 딕셔너리를 돌려준다. FastAPI는 이 딕셔너리를 JSON 응답으로 자동 변환 가능.

    입력: 차량 전체 이미지 경로
    출력: {
      "result": {
        "predict_result": "danger" / "safety" / "warning" / None, (predict_result: 팀원의 분류 모델의 최종 판단 결과.)
        "img_url": CAM_이미지_로컬경로 or None (img_url: CAM(heatmap) 결과 이미지의 파일 경로.)
      }
    }

    실제 HTTP로 이미지를 보내는 건 FastAPI 몫이고, 이 함수는 “어디에 저장됐는지 경로만 알려주는 역할”까지 담당.
    """
    # tire_rcnn_pipeline.py 가 있는 app 폴더
    script_dir = os.path.dirname(os.path.abspath(__file__)) # 현재 이 파이썬 파일(tire_rcnn_pipeline.py)의 경로를 절대 경로로 변환, 그 파일이 들어 있는 폴더 경로를 script_dir에 넣음. 즉, script_dir는 app 폴더 경로. 나중에 모델 파일, 출력 폴더, saved 폴더 등을 파일 위치 기준으로 상대 경로로 관리하고 싶어서.
    #어디서 실행하든(프로세스의 현재 작업 디렉토리가 어디든) tire_rcnn_pipeline.py가 있는 폴더를 기준으로 찾으면 경로가 꼬이지 않음.

    # RCNN 모델 / 출력 폴더 (app 기준)
    rcnn_model_path = os.path.join(script_dir, "models", "RCNN", "saved_model.pth")
    out_dir = os.path.join(script_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True) # outputs 폴더가 없다면 생성. exist_ok=True 때문에 이미 있어도 에러 없이 그냥 넘어감.

    # 1) 차량 이미지 로드, 조명 보정
    img = cv2.imread(car_image_path, cv2.IMREAD_GRAYSCALE) # 입력 이미지를 그레이스케일로 읽는다. 전체적으로 타이어 위치 찾기 / 엣지 / 명암 대비에 집중할 거라 1채널 처리로 단순화.
    if img is None:  # 경로가 잘못됐거나 파일이 손상된 경우 None.
        raise FileNotFoundError(f"Image not found: {car_image_path}") # 이때는 FileNotFoundError를 던져서 명확히 에러를 띄움. 이 예외는 아래의 try/except에서 다시 잡아서 JSON으로 정리된 실패 응답을 내보내게 됨.

    print(f"[INFO] input car image: {car_image_path} shape={img.shape[::-1]}") # img.shape[::-1] : (height, width) 형태인 shape를 뒤집어서 (width, height)로 출력하는 편의 표현.

    h, w = img.shape # 이미지 크기 확인.
    if (MAX_SIDE is not None) and (max(h, w) > MAX_SIDE): # 전역 파라미터 MAX_SIDE가 설정되어 있고, 이미지의 긴 변이 그 값보다 클 경우에만 리사이즈. 너무 큰 이미지는 처리 시간이 길어지니, 긴 변 기준으로 줄이는 옵션.
        s = MAX_SIDE / max(h, w) # 축소 비율.
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) # 줄일 때는 샘플링 특성상 INTER_AREA가 좋다고 판단함.
        print(f"[INFO] resized image -> {img.shape[::-1]}")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # CLAHE 객체 생성. 일반적으로 쓰는 튜닝 값인 2.0 과 (8,8) 넣음.
    img_n = clahe.apply(img) # CLAHE 적용된 정규화 이미지. 이후 모든 후보 탐색/RCNN 언랩은 이 img_n 기준으로 진행된다.

    try:
        # 2) 타이어 후보 탐색
        boxes, scores = find_candidates(img_n) # find_candidates() 사용
        print(f"[INFO] candidate boxes: {len(boxes)}") # 후보가 몇 개 나왔는지 콘솔에 Logging.

        # 3) RCNN 언랩
        best, all_results = run_rcnn_on_crops( # run_rcnn_on_crops() 사용
            img_n,
            boxes,
            rcnn_model_path,
            use_cuda=USE_CUDA,
            save_dir=out_dir
        )

        print(f"[INFO] total unwrap results: {len(all_results)}") # 총 몇 개의 언랩이 실제로 성공했는지 확인.

        if not best: #모든 후보에 대해 키포인트 없음/언랩 실패 등으로 아무 결과도 얻지 못했다는 뜻. 이 경우, 분류 모델을 돌릴 트레드 이미지가 없다는 건 더 진행 불가하다는 것이니 미리 정한 형식대로 아래를 반환하고 함수 종료.
            print("[WARN] RCNN failed on all candidates.")
            return {
                "result": {
                    "predict_result": None,
                    "img_url": None
                }
            }

        best_score, best_tread_path, best_box = best # best 튜플을 각각 변수로 풀어놓음. best_score: 점수(coverage/contrast/키포인트 수 조합). best_tread_path: 언랩된 트레드 이미지 파일 경로. best_box: 그 트레드가 나온 원래 ROI 박스 좌표.
        print(f"[BEST] score={best_score:.3f} file={os.path.basename(best_tread_path)}") # os.path.basename(best_tread_path): 경로에서 파일명만 뽑아서 로그로 보기 좋게 출력.

        # 여기까지, 차량 전체 이미지에서 뽑아낸 타이어/트레드 중 가장 질이 좋은 언랩 결과 1장을 best_tread_path로 확보한 상태.

        # 4) 언랩된 best 트레드 이미지로 팀원의 모델 돌림, 즉 classify_tread_image()는 팀원 코드에서 가져온 분류 함수. 대략적인 역할은 best_tread_path 이미지를 불러서 내부 CNN/분류 모델에 넣고 마모 상태(danger/safety/warning)와 CAM 이미지를 생성해서 저장. 결과로 (status, prob, cam_path)를 리턴하는 함수.
        status, prob, cam_path = classify_tread_image( # status: "danger", "warning", "safety" 같은 문자열 상태. prob: 그 상태에 대한 확률/신뢰도 (float값). cam_path: CAM 결과 이미지 파일 경로.
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

        # 팀원이 원하는 JSON 형태로 반환
        return {
            "result": {
                "predict_result": status,
                "img_url": cam_path
            }
        }
    # 예외 처리 부분. 위의 모든 과정(후보 탐색, RCNN, 분류 모델) 중 어느 한 곳에서라도 예외 발생 시, [ERROR] 로그를 출력, traceback.print_exc()로 스택 트레이스 출력 (디버깅용), 실패용 JSON 반환.
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

    response = analyze_car_image(img_path) # 엔트리 함수인 analyze_car_image 함수 실행후 결과 받음.
    print("FINAL_RESPONSE_JSON:")
    print(json.dumps(response, ensure_ascii=False, indent=2)) #json으로 덤프.

if __name__ == "__main__":
    main()
