# service/yolo_service.py

import os
import cv2
import numpy as np
from ultralytics import YOLO

# ————————————————
# 1) LOAD YOLO ONCE
# ————————————————
YO = YOLO("C:/Users/MBQ/Desktop/ML Projet Test/App Projet ML/models/best.pt")

# ————————————————
# 2) CITYSCAPES-STYLE CLASSES & COLORS
#   (must match your data.yaml names order)
# ————————————————
CLASS_NAMES = [
    'static',
    'road_sidewalk_parking',
    'traffic',
    'person',
    'vehicle',
    'pole',
    'structure',
    'ground',
    'vegetation',
    'terrain',
    'sky'
]
PALETTE = {
    'static':                   (  0,   0,   0),
    'road_sidewalk_parking':    (128,  64, 128),
    'traffic':                  (250, 170,  30),
    'person':                   (220,  20,  60),
    'vehicle':                  (  0,   0, 142),
    'pole':                     (153, 153, 153),
    'structure':                ( 70,  70,  70),
    'ground':                   ( 81,   0,  81),
    'vegetation':               (107, 142,  35),
    'terrain':                  (152, 251, 152),
    'sky':                      ( 70, 130, 180),
}

# small closing kernel
_CLOSING_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

def process_video(in_path, out_folder, sid, status_dict):
    cap   = cv2.VideoCapture(in_path)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    base      = os.path.splitext(os.path.basename(in_path))[0]
    seg_path  = os.path.join(out_folder, f"{base}_segmented.mp4")
    gray_path = os.path.join(out_folder, f"{base}_grayscale.mp4")
    over_path = os.path.join(out_folder, f"{base}_overlay.mp4")

    # H.264 FourCC so browsers play natively
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    vw_seg  = cv2.VideoWriter(seg_path,  fourcc, fps, (w,h))
    vw_gray = cv2.VideoWriter(gray_path, fourcc, fps, (w,h))
    vw_over = cv2.VideoWriter(over_path, fourcc, fps, (w,h))

    frame_i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # update progress but preserve 'type'
        status_dict[sid].update({
            'status':   'processing',
            'message':  f'Frame {frame_i+1}/{total}',
            'progress': int(frame_i/total * 100)
        })

        # run YOLOv8 seg
        res      = YO.predict(source=frame, verbose=False)[0]
        masks_np = res.masks.data.cpu().numpy()  # shape (N, mh, mw)
        classes  = res.boxes.cls.cpu().numpy().astype(int)  # shape (N,)

        # build up a low-res class map
        mh, mw = masks_np.shape[1:]
        class_map = np.zeros((mh, mw), dtype=np.uint8)

        for inst_mask, cls_id in zip(masks_np, classes):
            # wherever instance mask >0.5, assign that class
            m = inst_mask > 0.5
            class_map[m] = cls_id

        # upsample to original
        class_map = cv2.resize(class_map, (w, h), interpolation=cv2.INTER_NEAREST)

        # per-class closing to remove speckles
        cleaned = np.zeros_like(class_map)
        for cls_id in np.unique(class_map):
            if cls_id == 0:
                continue
            m = (class_map == cls_id).astype(np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _CLOSING_KERNEL)
            cleaned[m > 0] = cls_id

        # generate RGB segmentation & grayscale mask
        seg_image = np.zeros_like(frame)
        gray_mask = (cleaned > 0).astype(np.uint8) * 255

        for cls_id in np.unique(cleaned):
            if cls_id == 0:
                continue
            cls_name = CLASS_NAMES[cls_id]
            color    = PALETTE.get(cls_name, (0,255,0))
            seg_image[cleaned == cls_id] = color

        gray_image = cv2.cvtColor(gray_mask, cv2.COLOR_GRAY2BGR)
        overlay    = cv2.addWeighted(frame, 0.5, seg_image, 0.5, 0)

        # write
        vw_seg.write(seg_image)
        vw_gray.write(gray_image)
        vw_over.write(overlay)

        frame_i += 1

    cap.release()
    vw_seg.release()
    vw_gray.release()
    vw_over.release()

    # mark completed (again update to preserve 'type')
    status_dict[sid].update({
        'status':    'completed',
        'message':   'Processing completed',
        'progress':  100,
        'original':  os.path.basename(in_path),
        'segmented': os.path.basename(seg_path),
        'grayscale': os.path.basename(gray_path),
        'overlay':   os.path.basename(over_path),
    })


def process_image(in_path, out_folder):
    img      = cv2.imread(in_path)
    h, w     = img.shape[:2]
    res      = YO.predict(source=img, verbose=False)[0]
    masks_np = res.masks.data.cpu().numpy()
    classes  = res.boxes.cls.cpu().numpy().astype(int)

    # very similar pipeline to video
    class_map = np.zeros((h, w), dtype=np.uint8)
    for inst_mask, cls_id in zip(masks_np, classes):
        m = cv2.resize((inst_mask > 0.5).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        class_map[m > 0] = cls_id

    # closing
    cleaned = np.zeros_like(class_map)
    for cls_id in np.unique(class_map):
        if cls_id == 0: continue
        m = (class_map == cls_id).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _CLOSING_KERNEL)
        cleaned[m > 0] = cls_id

    seg_image = np.zeros_like(img)
    gray_mask = (cleaned > 0).astype(np.uint8) * 255

    for cls_id in np.unique(cleaned):
        if cls_id == 0: continue
        name  = CLASS_NAMES[cls_id]
        color = PALETTE.get(name, (0,255,0))
        seg_image[cleaned == cls_id] = color

    gray_image = cv2.cvtColor(gray_mask, cv2.COLOR_GRAY2BGR)
    overlay    = cv2.addWeighted(img, 0.5, seg_image, 0.5, 0)

    base      = os.path.splitext(os.path.basename(in_path))[0]
    seg_out   = os.path.join(out_folder, f"{base}_segmented.png")
    gray_out  = os.path.join(out_folder, f"{base}_grayscale.png")
    over_out  = os.path.join(out_folder, f"{base}_overlay.png")

    cv2.imwrite(seg_out, seg_image)
    cv2.imwrite(gray_out, gray_image)
    cv2.imwrite(over_out, overlay)

    return {
        'original':  os.path.basename(in_path),
        'segmented': os.path.basename(seg_out),
        'grayscale': os.path.basename(gray_out),
        'overlay':   os.path.basename(over_out),
    }
