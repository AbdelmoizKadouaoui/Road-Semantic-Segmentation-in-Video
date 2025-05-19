import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from scipy.ndimage import median_filter
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian

# -------------- CONFIG --------------
MODEL_PATH = "C:/Users/MBQ/Desktop/ML Projet Test/App Projet ML/models/best_model_scratch.h5"
INPUT_H, INPUT_W = 256, 96

# CRF params
CRF_ITER       = 5
GAUSS_SXY      = 3
BILAT_SXY      = 80
BILAT_SRGB     = 13
CRF_COMP_GAUSS = 3
CRF_COMP_BILAT = 10

# -------------- LOAD MODEL --------------
_model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# -------------- PALETTE & MERGE --------------
merged_names = [
    'caravan','dynamic','ground','guard rail','license plate','person','pole',
    'rail track','road_sidewalk_parking','sky','static','structure','terrain',
    'traffic','trailer','tunnel','vegetation','vehicle'
]

original_colors = {
    'static':        (  0,   0,   0),
    'dynamic':       (111,  74,   0),
    'ground':        ( 81,   0,  81),
    'road':          (128,  64, 128),
    'sidewalk':      (244,  35, 232),
    'parking':       (250, 170, 160),
    'rail track':    (230, 150, 140),
    'building':      ( 70,  70,  70),
    'wall':          (102, 102, 156),
    'fence':         (190, 153, 153),
    'guard rail':    (180, 165, 180),
    'bridge':        (150, 100, 100),
    'tunnel':        (150, 120,  90),
    'pole':          (153, 153, 153),
    'polegroup':     (153, 153, 153),
    'traffic light': (250, 170,  30),
    'traffic sign':  (220, 220,   0),
    'vegetation':    (107, 142,  35),
    'terrain':       (152, 251, 152),
    'sky':           ( 70, 130, 180),
    'person':        (220,  20,  60),
    'rider':         (255,   0,   0),
    'car':           (  0,   0, 142),
    'truck':         (  0,   0,  70),
    'bus':           (  0,  60, 100),
    'caravan':       (  0,   0,  90),
    'trailer':       (  0,   0, 110),
    'train':         (  0,  80, 100),
    'motorcycle':    (  0,   0, 230),
    'bicycle':       (119,  11,  32),
    'license plate': (  0,   0, 142),
}

merge_map = {
    'road':'road_sidewalk_parking', 'sidewalk':'road_sidewalk_parking','parking':'road_sidewalk_parking',
    'traffic light':'traffic','traffic sign':'traffic',
    'person':'person','rider':'person',
    'car':'vehicle','truck':'vehicle','bus':'vehicle',
    'train':'vehicle','motorcycle':'vehicle','bicycle':'vehicle',
    'pole':'pole','polegroup':'pole',
    'wall':'structure','building':'structure','fence':'structure','bridge':'structure'
}
for name in original_colors:
    merge_map.setdefault(name, name)

RGB2IDX = { rgb: merged_names.index(merge_map[name])
            for name, rgb in original_colors.items() }
PALETTE = np.zeros((len(merged_names),3), dtype=np.uint8)
for rgb, idx in RGB2IDX.items():
    PALETTE[idx] = rgb

# -------------- HELPERS --------------

def tta_predict(x: np.ndarray) -> np.ndarray:
    p1 = _model.predict(x, verbose=0)[0]
    x2 = x[:, :, ::-1, :]
    p2 = _model.predict(x2, verbose=0)[0][:, ::-1, :]
    return (p1 + p2) / 2.0

def crf_refine(img: np.ndarray, prob: np.ndarray) -> np.ndarray:
    # ensure contiguous
    img  = np.ascontiguousarray(img)
    prob = np.ascontiguousarray(prob)
    h,w,c = prob.shape
    d = dcrf.DenseCRF2D(w,h,c)
    # transpose and copy to guarantee C‐contiguous
    arr = prob.transpose(2,0,1).copy(order='C')
    unary = unary_from_softmax(arr)
    d.setUnaryEnergy(unary)
    g_feats = create_pairwise_gaussian(sdims=(GAUSS_SXY,GAUSS_SXY), shape=(h,w))
    b_feats = create_pairwise_bilateral(sdims=(BILAT_SXY,BILAT_SXY),
                                        schan=(BILAT_SRGB,)*3,
                                        img=img, chdim=2)
    d.addPairwiseEnergy(g_feats, compat=CRF_COMP_GAUSS)
    d.addPairwiseEnergy(b_feats, compat=CRF_COMP_BILAT)
    Q = d.inference(CRF_ITER)
    return np.array(Q).reshape((c,h,w)).transpose(1,2,0)

def clean_cc(idx_map: np.ndarray, min_size=200) -> np.ndarray:
    out = np.zeros_like(idx_map)
    for cls in range(len(merged_names)):
        mask = (idx_map==cls).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        for lbl in range(1,n):
            if stats[lbl, cv2.CC_STAT_AREA] >= min_size:
                out[labels==lbl] = cls
    return out

def postprocess(pred_softmax: np.ndarray, orig_img: Image.Image) -> Image.Image:
    ow, oh = orig_img.size
    # resize softmax to original resolution
    prob = cv2.resize(pred_softmax, (ow,oh), interpolation=cv2.INTER_LINEAR)
    prob = crf_refine(np.array(orig_img), prob)
    idx  = np.argmax(prob, axis=-1).astype(np.uint8)
    idx  = clean_cc(idx)
    # close + median
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    closed = cv2.morphologyEx(idx, cv2.MORPH_CLOSE, kernel)
    smooth = median_filter(closed, size=3)
    return Image.fromarray(PALETTE[smooth])

# -------------- PUBLIC --------------

def process_image(in_path: str, out_folder: str) -> dict:
    orig = Image.open(in_path).convert("RGB")
    x = np.array(orig.resize((INPUT_W,INPUT_H), Image.BILINEAR),dtype=np.float32)/255.0
    soft = tta_predict(x[None,...])
    mask = postprocess(soft, orig)
    base,_ = os.path.splitext(os.path.basename(in_path))

    # save outputs
    out = {}
    for name,img in [
      ("original", orig),
      ("segmented", mask),
      ("overlay", Image.blend(orig, mask, 0.5))
    ]:
        fn = f"{base}_{name}.png"
        img.save(os.path.join(out_folder, fn))
        out[name] = fn

    # grayscale
    gray = mask.convert("L").point(lambda i: int(i*(255/len(merged_names))))
    fn = f"{base}_grayscale.png"
    gray.save(os.path.join(out_folder, fn))
    out["grayscale"] = fn

    return out

def process_video(in_path: str,
                  out_folder: str,
                  sid: str,
                  status_dict: dict) -> None:
    cap = cv2.VideoCapture(in_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    base = os.path.splitext(os.path.basename(in_path))[0]

    # target 10 FPS
    target_fps = 10
    skip = max(1, round(orig_fps / target_fps))

    # H.264 in MP4 container
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    paths = {
        "segmented": os.path.join(out_folder, f"{base}_segmented.mp4"),
        "grayscale": os.path.join(out_folder, f"{base}_grayscale.mp4"),
        "overlay":   os.path.join(out_folder, f"{base}_overlay.mp4")
    }
    writers = {
        k: cv2.VideoWriter(p, fourcc, target_fps, (w, h))
        for k, p in paths.items()
    }

    frame_i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # update status
        status_dict[sid] = {
            'type':     'unet-video',
            'status':   'processing',
            'message':  f'Frame {frame_i+1}/{total}',
            'progress': int(frame_i / total * 100) if total else 0
        }

        if frame_i % skip == 0:
            # preprocess + predict
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            arr = np.array(img.resize((INPUT_W, INPUT_H)), np.float32) / 255.0
            soft = tta_predict(arr[None, ...])
            mask_img = postprocess(soft, img)

            seg  = cv2.cvtColor(np.array(mask_img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(np.array(mask_img.convert("L")), cv2.COLOR_GRAY2BGR)
            over = cv2.addWeighted(frame, 0.5, seg, 0.5, 0)

            writers["segmented"].write(seg)
            writers["grayscale"].write(gray)
            writers["overlay"].write(over)

        frame_i += 1

    cap.release()
    for w in writers.values():
        w.release()

    # final status
    status_dict[sid] = {
        'type':      'unet-video',
        'status':    'completed',
        'message':   'Processing completed',
        'progress':  100,
        'original':   os.path.basename(in_path),
        'segmented':  os.path.basename(paths["segmented"]),
        'grayscale':  os.path.basename(paths["grayscale"]),
        'overlay':    os.path.basename(paths["overlay"])
    }
