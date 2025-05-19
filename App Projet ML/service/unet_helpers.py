import os
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
from scipy.ndimage import median_filter
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian

# --- Model input size ---
INPUT_HEIGHT = 256
INPUT_WIDTH  = 96

# --- Cityscapes merged palette setup ---
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

# merge map for scratch model
merge_map = {
    'road':'road_sidewalk_parking','sidewalk':'road_sidewalk_parking','parking':'road_sidewalk_parking',
    'traffic light':'traffic','traffic sign':'traffic',
    'person':'person','rider':'person',
    'car':'vehicle','truck':'vehicle','bus':'vehicle',
    'train':'vehicle','motorcycle':'vehicle','bicycle':'vehicle',
    'pole':'pole','polegroup':'pole',
    'wall':'structure','building':'structure','fence':'structure','bridge':'structure'
}
for name in original_colors:
    merge_map.setdefault(name, name)

RGB2IDX = {rgb: merged_names.index(merge_map[name])
           for name, rgb in original_colors.items()}

# build palette array
palette = np.zeros((len(merged_names), 3), dtype=np.uint8)
for rgb, idx in RGB2IDX.items():
    palette[idx] = rgb

# load TF model once
_MODEL = tf.keras.models.load_model(
    os.path.join('checkpoints','best_model_scratch.h5'),
    compile=False
)

# CRF parameters
CRF_ITER       = 5
GAUSS_SXY      = 3
BILAT_SXY      = 80
BILAT_SRGB     = 13
CRF_COMP_GAUSS = 3
CRF_COMP_BILAT = 10


def preprocess(img: Image.Image) -> np.ndarray:
    """Resize & normalize to model input"""
    arr = np.array(img.resize((INPUT_WIDTH, INPUT_HEIGHT), Image.BILINEAR),
                   dtype=np.float32) / 255.0
    return arr[None, ...]


def tta_predict(x: np.ndarray) -> np.ndarray:
    """Predict + horizontal flip TTA"""
    p1 = _MODEL.predict(x)[0]
    xf = x[:, :, ::-1, :]
    p2 = _MODEL.predict(xf)[0][:, ::-1, :]
    return (p1 + p2) / 2.0


def crf_refine(img: np.ndarray, prob: np.ndarray) -> np.ndarray:
    """DenseCRF + pairwise cleanup"""
    h, w, c = prob.shape
    d = dcrf.DenseCRF2D(w, h, c)
    U = unary_from_softmax(np.ascontiguousarray(prob.transpose(2,0,1)))
    d.setUnaryEnergy(U)
    feats_g = create_pairwise_gaussian(sdims=(GAUSS_SXY,GAUSS_SXY), shape=prob.shape[:2])
    d.addPairwiseEnergy(feats_g, compat=CRF_COMP_GAUSS)
    feats_b = create_pairwise_bilateral(sdims=(BILAT_SXY,BILAT_SXY),
                                        schan=(BILAT_SRGB,)*3,
                                        img=img, chdim=2)
    d.addPairwiseEnergy(feats_b, compat=CRF_COMP_BILAT)
    Q = d.inference(CRF_ITER)
    refined = np.array(Q).reshape((c, h, w)).transpose(1,2,0)
    return refined


def clean_cc(idx_map: np.ndarray, min_size=200) -> np.ndarray:
    """Remove tiny connected components"""
    out = np.zeros_like(idx_map)
    for cls in range(len(merged_names)):
        mask = (idx_map == cls).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        for lbl in range(1, n):
            if stats[lbl, cv2.CC_STAT_AREA] >= min_size:
                out[labels == lbl] = cls
    return out


def postprocess(pred_softmax: np.ndarray, orig_img: Image.Image) -> Image.Image:
    """Full-res CRF + cleanup + palette application"""
    ow, oh = orig_img.size
    prob = tf.image.resize(pred_softmax[None], [oh, ow], method='bilinear').numpy()[0]
    prob = crf_refine(np.array(orig_img), prob)
    idx  = np.argmax(prob, axis=-1).astype(np.uint8)
    idx  = clean_cc(idx, min_size=200)
    # morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    closed = np.zeros_like(idx)
    for cls in range(len(merged_names)):
        bm = (idx == cls).astype(np.uint8)
        cm = cv2.morphologyEx(bm, cv2.MORPH_CLOSE, kernel)
        closed[cm == 1] = cls
    smooth = np.zeros_like(closed)
    for cls in range(len(merged_names)):
        bm = (closed == cls).astype(np.uint8)
        bm = median_filter(bm, size=3)
        smooth[bm == 1] = cls
    # map to palette
    return Image.fromarray(palette[smooth])
