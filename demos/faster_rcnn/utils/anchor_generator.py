import math
import numpy as np

def get_base_anchor(scales, ratios):
    base_anchors = []

    for ratio in ratios:
        for scale in scales:
            w = math.sqrt(scale ** 2  / ratio)
            h = scale ** 2 / w
            base_anchors.append((w, h))

    base_anchors = np.array(base_anchors, dtype=np.float32).reshape([-1, 2])
    return base_anchors