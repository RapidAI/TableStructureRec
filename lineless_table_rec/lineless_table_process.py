# ------------------------------------------------------------------------------
# Part of implementation is adopted from CenterNet,
# made publicly available under the MIT License at https://github.com/xingyizhou/CenterNet.git
# ------------------------------------------------------------------------------
import warnings
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np

# suppress warnings
warnings.filterwarnings("ignore")


class DetProcess:
    def __init__(self, K: int = 3000, num_classes: int = 2, scale: float = 1.0):
        self.K = K
        self.num_classes = num_classes
        self.scale = scale
        self.max_per_image = 3000

    def __call__(
        self, det_out: Dict[str, np.ndarray], meta: Dict[str, Union[int, np.ndarray]]
    ):
        hm = self.sigmoid(det_out["hm"])
        dets, keep, logi, cr = ctdet_4ps_decode(
            hm[:, 0:1, :, :],
            det_out["wh"],
            det_out["ax"],
            det_out["cr"],
            reg=det_out["reg"],
            K=self.K,
        )

        raw_dets = dets
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_4ps_post_process_upper_left(
            dets.copy(),
            [meta["c"]],
            [meta["s"]],
            meta["out_height"],
            meta["out_width"],
            2,
        )
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 9)
            dets[0][j][:, :8] /= self.scale
        dets = dets[0]
        detections = [dets]

        logi += cr
        results = self.merge_outputs(detections)
        slct_logi_feat, slct_dets_feat = self.filter(results, logi, raw_dets[:, :, :8])
        slct_output_dets = results[1][: slct_logi_feat.shape[1], :8]
        return slct_logi_feat, slct_dets_feat, slct_output_dets

    @staticmethod
    def sigmoid(data: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-data))

    def merge_outputs(self, detections: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        # thresh_conf, thresh_min, thresh_max = 0.1, 0.5, 0.7
        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0
            ).astype(np.float32)

        scores = np.hstack([results[j][:, 8] for j in range(1, self.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_inds = results[j][:, 8] >= thresh
                results[j] = results[j][keep_inds]
        return results

    @staticmethod
    def filter(
        results: Dict[int, np.ndarray], logi: np.ndarray, ps: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # this function select boxes
        batch_size, feat_dim = logi.shape[0], logi.shape[2]
        num_valid = sum(results[1][:, 8] >= 0.15)

        slct_logi = np.zeros((batch_size, num_valid, feat_dim), dtype=np.float32)
        slct_dets = np.zeros((batch_size, num_valid, 8), dtype=np.int32)
        for i in range(batch_size):
            for j in range(num_valid):
                slct_logi[i, j, :] = logi[i, j, :]
                slct_dets[i, j, :] = ps[i, j, :]

        return slct_logi, slct_dets


def ctdet_4ps_decode(
    heat: np.ndarray,
    wh: np.ndarray,
    ax: np.ndarray,
    cr: np.ndarray,
    reg: np.ndarray = None,
    cat_spec_wh: bool = False,
    K: int = 100,
):
    batch, cat, _, width = heat.shape
    heat, keep = _nms(heat)
    scores, inds, clses, ys, xs = _topk(heat, K=K)

    if reg is not None:
        reg = _tranpose_and_gather_feat(reg, inds)
        reg = reg.reshape(batch, K, 2)
        xs = xs.reshape(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.reshape(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.reshape(batch, K, 1) + 0.5
        ys = ys.reshape(batch, K, 1) + 0.5

    wh = _tranpose_and_gather_feat(wh, inds)
    ax = _tranpose_and_gather_feat(ax, inds)

    if cat_spec_wh:
        wh = wh.reshape(batch, K, cat, 8)
        clses_ind = clses.reshape(batch, K, 1, 1).expand(batch, K, 1, 8)
        wh = wh.gather(2, clses_ind).reshape(batch, K, 8)
    else:
        wh = wh.reshape(batch, K, 8)

    clses = clses.reshape(batch, K, 1)
    scores = scores.reshape(batch, K, 1)

    bboxes_vec = [
        xs - wh[..., 0:1],
        ys - wh[..., 1:2],
        xs - wh[..., 2:3],
        ys - wh[..., 3:4],
        xs - wh[..., 4:5],
        ys - wh[..., 5:6],
        xs - wh[..., 6:7],
        ys - wh[..., 7:8],
    ]
    bboxes = np.concatenate(bboxes_vec, axis=2)

    cc_match = np.concatenate(
        [
            (xs - wh[..., 0:1]) + width * np.round(ys - wh[..., 1:2]),
            (xs - wh[..., 2:3]) + width * np.round(ys - wh[..., 3:4]),
            (xs - wh[..., 4:5]) + width * np.round(ys - wh[..., 5:6]),
            (xs - wh[..., 6:7]) + width * np.round(ys - wh[..., 7:8]),
        ],
        axis=2,
    )
    cc_match = np.round(cc_match).astype(np.int64)
    cr_feat = _get_4ps_feat(cc_match, cr)
    cr_feat = cr_feat.sum(axis=3)

    detections = np.concatenate([bboxes, scores, clses], axis=2)
    return detections, keep, ax, cr_feat


def _nms(heat: np.ndarray, kernel: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    pad = (kernel - 1) // 2
    hmax = max_pool(heat, kernel_size=kernel, stride=1, padding=pad)
    keep = hmax == heat
    return heat * keep, keep


def max_pool(
    img: np.ndarray, kernel_size: int, stride: int, padding: int
) -> np.ndarray:
    h, w = img.shape[2:]
    img = np.pad(
        img,
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        "constant",
        constant_values=0,
    )

    res_h = ((h + 2 - kernel_size) // stride) + 1
    res_w = ((w + 2 - kernel_size) // stride) + 1
    res = np.zeros((img.shape[0], img.shape[1], res_h, res_w))
    for i in range(res_h):
        for j in range(res_w):
            temp = img[
                :,
                :,
                i * stride : i * stride + kernel_size,
                j * stride : j * stride + kernel_size,
            ]
            res[:, :, i, j] = temp.max()
    return res


def _topk(
    scores: np.ndarray, K: int = 40
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    batch, cat, height, width = scores.shape

    topk_scores, topk_inds = find_topk(scores.reshape(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = topk_inds / width
    topk_xs = np.float32(np.int32(topk_inds % width))

    topk_score, topk_ind = find_topk(topk_scores.reshape(batch, -1), K)
    topk_clses = np.int32(topk_ind / K)
    topk_inds = _gather_feat(topk_inds.reshape(batch, -1, 1), topk_ind).reshape(
        batch, K
    )
    topk_ys = _gather_feat(topk_ys.reshape(batch, -1, 1), topk_ind).reshape(batch, K)
    topk_xs = _gather_feat(topk_xs.reshape(batch, -1, 1), topk_ind).reshape(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def find_topk(
    a: np.ndarray, k: int, axis: int = -1, largest: bool = True, sorted: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    if axis is None:
        axis_size = a.size
    else:
        axis_size = a.shape[axis]
    assert 1 <= k <= axis_size

    a = np.asanyarray(a)
    if largest:
        index_array = np.argpartition(a, axis_size - k, axis=axis)
        topk_indices = np.take(index_array, -np.arange(k) - 1, axis=axis)
    else:
        index_array = np.argpartition(a, k - 1, axis=axis)
        topk_indices = np.take(index_array, np.arange(k), axis=axis)

    topk_values = np.take_along_axis(a, topk_indices, axis=axis)
    if sorted:
        sorted_indices_in_topk = np.argsort(topk_values, axis=axis)
        if largest:
            sorted_indices_in_topk = np.flip(sorted_indices_in_topk, axis=axis)

        sorted_topk_values = np.take_along_axis(
            topk_values, sorted_indices_in_topk, axis=axis
        )
        sorted_topk_indices = np.take_along_axis(
            topk_indices, sorted_indices_in_topk, axis=axis
        )
        return sorted_topk_values, sorted_topk_indices
    return topk_values, topk_indices


def _gather_feat(feat: np.ndarray, ind: np.ndarray) -> np.ndarray:
    dim = feat.shape[2]
    ind = np.broadcast_to(ind[:, :, None], (ind.shape[0], ind.shape[1], dim))
    feat = _gather(feat, 1, ind)
    return feat


def _gather(data: np.ndarray, dim: int, index: np.ndarray) -> np.ndarray:
    """
    Gathers values along an axis specified by dim.
    For a 3-D tensor the output is specified by:
        out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
        out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

    :param dim: The axis along which to index
    :param index: A tensor of indices of elements to gather
    :return: tensor of gathered values
    """
    idx_xsection_shape = index.shape[:dim] + index.shape[dim + 1 :]
    data_xsection_shape = data.shape[:dim] + data.shape[dim + 1 :]
    if idx_xsection_shape != data_xsection_shape:
        raise ValueError(
            "Except for dimension "
            + str(dim)
            + ", all dimensions of index and data should be the same size"
        )

    if index.dtype != np.int64:
        raise TypeError("The values of index must be integers")

    data_swaped = np.swapaxes(data, 0, dim)
    index_swaped = np.swapaxes(index, 0, dim)
    gathered = np.take_along_axis(data_swaped, index_swaped, axis=0)
    return np.swapaxes(gathered, 0, dim)


def _tranpose_and_gather_feat(feat: np.ndarray, ind: np.ndarray) -> np.ndarray:
    feat = np.ascontiguousarray(np.transpose(feat, [0, 2, 3, 1]))
    feat = feat.reshape(feat.shape[0], -1, feat.shape[3])
    feat = _gather_feat(feat, ind)
    return feat


def _get_4ps_feat(cc_match: np.ndarray, output: np.ndarray) -> np.ndarray:
    if isinstance(output, dict):
        feat = output["cr"]
    else:
        feat = output

    feat = np.ascontiguousarray(feat.transpose(0, 2, 3, 1))
    feat = feat.reshape(feat.shape[0], -1, feat.shape[3])
    feat = feat[..., None]
    feat = np.concatenate([feat] * 4, axis=-1)

    dim = feat.shape[2]
    cc_match = cc_match[..., None, :]
    cc_match = np.concatenate([cc_match] * dim, axis=2)
    if not (isinstance(output, dict)):
        cc_match = np.where(
            cc_match < feat.shape[1],
            cc_match,
            (feat.shape[0] - 1) * np.ones(cc_match.shape).astype(np.int64),
        )

        cc_match = np.where(
            cc_match >= 0, cc_match, np.zeros(cc_match.shape).astype(np.int64)
        )
    feat = np.take_along_axis(feat, cc_match, axis=1)
    return feat


def ctdet_4ps_post_process_upper_left(
    dets: np.ndarray,
    c: List[np.ndarray],
    s: List[float],
    h: int,
    w: int,
    num_classes: int,
) -> np.ndarray:
    # dets: batch x max_dets x dim
    # return 1-based class det dict
    ret = []
    for i in range(dets.shape[0]):
        top_preds = {}
        dets[i, :, 0:2] = transform_preds_upper_left(
            dets[i, :, 0:2], c[i], s[i], (w, h)
        )
        dets[i, :, 2:4] = transform_preds_upper_left(
            dets[i, :, 2:4], c[i], s[i], (w, h)
        )
        dets[i, :, 4:6] = transform_preds_upper_left(
            dets[i, :, 4:6], c[i], s[i], (w, h)
        )
        dets[i, :, 6:8] = transform_preds_upper_left(
            dets[i, :, 6:8], c[i], s[i], (w, h)
        )
        classes = dets[i, :, -1]
        for j in range(num_classes):
            inds = classes == j
            tmp_top_pred = [
                dets[i, inds, :8].astype(np.float32),
                dets[i, inds, 8:9].astype(np.float32),
            ]
            top_preds[j + 1] = np.concatenate(tmp_top_pred, axis=1).tolist()
        ret.append(top_preds)
    return ret


def transform_preds_upper_left(
    coords: np.ndarray,
    center: np.ndarray,
    scale: float,
    output_size: Tuple[int, int],
) -> np.ndarray:
    target_coords = np.zeros(coords.shape)

    trans = get_affine_transform_upper_left(center, scale, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_affine_transform_upper_left(
    center: np.ndarray,
    scale: float,
    output_size: List[Tuple[int, int]],
    inv: int = 0,
) -> np.ndarray:
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    dst[0, :] = [0, 0]
    if center[0] < center[1]:
        src[1, :] = [scale[0], center[1]]
        dst[1, :] = [output_size[0], 0]
    else:
        src[1, :] = [center[0], scale[0]]
        dst[1, :] = [0, output_size[0]]
    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans


def get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def affine_transform(pt: np.ndarray, t: np.ndarray) -> np.ndarray:
    new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]
