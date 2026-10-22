"""逐格「有没有棋子」的判定（皮肤无关）。

棋子是实心圆盘（中心亮），棋盘底色更深且随山水背景渐变——所以「有子」
就是「格中心比棋盘底色亮」。不写死亮度阈值：每帧对全盘 90 格做迭代
两类中点自适应分界，皮肤/配色换了也不用调参。

实测（当前皮肤，精修网格）：有子 202~237，空位 36~114，余量 +88，零误判。

这个判定被两处共用：
    * tracking：走子检测（哪格空了、哪格多了子）
    * recognize：先定「有没有子」，再让模板只负责「是哪颗子」
      ——模板匹配对网格偏移很敏感，空位容易被邻格棋子蹭出高分，
      用占空图做硬约束就不会无中生有。

反面教材：早先用「中心最暗 20% 像素」判「墨迹深=有子」，在深色棋盘上
完全反了（空位比白圆盘还暗），这正是真机走子检测死锁的根因。
"""

import cv2
import numpy as np

from .calibrate import FILES, RANKS

OCCUPANCY_FRAC = 0.28      # 采样区半径 = 0.28 * cell（避开格线与相邻棋子）
OCCUPANCY_TOL = 0.0        # 自适应阈值再向「有子」侧收紧的量（0 = 取两类中点）


def center_luminance(rect, cell, margin, r, f, frac=OCCUPANCY_FRAC):
    """格中心区域的中位亮度。

    用中位数而不是均值：棋盘格线正好穿过交叉点中心，线比底色/棋子都亮，
    均值会被线条拉高（实测某空位因此从 60 跳到 116），中位数不受影响。
    """
    k = max(3, int(cell * frac))
    x = margin + f * cell
    y = margin + r * cell
    patch = rect[int(y) - k : int(y) + k, int(x) - k : int(x) + k]
    if patch.size == 0:
        return 0.0
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return float(np.median(gray))


def ring_luminance(rect, cell, margin, r, f, inner=0.20, outer=0.28):
    """采样区**外圈**的中位亮度：真棋子的圆盘铺满整个采样区，外圈也是亮的。

    「上一步落点」的白色十字光点只有中心一小块亮，光晕到不了外圈——实测
    它在 (0,4) 让格中心亮度虚高、被判成有子，随后被模板认成黑「车」，把
    红帅身边造出一辆非法黑车、FEN 直接不合法（引擎报 King can be captured）。
    只看中心亮度分不出「实心圆盘」和「小光点」，加一圈外环就能分开。
    """
    r_out = max(5, int(cell * outer))
    r_in = max(2, int(cell * inner))
    x = margin + f * cell
    y = margin + r * cell
    patch = rect[int(y) - r_out : int(y) + r_out, int(x) - r_out : int(x) + r_out]
    if patch.size == 0:
        return 0.0
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    d2 = (xx - r_out) ** 2 + (yy - r_out) ** 2
    mask = (d2 >= r_in * r_in) & (d2 <= r_out * r_out)
    vals = gray[mask]
    return float(np.median(vals)) if vals.size else 0.0


def glare_ratio(rect, cell, margin, r, f, frac=OCCUPANCY_FRAC, level=245):
    """中心区「接近纯白」的像素占比。

    棋子的白圆盘实测亮度 202~237（带一点灰），而「上一步落点」的十字光点
    接近纯白并带光晕。光点会被模板认成棋子（实测在红帅旁边造出一辆黑车，
    FEN 直接不合法、引擎报 King can be captured），亮度中位数和外圈判据都
    挡不住它——因为它比棋子还亮。这里直接卡「过亮」：占比过高就不是棋子。
    """
    k = max(3, int(cell * frac))
    x = margin + f * cell
    y = margin + r * cell
    patch = rect[int(y) - k : int(y) + k, int(x) - k : int(x) + k]
    if patch.size == 0:
        return 0.0
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return float((gray >= level).mean())


def occupancy_threshold(values, tol=OCCUPANCY_TOL):
    """对全盘格中心亮度做「迭代两类中点」，得到有子/空位的自适应分界。

    Otsu 会被离群值拉偏（实测被一个背景偏亮的空位从 145 拉到 116），
    迭代取两类均值的中点更稳：只要有子/空位两簇存在，分界就落在谷底。
    全盘都空或都有子时两簇退化，函数会收敛到原值附近，不影响判断。
    """
    flat = np.asarray(values, dtype=np.float64).ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 0.0
    t = float(np.median(flat))
    for _ in range(20):
        lo = flat[flat <= t]
        hi = flat[flat > t]
        if lo.size == 0 or hi.size == 0:
            break
        nt = (lo.mean() + hi.mean()) / 2.0
        if abs(nt - t) < 0.2:
            t = nt
            break
        t = nt
    return t + tol


def warmth_map(rect, cell, margin, frac=OCCUPANCY_FRAC):
    """格中心区域的暖度（R−B 中位数）。

    换皮肤时亮度这一路会整条失效：木黄棋盘配白盘棋子时「有子比空位亮」，
    换成青灰棋盘配暖木牌棋子后两者亮度几乎重叠（166 vs 164，分离度 −38），
    而 R−B 是 96 vs 14（+44）。所以两个特征都算，谁分得开用谁。
    """
    k = max(3, int(cell * frac))
    out = np.empty((RANKS, FILES), dtype=np.float64)
    for r in range(RANKS):
        for f in range(FILES):
            x = margin + f * cell
            y = margin + r * cell
            patch = rect[int(y) - k : int(y) + k, int(x) - k : int(x) + k]
            if patch.size == 0:
                out[r, f] = 0.0
                continue
            b, _, rr = [float(np.median(patch[:, :, i])) for i in range(3)]
            out[r, f] = rr - b
    return out


def _split(feat):
    """切一刀分成两簇，返回 (「数值更大」那侧的掩码, 两簇均值差)。"""
    thr = occupancy_threshold(feat)
    hi = feat > thr
    n = feat.size
    if hi.sum() == 0 or hi.sum() == n:
        return hi, 0.0
    return hi, float(abs(feat[hi].mean() - feat[~hi].mean()))


def occupancy_map(rect, cell, margin):
    """整盘「每格是否真有棋子」的布尔表（屏幕坐标）。

    自适应选特征：亮度和暖度各切一刀，取两簇分得更开的那个（换皮肤时
    总有一个能用）。切出来的「多数簇」一律视为空位——棋子最多 32 个，
    棋盘有 90 格，有子必然是少数。
    """
    lum = np.empty((RANKS, FILES), dtype=np.float64)
    ring = np.empty((RANKS, FILES), dtype=np.float64)
    for r in range(RANKS):
        for f in range(FILES):
            lum[r, f] = center_luminance(rect, cell, margin, r, f)
            ring[r, f] = ring_luminance(rect, cell, margin, r, f)
    warm = warmth_map(rect, cell, margin)

    mask_l, sep_l = _split(lum)
    mask_w, sep_w = _split(warm)
    if sep_w > sep_l:
        mask = mask_w
        if mask.sum() > mask.size - mask.sum():
            mask = ~mask
        # 暖度本身就把光点排除了（光点是白的，R−B 很低），不必再叠加外圈判据
        return mask
    mask = mask_l & (ring > occupancy_threshold(lum))
    if mask.sum() > mask.size - mask.sum():
        return ~mask & (ring > occupancy_threshold(lum))
    return mask
