"""棋盘几何：四角 → 网格交叉点；以及棋子模板生成。"""

import cv2
import numpy as np

FILES = 9
RANKS = 10


def grid_intersections(corners, ranks=RANKS, files=FILES):
    """给定四角（左上、右上、右下、左下），返回 (ranks, files, 2) 交叉点坐标。

    在原图坐标里做双线性插值；[r][f] = (x, y)，r=0 为顶部，f=0 为左侧。
    """
    corners = np.asarray(corners, dtype=np.float32)
    tl, tr, br, bl = corners[0], corners[1], corners[2], corners[3]
    pts = []
    for r in range(ranks):
        t = r / (ranks - 1)
        left = (1 - t) * tl + t * bl
        right = (1 - t) * tr + t * br
        row = []
        for f in range(files):
            s = f / (files - 1)
            row.append((1 - s) * left + s * right)
        pts.append(row)
    return np.array(pts, dtype=np.float32)


def rectify(image, corners, cell=140):
    """把棋盘四角区域矫正成正面图。

    返回 (矫正图, cell, margin)。交叉点 (r, f) 在矫正图中位于
    (margin + f*cell, margin + r*cell)，四周留 margin 空白便于裁剪。
    """
    margin = cell
    W = (FILES - 1) * cell + 2 * margin
    H = (RANKS - 1) * cell + 2 * margin
    dst = np.array(
        [
            [margin, margin],
            [margin + (FILES - 1) * cell, margin],
            [margin + (FILES - 1) * cell, margin + (RANKS - 1) * cell],
            [margin, margin + (RANKS - 1) * cell],
        ],
        dtype=np.float32,
    )
    src = np.asarray(corners, dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    # 边缘复制而非默认的常数填充：棋盘最左/最右一列的棋子贴着屏幕边，
    # 校正后窗口会伸到原图外，填黑会在这格里造出一块高反差区域，把棋子
    # 认成别的（实测新皮肤两个红车因此漏检）。复制边缘像素就没有这个突变。
    rectified = cv2.warpPerspective(image, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
    return rectified, cell, margin


def _dst_corners(cell):
    """矫正图的标准四角（TL, TR, BR, BL），与 rectify 一致。"""
    margin = cell
    return np.array(
        [
            [margin, margin],
            [margin + (FILES - 1) * cell, margin],
            [margin + (FILES - 1) * cell, margin + (RANKS - 1) * cell],
            [margin, margin + (RANKS - 1) * cell],
        ],
        dtype=np.float32,
    )


def detect_disc_blobs(rectified, warm_gap=15.0, bright=120, min_area_ratio=0.10):
    """检测「暖色亮圆盘」（棋子圆面）的连通域，返回质心列表 [(cx, cy, area), ...]。

    JJ象棋 主题里棋子是米黄圆盘（R-G 明显为正、亮度高），棋盘是灰绿、
    边框/装饰偏暗，借此把圆盘从背景里分出来。换主题（如深色圆盘）时
    可能一个都检不到，调用方应容忍空列表（退回未精修的网格）。
    """
    b = rectified[:, :, 0].astype(np.int16)
    g = rectified[:, :, 1].astype(np.int16)
    r = rectified[:, :, 2].astype(np.int16)
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    mask = (((r - g) > warm_gap) & (gray > bright)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    min_area = min_area_ratio * rectified.shape[0] * rectified.shape[1] / (
        (RANKS + 4) * (FILES + 4)
    )
    out = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        out.append((float(cents[i][0]), float(cents[i][1]), float(area)))
    return out


def _seed_axis(vals, cell, margin, count, tol=12.0, min_score=8):
    """一维搜寻等差数列 (origin, spacing)，使最多质心落在格线上。

    只搜 count 个格位组成的窗口（不是无限延长线）：整体平移一格的
    假设会错过棋盘另一侧的圆盘，借此把「真实原点」和「平移一格的等价解」
    区分开。返回 (origin, spacing) 或 None。
    """
    vals = np.asarray(vals, dtype=np.float64)
    if vals.size < min_score:
        return None
    best = None  # ((score, -residual), origin, spacing)
    ks = np.arange(count)

    def _scan(origins, spacings):
        nonlocal best
        for spacing in spacings:
            pos = origins[:, None] + ks[None, :] * spacing
            # d[v, o] = 第 v 个质心到第 o 组格线的最近距离
            d = np.abs(vals[:, None, None] - pos[None, :, :]).min(axis=2)
            for oi in range(len(origins)):
                dd = d[:, oi]
                inl = dd < tol
                score = int(inl.sum())
                if score >= min_score:
                    key = (score, -float(dd[inl].sum()))
                    if best is None or key > best[0]:
                        best = (key, float(origins[oi]), float(spacing))

    _scan(
        np.arange(margin - cell * 0.7, margin + cell * 0.7, 4.0),
        np.arange(cell * 0.80, cell * 1.20, 2.0),
    )
    if best is None:
        return None
    # 在粗搜最优附近细搜
    _, bo, bs = best
    _scan(
        np.arange(bo - 5.0, bo + 5.0, 1.0),
        np.arange(bs - 2.5, bs + 2.5, 0.5),
    )
    return best[1], best[2]


def fit_grid(rectified, cell, margin, min_matches=8, iters=8, max_shift=0.45,
             tol_residual=10.0):
    """用圆盘质心迭代拟合真实网格，返回 (x0, y0, cell_x, cell_y)。

    背景：用户标定的四角往往是棋盘外框，比 9x10 交叉点网格大约半个格子，
    直接按四角矫正会让采样点系统性偏离棋子圆心（实测偏移可达 0.6 格），
    墨迹亮度判占/空因此完全失效。本函数从圆盘质心反解真实网格：

        1. 一维搜索：x/y 各自寻找让最多质心落在格线上的 (原点, 间距)，
           得到接近真实的初始变换（容差收紧到 12px，整体平移一格的
           等价解会错过棋盘另一侧圆盘而被排除）；
        2. 二维迭代：以初始变换把圆盘贪心匹配到 90 个格点，最小二乘
           重解 (x = x0 + f*cell_x, y = y0 + r*cell_y)，迭代至收敛。

    返回 None 表示圆盘太少或拟合不收敛（调用方应退回粗略网格）。
    """
    blobs = detect_disc_blobs(rectified)
    if len(blobs) < min_matches:
        return None
    seed_x = _seed_axis([b[0] for b in blobs], cell, margin, FILES)
    seed_y = _seed_axis([b[1] for b in blobs], cell, margin, RANKS)
    if seed_x is None or seed_y is None:
        return None
    x0, cx = seed_x
    y0, cy = seed_y

    max_d2 = (max_shift * cell) ** 2
    matched = []
    res = 0.0
    for _ in range(iters):
        points = [
            (f, r, x0 + f * cx, y0 + r * cy)
            for r in range(RANKS)
            for f in range(FILES)
        ]
        cand = []
        for bi, (bx, by, _) in enumerate(blobs):
            for pi, (f, r, px, py) in enumerate(points):
                d2 = (bx - px) ** 2 + (by - py) ** 2
                if d2 <= max_d2:
                    cand.append((d2, bi, pi))
        cand.sort()
        used_b, used_p, matched = set(), set(), []
        for d2, bi, pi in cand:
            if bi in used_b or pi in used_p:
                continue
            used_b.add(bi)
            used_p.add(pi)
            bx, by, _ = blobs[bi]
            f, r, _, _ = points[pi]
            matched.append((f, r, bx, by))
        if len(matched) < min_matches:
            return None
        ax = np.array([[1.0, m[0]] for m in matched])
        sx = np.linalg.lstsq(ax, np.array([m[2] for m in matched]), rcond=None)[0]
        ay = np.array([[1.0, m[1]] for m in matched])
        sy = np.linalg.lstsq(ay, np.array([m[3] for m in matched]), rcond=None)[0]
        nx0, ncx = float(sx[0]), float(sx[1])
        ny0, ncy = float(sy[0]), float(sy[1])
        if not (0.5 * cell < ncx < 1.5 * cell and 0.5 * cell < ncy < 1.5 * cell):
            return None  # 尺度发散，放弃
        res = max(
            max(abs(m[2] - (nx0 + m[0] * ncx)) for m in matched),
            max(abs(m[3] - (ny0 + m[1] * ncy)) for m in matched),
        )
        converged = (
            abs(nx0 - x0) < 0.5 and abs(ny0 - y0) < 0.5
            and abs(ncx - cx) < 0.01 and abs(ncy - cy) < 0.01
        )
        x0, y0, cx, cy = nx0, ny0, ncx, ncy
        if converged:
            break

    # 收敛后剔除离群匹配对再拟合：高亮光晕/粘连会把个别圆盘质心拉偏
    # （实测单点偏差可达 35px），但多数匹配对仍精确，稳健化后采用。
    for _ in range(3):
        if len(matched) < min_matches:
            return None
        ax = np.array([[1.0, m[0]] for m in matched])
        sx = np.linalg.lstsq(ax, np.array([m[2] for m in matched]), rcond=None)[0]
        ay = np.array([[1.0, m[1]] for m in matched])
        sy = np.linalg.lstsq(ay, np.array([m[3] for m in matched]), rcond=None)[0]
        x0, cx = float(sx[0]), float(sx[1])
        y0, cy = float(sy[0]), float(sy[1])
        if not (0.5 * cell < cx < 1.5 * cell and 0.5 * cell < cy < 1.5 * cell):
            return None
        resids = [
            max(abs(m[2] - (x0 + m[0] * cx)), abs(m[3] - (y0 + m[1] * cy)))
            for m in matched
        ]
        med = float(np.median(resids))
        limit = max(12.0, 3.0 * med)
        kept = [m for m, rd in zip(matched, resids) if rd <= limit]
        if len(kept) == len(matched):
            matched = kept
            break
        matched = kept
    res = max(
        max(abs(m[2] - (x0 + m[0] * cx)) for m in matched),
        max(abs(m[3] - (y0 + m[1] * cy)) for m in matched),
    ) if matched else 1e9
    if res > tol_residual or len(matched) < min_matches:
        return None
    return (x0, y0, cx, cy)


def scale_corners(corners, ref_size, image_size):
    """把「标定坐标系」下的四角等比换算到当前图像的实际尺寸。

    为什么需要：board_corners 是在某一张截图上标定的，坐标属于那个分辨率。
    抓图时可以要求设备端降采样（PNG 编码从 3.4s 降到 1.6s），此时图像变小，
    四角必须按同一比例缩放，否则矫正会裁到棋盘外。手机换分辨率同样受益。

    ref_size / image_size 均为 (宽, 高)；两者相同则原样返回。
    """
    if not ref_size:
        return corners
    rw, rh = float(ref_size[0]), float(ref_size[1])
    iw, ih = float(image_size[0]), float(image_size[1])
    if rw <= 0 or rh <= 0 or (abs(rw - iw) < 1.0 and abs(rh - ih) < 1.0):
        return corners
    sx, sy = iw / rw, ih / rh
    return [[p[0] * sx, p[1] * sy] for p in corners]


def rectify_with_fit(image, corners, fit, cell=140):
    """按 fit_grid 的拟合结果修正四角，从原图重新矫正。

    fit = (x0, y0, cell_x, cell_y)（相对「按 corners 粗矫正后的图」的坐标系）。
    返回 (矫正图, cell, margin, 修正后的四角)——真实网格此后精确落在
    (margin + f*cell, margin + r*cell)，下游识别/追踪/标注全部受益。
    """
    rough, _, _ = rectify(image, corners, cell)
    M = cv2.getPerspectiveTransform(np.asarray(corners, dtype=np.float32), _dst_corners(cell))
    x0, y0, cx, cy = fit
    fitted = np.array(
        [
            [x0, y0],
            [x0 + (FILES - 1) * cx, y0],
            [x0 + (FILES - 1) * cx, y0 + (RANKS - 1) * cy],
            [x0, y0 + (RANKS - 1) * cy],
        ],
        dtype=np.float32,
    )
    # 拟合四角（粗矫正图坐标）→ 原图坐标
    Minv = np.linalg.inv(M.astype(np.float64))
    orig = cv2.perspectiveTransform(fitted.reshape(-1, 1, 2), Minv).reshape(-1, 2)
    rectified, _, margin = rectify(image, orig.astype(np.float32), cell)
    return rectified, cell, margin, orig


def refine_rectify(image, corners, cell=140, max_shift_frac=0.7):
    """矫正 + 网格精修一站式入口。

    返回 (矫正图, cell, margin, 修正四角, fit)。fit 为 None 表示精修失败
    （圆盘检测不到、拟合不收敛，或**修正幅度过大**），此时退回按原四角矫正。

    为什么限制修正幅度：精修的正常工作量是「把用户标的四角拉回真实网格」，
    典型是四角落在棋盘外框、比交叉点网格大约半格（0.5 格）。但拟合偶尔会锁到
    相邻一行（换阵营后界面布局变了、场上棋子分布不利），一路把网格推走整格
    ——实测黑方局被推了 1.02 格，整盘错行、两条底线全空。所以把上限定在
    0.7 格：半格内的修正照做，整格级的漂移直接判失败，用配置里的四角
    （那是人工标定过的可靠值）。
    """
    rect, cell, margin = rectify(image, corners, cell)
    fit = fit_grid(rect, cell, margin)
    if fit is None:
        return rect, cell, margin, np.asarray(corners, dtype=np.float32), None
    rect2, cell, margin, corners2 = rectify_with_fit(image, corners, fit, cell)
    src = np.asarray(corners, dtype=np.float32)
    orig_cell = float(np.linalg.norm(src[1] - src[0])) / (FILES - 1)
    shift = float(np.linalg.norm(np.asarray(corners2, dtype=np.float32) - src, axis=1).max())
    if orig_cell > 0 and shift > max_shift_frac * orig_cell:
        return rect, cell, margin, src, None
    return rect2, cell, margin, corners2, fit


def extract_templates(rectified, board, cell, margin, size=80):
    """从矫正后的初始局面图，按已知 board 提取每类棋子的模板。

    board: 10 行字符串（初始局面）。返回 {棋子字符: size x size x 3 彩色 BGR 图}。
    同名棋子（如两颗车、五颗兵）会取平均值，避免只保留最后一颗造成的偏差。
    保留颜色：兵/炮是暖色圆面，帅仕相马车是暗色圆面，靠颜色才能与木色棋盘区分。
    """
    acc = {}
    cnt = {}
    half = cell // 2
    for r in range(RANKS):
        for f in range(FILES):
            ch = board[r][f]
            if ch == " ":
                continue
            x = margin + f * cell
            y = margin + r * cell
            crop = rectified[y - half : y + half, x - half : x + half]
            crop = cv2.resize(crop, (size, size)).astype(np.float32)
            acc[ch] = acc.get(ch, 0.0) + crop
            cnt[ch] = cnt.get(ch, 0) + 1
    return {ch: (acc[ch] / cnt[ch]).astype(np.uint8) for ch in acc}


def detect_flip(rectified, cell, margin, threshold=15.0):
    """判断棋盘是否上下翻转（红方在上）。返回 True 表示红方在上，FEN 需垂直翻转。

    原理：红方棋子墨迹偏红（最暗像素 R-G 明显为正），黑方棋子墨迹近中性
    （R-G ≈ 0）。比较顶部底线与底部底线棋子的「红度」，上方明显更红 →
     红方在上。实测 JJ象棋 主题：红方底线 R-G≈+87、黑方≈+7，区分度极大。
    仅在初始局面（两条底线都有棋子）下调用。
    """
    rad = max(8, int(cell * 0.30))
    yy, xx = np.ogrid[: 2 * rad, : 2 * rad]
    mask = (xx - rad) ** 2 + (yy - rad) ** 2 <= rad * rad

    def _rank_ink_redness(rank):
        gs = []
        for f in range(FILES):
            x = margin + f * cell
            y = margin + rank * cell
            crop = rectified[y - rad : y + rad, x - rad : x + rad]
            b = crop[:, :, 0].astype(np.float32)
            g = crop[:, :, 1].astype(np.float32)
            r = crop[:, :, 2].astype(np.float32)
            lum = (r + g + b)[mask]
            k = max(1, int(mask.sum() * 0.15))
            idx = np.argsort(lum)[:k]
            gs.append(float(r[mask][idx].mean()) - float(g[mask][idx].mean()))
        return float(np.mean(gs))

    top = _rank_ink_redness(0)
    bottom = _rank_ink_redness(RANKS - 1)
    return (top - bottom) > threshold


def _template_filename(ch):
    """模板文件名：red_R.png（红方大写）/ black_r.png（黑方小写）。

    用前缀区分颜色，避免 Windows 大小写不敏感文件系统上 R.png/r.png 互相覆盖。
    """
    prefix = "red" if ch.isupper() else "black"
    return f"{prefix}_{ch}.png"


def save_templates(templates, templates_dir):
    """把 {棋子字符: 灰度模板} 存到目录，文件名用 red_/black_ 前缀区分颜色。"""
    import os

    os.makedirs(templates_dir, exist_ok=True)
    for ch, tmpl in templates.items():
        fname = _template_filename(ch)
        cv2.imwrite(os.path.join(templates_dir, fname), tmpl)


def _imread_any(path):
    """读图，兼容含中文/特殊字符的路径（cv2.imread 在 Windows 上会失败）。"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except (OSError, ValueError):
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


PIECE_CHARS = set("KABNCRPkabncrp")


def load_templates(templates_dir, size=80):
    """从目录加载棋子模板。文件名 red_R.png / black_r.png → 字符 R / r。

    目录不存在时返回空 dict（由调用方给出「请先生成模板」的友好提示），
    不抛 FileNotFoundError。
    """
    import os

    if not os.path.isdir(templates_dir):
        return {}
    templates = {}
    for fname in os.listdir(templates_dir):
        if not fname.endswith(".png"):
            continue
        stem = os.path.splitext(fname)[0]  # 如 "red_R" / "black_r"
        if "_" not in stem:
            continue
        ch = stem.split("_", 1)[1]  # "R" / "r"
        if ch not in PIECE_CHARS:
            continue  # 只认合法棋子字符，忽略目录里的杂图
        img = _imread_any(os.path.join(templates_dir, fname))
        if img is None:
            continue
        templates[ch] = cv2.resize(img, (size, size))
    return templates
