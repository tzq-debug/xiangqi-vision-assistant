"""矫正后的棋盘图 → 逐格识别 → 棋盘。"""

import cv2
import numpy as np

from .board import MAX_PIECES, sanitize_positions
from .calibrate import FILES, RANKS
from .occupancy import occupancy_map

TEMPLATE_SIZE = 80

# 模板匹配对网格偏移极其敏感：矫正图的网格来自每帧的圆盘拟合，实测帧间
# 有 1~4 像素抖动（矫正空间里 3~14 像素），只偏 2 像素就能把匹配分从 0.95
# 打到 0.55——真机第 13 回合就是这样把「将」认丢、把「车」认成 4 个的。
# 所以不按格硬切，而是多留一圈余量做小范围滑动匹配，取最高分。
# 实测：同一帧均分 0.78 → 0.94，格子偏移 5 像素以内都能对上。
SEARCH_PAD_FRAC = 0.11   # 每边多留 0.11*cell 的搜索余量
MATCH_THRESHOLD = 0.41   # 彩色模板匹配高于此分即为棋子
WARM_MIN_SCORE = 0.25    # 兵/炮等暖色圆面在暗角仍可接受的最低分
WARM_RGB_GAP = 0.0       # 圆心 R-G 高于此判为暖色（兵/炮圆面偏红，棋盘偏绿/中性）
WARM_PIECES = {"P", "p", "C", "c"}  # 暖色规则只对兵/炮生效，避免暖色色带误判士相马车
OFFICERS = set("KABNkabn")          # 士相马车将帅：复杂字形，匹配分更高才可信
OFFICER_MIN_SCORE = 0.55            # 车炮兵卒圆面简单，沿用 MATCH_THRESHOLD

REPAIR_MAX_LOSS = 0.12   # 纠错时允许「次选分比首选低多少」，超过就不动
REPAIR_TRUST = 0.80      # 首选分高于此的格子视为有把握，绝不改动
REPAIR_PASSES = 4        # 最多纠正几处，避免连锁误改

# 墨迹红度（格内最暗 15% 像素的 R-G 均值）：红子约 +130，黑子约 0，空位为负。
# 用它纠正大小写——模板含背景，棋盘最上/最下两行外侧背景更亮，红子容易被
# 认成黑子（实测黑方局红马读成 n、红车读成 r，随后被数量纠错当成多余的删掉，
# 整条底线丢三子）。这条判据只看棋子墨色，与背景无关。
RED_INK_MIN = 60.0

# 占空图判「有子」但模板最高分低于此值时，判为光点/残影而不是棋子。
# 实测真棋子最低 0.55（将帅），落点光点 0.42。
GHOST_MAX_SCORE = 0.50
# 超量棋子要「改判」而不是删掉时，次选分必须够高：真棋子错认时次选通常 0.8+，
# 落点光点那种假棋子次选分很低，仍应直接清掉。
REBALANCE_MIN_SCORE = 0.70


def _ink_redness(rectified, cell, margin, r, f):
    """格中心圆内「墨迹」（最暗 15% 像素）的 R-G 均值。"""
    rad = max(8, int(cell * 0.30))
    x = margin + f * cell
    y = margin + r * cell
    crop = rectified[y - rad : y + rad, x - rad : x + rad].astype(np.float32)
    h, w = crop.shape[:2]
    if h < 2 or w < 2:
        return 0.0
    yy, xx = np.ogrid[:h, :w]
    mask = (xx - rad) ** 2 + (yy - rad) ** 2 <= rad * rad
    vals = crop.sum(axis=2)[mask]
    if vals.size == 0:
        return 0.0
    k = max(1, int(vals.size * 0.15))
    idx = np.argpartition(vals, k - 1)[:k]
    return float((crop[:, :, 2][mask][idx] - crop[:, :, 1][mask][idx]).mean())


def _fix_color(ch, rectified, cell, margin, r, f):
    """按墨迹红度纠正红/黑大小写。"""
    if ch == " ":
        return ch
    is_red = ch.isupper()
    red = _ink_redness(rectified, cell, margin, r, f) >= RED_INK_MIN
    if red == is_red:
        return ch
    return ch.upper() if red else ch.lower()


def _prep(img):
    """匹配前统一轻度模糊。

    棋子圆盘上有光斑、阴影、抗锯齿边缘，这些都是局部高频噪声，会把
    TM_CCOEFF_NORMED 的分数拉低：同一颗黑马在阴影格原本只有 0.51（低于
    阈值被漏检），模糊后 0.69；而空位仍是 0.02~0.09，分离反而更开。
    """
    return cv2.GaussianBlur(img, (5, 5), 0)


def _scores(window, templates, cell):
    """返回 ({字符: 匹配分}, 圆心是否暖色)。

    window 比一格略大（见 SEARCH_PAD_FRAC），缩放到「一格 = TEMPLATE_SIZE」
    后与整块模板做滑动匹配，取最高分：这样网格偏移几个像素也能对上。
    """
    h, w = window.shape[:2]
    cx, cy = w // 2, h // 2
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (cx, cy), int(cell * 0.30), 255, -1)
    b, g, r = cv2.mean(window, mask=mask)[:3]
    warm = (r - g) > WARM_RGB_GAP

    size = max(TEMPLATE_SIZE, int(round(w * TEMPLATE_SIZE / float(cell))))
    small = _prep(cv2.resize(window, (size, size)))
    out = {}
    for ch, tmpl in templates.items():
        res = cv2.matchTemplate(small, _prep(tmpl), cv2.TM_CCOEFF_NORMED)
        out[ch] = float(res.max())
    return out, warm


def _decide(scores, warm):
    """按分数与阈值给出 (字符, 置信度)；空位返回 " "。"""
    if not scores:
        return "?", 0.0
    best = max(scores, key=scores.get)
    best_score = scores[best]
    # 士相马车将帅字形复杂，模板匹配分可靠（实测 0.63~1.00），
    # 空位偶尔会以 0.45 左右误配成此类，故单独提高阈值；
    # 车炮兵卒圆面简单、匹配分散，沿用较低的 MATCH_THRESHOLD。
    threshold = OFFICER_MIN_SCORE if best in OFFICERS else MATCH_THRESHOLD
    if best_score >= threshold:
        return best, best_score
    if warm and best in WARM_PIECES and best_score >= WARM_MIN_SCORE:
        return best, best_score
    return " ", best_score


def classify_cell(window, templates, cell):
    """识别单个交叉点：返回 (字符, 置信度)。window 需含搜索余量。"""
    scores, warm = _scores(window, templates, cell)
    return _decide(scores, warm)


def repair_counts(board, score_maps, max_loss=REPAIR_MAX_LOSS, passes=REPAIR_PASSES):
    """用「棋子数量上限」纠正个别认错的棋子，返回新棋盘。

    落子高亮会给刚走的那颗棋子套一圈径向光晕，圆面被洗白、笔画对比度下降，
    模板匹配可能把黑车认成黑象（真机实测：黑车 (4,7) 灰度匹配 象 0.568 >
    车 0.525，而三次独立截图都一样，不是偶发）。这类误认无法靠预处理消除
    （试过模糊、边缘、局部亮度归一化，都没把「车」救回来）。

    但棋子数量是硬约束：一方最多 2 象 2 车。当某类棋子超限、而另一类亏空时，
    把「最没把握」的那颗改成次选字符——只要次选分不比首选低太多（≤ max_loss），
    且改动只发生在同色棋子之间。
    """
    rows = [list(r) for r in board]
    limits = {}
    for ch, cap in MAX_PIECES.items():
        limits[ch] = cap
        limits[ch.lower()] = cap

    changed = True
    for _ in range(passes):
        if not changed:
            break
        changed = False
        counts = {}
        for row in rows:
            for ch in row:
                if ch != " ":
                    counts[ch] = counts.get(ch, 0) + 1
        over = [ch for ch, n in counts.items() if n > limits.get(ch, 99)]
        if not over:
            break
        # 候选 = (首选分, 损失, r, f, from, to)；按「首选分最低」再「损失最小」排序。
        # 先看首选分：认错的那格分数天然偏低（实测黑车被认成象只有 0.57，
        # 而真象有 0.95），若按「损失最小」挑就会去改真象那一格，越改越错。
        fixes = []
        for ch in over:
            for r in range(RANKS):
                for f in range(FILES):
                    if rows[r][f] != ch:
                        continue
                    sc = score_maps[r * FILES + f]
                    cur = sc.get(ch, 0.0)
                    if cur >= REPAIR_TRUST:
                        continue      # 有把握的格子不动
                    for alt, val in sc.items():
                        if alt == ch or alt == " ":
                            continue
                        if alt.isupper() != ch.isupper():
                            continue      # 不许把红子改成黑子
                        if counts.get(alt, 0) >= limits.get(alt, 99):
                            continue      # 改了也还是超限/不亏空
                        loss = cur - val
                        if loss <= max_loss:
                            fixes.append((cur, loss, r, f, ch, alt))
        if not fixes:
            break             # 没有可信的改法就不动，交给调用方告警
        fixes.sort()
        _, _, r, f, ch, alt = fixes[0]
        rows[r][f] = alt
        changed = True
    return ["".join(row) for row in rows]


def recognize_board(rectified, templates, cell, margin, repair=True):
    """识别整盘。返回 (board, confidences)。

    repair=True 时对「棋子数量不合理」的结果做一次纠正（见 repair_counts）。
    """
    board_rows, confidences, score_maps = [], [], []
    half = cell // 2
    pad = int(round(cell * SEARCH_PAD_FRAC))
    # 有没有棋子由占空图（亮度）决定，模板只负责「是哪一颗」：
    # 模板匹配遇到邻格棋子容易蹭出高分而凭空造子（实测空位被认成炮），
    # 亮度判定则与字母/皮肤无关，两道判据合起来才稳。
    occ = occupancy_map(rectified, cell, margin)
    for r in range(RANKS):
        row = ""
        for f in range(FILES):
            x = margin + f * cell
            y = margin + r * cell
            window = rectified[
                max(0, y - half - pad) : y + half + pad,
                max(0, x - half - pad) : x + half + pad,
            ]
            if not occ[r][f]:
                row += " "
                confidences.append(0.0)
                score_maps.append({})
                continue
            scores, warm = _scores(window, templates, cell)
            ch, conf = _decide(scores, warm)
            if ch == " " and scores:
                # 占空图说有子，但模板也说不出是什么子 → 只有当「最高分还像样」
                # 才补一个字符。实测「上一步落点」的十字光点最高分只有 0.42
                # （真棋子 0.55~0.89），它会被硬填成某颗棋子，在红帅旁边造出
                # 一辆非法黑车、FEN 直接不合法。宁可在这一格留空。
                best = max(scores, key=scores.get)
                if scores[best] >= GHOST_MAX_SCORE:
                    ch = best
            ch = _fix_color(ch, rectified, cell, margin, r, f)
            row += ch
            confidences.append(conf)
            score_maps.append(scores)
        board_rows.append(row)
    if repair:
        board_rows = repair_counts(board_rows, score_maps)
    # repair 会在同色格之间互改，可能把士挪到九宫外，所以合法性清理放最后
    board_rows, _ = sanitize_positions(board_rows)
    board_rows = _enforce_counts(
        board_rows, lambda r, f: confidences[r * FILES + f], score_maps
    )
    board_rows = _ensure_kings(board_rows, occ, score_maps)
    return board_rows, confidences


def _enforce_counts(board, weight, score_maps=None):
    """某类棋子超出数量上限时，清掉 weight 最低的多余格。

    「上一步落点」的白色十字光点会让空格被判成有子，模板又给出一个低分字符：
    实测同一格一会儿读成「将」、一会儿读成「马」，盘面于是出现 2 个将 / 3 个马，
    FEN 非法、算棋被静默跳过——表现就是「走完这一步不给方案了」。数量是硬约束，
    多出来的那颗一定是分最低的那颗，清掉比留着安全（少一颗只会让评分略变，
    多一颗会让整个 FEN 不合法）。
    """
    rows = [list(row) for row in board]
    for ch, cap in MAX_PIECES.items():
        for c in (ch, ch.lower()):
            group = [
                (weight(r, f), r, f)
                for r in range(RANKS)
                for f in range(FILES)
                if rows[r][f] == c
            ]
            if len(group) <= cap:
                continue
            group.sort()
            for _, r, f in group[: len(group) - cap]:
                # 先试「改判」再试「删除」。实测同一方两个车被认成两个马：
                # 直接把多出来的马删掉，盘面就凭空少了一辆车（用户一眼看出）；
                # 而那一格的次高分往往正是「车」（0.90 vs 0.91），改判即可还原。
                target = _deficit_target(rows, c, score_maps, r, f)
                rows[r][f] = target if target else " "
    return ["".join(row) for row in rows]


def majority_board(rectifieds, templates, cell, margin):
    """对多帧矫正图逐格识别后按多数票决，返回 (board, 每格票数)。

    单帧识别可能被高亮/反光/残影干扰，多帧投票利用「噪声不持续」的特点
    得到更稳的盘面，用于追踪失联后的重新识别。
    """
    boards = [recognize_board(rect, templates, cell, margin)[0] for rect in rectifieds]
    if not boards:
        return [], []
    out_rows, votes = [], []
    for r in range(RANKS):
        row, vrow = "", []
        for f in range(FILES):
            chars = [b[r][f] for b in boards]
            best = max(set(chars), key=chars.count)
            row += best
            vrow.append(chars.count(best))
        out_rows.append(row)
        votes.append(vrow)
    out_rows, _ = sanitize_positions(out_rows)
    out_rows = _enforce_counts(out_rows, lambda r, f: votes[r][f])
    return out_rows, votes

def _deficit_target(rows, ch, score_maps, r, f):
    """某格该改判成什么：同方「还没到数量上限」的棋子中，该格分数最高的那个。

    只在真正超量时调用（超量本身就说明识别错了）。找不到合适的返回 None，
    调用方会退回「清空」。
    """
    if not score_maps:
        return None
    is_red = ch.isupper()
    counts = {}
    for rr in range(RANKS):
        for ff in range(FILES):
            c2 = rows[rr][ff]
            if c2 != " " and c2.isupper() == is_red:
                counts[c2] = counts.get(c2, 0) + 1

    idx = r * FILES + f
    if idx >= len(score_maps):
        return None
    scores = score_maps[idx] or {}

    best, best_score = None, 0.0
    for base, cap in MAX_PIECES.items():
        cand = base if is_red else base.lower()
        if cand == ch:
            continue
        if counts.get(cand, 0) >= cap:
            continue
        sc = scores.get(cand, 0.0)
        if sc > best_score and sc >= REBALANCE_MIN_SCORE:
            best, best_score = cand, sc
    return best

def _ensure_kings(rows, occ, score_maps):
    """返回新的一维行字符串列表（原传入的是不可变字符串，不能原地改）。"""
    grid = [list(row) for row in rows]
    """将/帅各只有一个，缺了整盘就没法算棋（FEN 直接非法、引擎会拒收）。

    实测换皮肤重提模板后偶发「将帅都读成空」——占空和模板分都够，只是被外圈
    判据挡掉。它俩是全盘最确定的两个子，缺失时按模板分在「判有子但没认出
    字符」的格里补回来。
    """
    for king in ("K", "k"):
        if any(ch == king for row in grid for ch in row):
            continue
        best = None
        for r in range(RANKS):
            for f in range(FILES):
                # 不要求 occ 为真：换皮肤后占空偶尔把将帅判成空（亮度/暖度
                # 都有余量，只是切分点偏了一点），而它俩的模板分是全盘最高的。
                if grid[r][f] != " ":
                    continue
                idx = r * FILES + f
                sc = (score_maps[idx] or {}).get(king, 0.0) if idx < len(score_maps) else 0.0
                if best is None or sc > best[0]:
                    best = (sc, r, f)
        if best and best[0] >= 0.65:
            grid[best[1]][best[2]] = king
    return ["".join(row) for row in grid]
