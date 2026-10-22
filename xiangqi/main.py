"""主循环：采集 → 矫正 → 追踪走子 → 算棋 → 展示。

与旧版实现的差异：
    * 走子追踪交给 tracking.BoardTracker（画面差分 + 墨迹亮度 + 合法性校验
      + 失联自动重同步），不再自己在主循环里拼走子，也不会再「等待稳定」卡死；
    * 开局自动判定棋盘方向（红方在上/在下都认），不再强依赖 config 的 flip_board；
    * 引擎崩溃自动重启重试，FEN 先做 sanity 检查再送入；
    * 截屏失败明确报错重试，不会误用上一张旧图；
    * 可选 --show 实时窗口（tkinter），默认只写 bestmove.png，无显示环境也能跑。

用法：
    python -m xiangqi.main [--config config.json] [--interval 0.4]
                           [--show] [--debug] [--movetime 3000]

我方是谁（config 的两种写法，player_side 优先）：
    "player_side": "bottom"  截屏下方是我方（默认；JJ象棋 自己的棋子在下方）
    "player_side": "top"     截屏上方是我方
    "player_color": "red"    直接指定颜色（兼容旧配置）
屏幕方向（红方在上/在下）由启动时自动判定，所以 player_side 与红黑无关：
红在上时「下方」就是黑方，红在下时「下方」就是红方。

依赖（真机）：手机开 USB 调试、hdc 路径正确、engine/pikafish.exe 存在、
templates/ 已用初始局面截图生成（python -m xiangqi.retemplate <截图>）。
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

from . import capture
from .board import (
    FILE_COUNT,
    STARTING_BOARD,
    board_diff as _board_diff,
    board_problems,
    board_to_fen,
    flip_board,
    is_new_game_restart,
    material_jump,
    sanitize_positions,
)
from .calibrate import RANKS, load_templates, rectify, refine_rectify, scale_corners
from .display import LiveView, draw_move, draw_void
from .turn_detect import active_side, region_diff, to_fen_side
from .engine import PikafishEngine
from .moves import is_legal as is_legal_move, side_of
from .notation import move_to_chinese, parse_move
from .recognize import majority_board, recognize_board
from .tracking import BoardTracker

REQUIRED_KEYS = (
    "hdc_path",
    "remote_path",
    "local_path",
    "board_corners",
    "templates_dir",
    "engine_path",
)

# 各类棋子每方数量上限（FEN sanity 检查用）
PIECE_LIMITS = {"K": 1, "A": 2, "B": 2, "N": 2, "R": 2, "C": 2, "P": 5}

# 配置里的相对路径按「配置文件所在目录」解析，这样从任何目录启动都能跑
PATH_KEYS = ("templates_dir", "engine_path", "local_path")

# 「我方在截屏的哪一边」（与红黑无关，颜色由屏幕方向推出）
SIDE_POSITIONS = ("bottom", "top")


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve_paths(cfg, config_path):
    """把配置里的相对路径解析成绝对路径（基准 = 配置文件所在目录）。

    hdc_path 单独处理：它既可能是绝对路径，也可能只是 PATH 里的命令名
    （"hdc"），只有确实像路径时才解析，避免把命令名拼成不存在的路径。
    """
    base = os.path.dirname(os.path.abspath(config_path))
    for key in PATH_KEYS:
        value = cfg.get(key)
        if isinstance(value, str) and value and not os.path.isabs(value):
            cfg[key] = os.path.normpath(os.path.join(base, value))
    hdc = cfg.get("hdc_path")
    if isinstance(hdc, str) and ("/" in hdc or "\\" in hdc) and not os.path.isabs(hdc):
        cfg["hdc_path"] = os.path.normpath(os.path.join(base, hdc))
    return cfg


def validate_config(cfg):
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise SystemExit(
            "config.json 缺少字段: " + ", ".join(missing)
            + "\n请参考 config.example.json，或先运行校准：python -m xiangqi.calibrate_cli <初始局面截图>"
        )
    corners = cfg["board_corners"]
    if not (isinstance(corners, list) and len(corners) == 4):
        raise SystemExit("config.json 的 board_corners 必须是四个点 [[x,y],...]")
    if any(abs(c[0]) + abs(c[1]) == 0 for c in corners):
        raise SystemExit("config.json 的 board_corners 还是占位的 [0,0]，请先运行校准：python -m xiangqi.calibrate_cli <初始局面截图>")
    pos = cfg.get("player_side")
    if pos is not None and str(pos).strip().lower() not in SIDE_POSITIONS:
        raise SystemExit(
            f"config.json 的 player_side 只能是 {'/'.join(SIDE_POSITIONS)}（当前: {pos!r}）\n"
            "bottom = 截屏下方是我方（默认），top = 上方是我方。"
        )
    size = cfg.get("screen_size")
    if size is not None and not (
        isinstance(size, (list, tuple)) and len(size) == 2
        and all(isinstance(v, (int, float)) and v > 0 for v in size)
    ):
        raise SystemExit("config.json 的 screen_size 必须是 [宽, 高]（标定 board_corners 时那张截图的尺寸）")
    scale = cfg.get("capture_scale")
    if scale is not None:
        try:
            scale = float(scale)
        except (TypeError, ValueError):
            raise SystemExit(f"config.json 的 capture_scale 必须是数字（当前: {cfg.get('capture_scale')!r}）")
        if not 0.2 <= scale <= 1.0:
            raise SystemExit(f"config.json 的 capture_scale 应在 0.2~1.0 之间（当前: {scale}）")
    return cfg


def fen_is_sane(fen):
    """FEN 基本合理性：将/帅各一个、每方每类棋子不超上限、总数不超 32。"""
    try:
        placement = fen.split()[0]
    except IndexError:
        return False
    if placement.count("K") != 1 or placement.count("k") != 1:
        return False
    counts = {}
    for ch in placement:
        if ch == "/":
            continue
        if ch.isdigit():
            continue
        counts[ch] = counts.get(ch, 0) + 1
    if sum(counts.values()) > 32:
        return False
    for ch, n in counts.items():
        if n > PIECE_LIMITS.get(ch.upper(), 0):
            return False
    return True


def detect_orientation(rect, templates, cell, margin, prefer_flip=None):
    """对比标准初始局面，自动判定屏幕方向。

    返回 (board, flip, matches)：识别结果按 flip 换算成标准坐标后与
    STARTING_BOARD 比较，取匹配更多的一方；平局时按 prefer_flip。
    不是开局（匹配不足 85）时改用帅/将位置判定：红帅在屏幕上半 →
    红方在上（flip=True）。board 一律返回标准坐标盘面。
    """
    raw, _ = recognize_board(rect, templates, cell, margin)
    best = None
    for flip in (False, True):
        std = flip_board(raw) if flip else raw
        matches = sum(
            1
            for r in range(RANKS)
            for f in range(9)
            if std[r][f] == STARTING_BOARD[r][f]
        )
        if best is None or matches > best[2] or (matches == best[2] and flip == prefer_flip):
            best = (std, flip, matches)
    if best[2] >= 85:
        return best
    # 非开局：按帅/将位置判定方向。红帅在屏幕上半 → 红方在上（flip=True）。
    king_rows = sorted(r for r in range(RANKS) for f in range(9) if raw[r][f] == "K")
    if king_rows:
        flip = king_rows[len(king_rows) // 2] < RANKS // 2
        return (flip_board(raw) if flip else raw, flip, best[2])
    # 那一帧恰好没认到红帅（走子动画、被遮挡）时用黑将反推：将在上半 → 黑在上
    black_rows = sorted(r for r in range(RANKS) for f in range(9) if raw[r][f] == "k")
    if black_rows:
        flip = black_rows[len(black_rows) // 2] >= RANKS // 2
        return (flip_board(raw) if flip else raw, flip, best[2])
    return best


def print_board(board, prefix="  "):
    for row in board:
        print(prefix + row)


def check_engine_move(board, side, move):
    """校验引擎返回的走法：合法返回 None，否则返回不可用原因。

    引擎理论上只会给「走子方」的走法，但 FEN 拼错、引擎状态残留、或误把
    ponder 走法当成主变，都会让画出来的箭头指向一颗不该动的棋子。宁可
    不画，也不要画一条自信的错箭头。
    """
    try:
        (fr, ff), (tr, tf) = parse_move(move)
    except (ValueError, IndexError, TypeError):
        return "坐标无法解析"
    ranks, files = len(board), len(board[0])
    if not (0 <= fr < ranks and 0 <= tr < ranks and 0 <= ff < files and 0 <= tf < files):
        return "坐标越界"
    piece = board[fr][ff]
    if piece == " ":
        return "起点没有棋子"
    if side_of(piece) != side:
        return "起点不是该走子方的棋子"
    if not is_legal_move(board, side, ((fr, ff), (tr, tf))):
        return "不符合行棋规则或会送将"
    return None


def side_names(side):
    return "红" if side == "w" else "黑"


def resolve_my_side(cfg, flip):
    """确定「我方」是红(w)还是黑(b)。

    优先用 player_side："bottom" = 截屏下方是我方，"top" = 上方是我方。
    这是玩家真正感知到的位置，不依赖「红方在下」这类隐含约定——屏幕方向
    flip 是启动时自动判定的，所以无论红在上还是红在下都能推出正确颜色。

    未配置 player_side 时才退回 player_color（"red"/"black"）；两者都没配
    则默认「截屏下方是我方」。

    flip=True 表示屏幕上方是红方，此时：下方=黑，上方=红。
    """
    pos = str(cfg.get("player_side") or "").strip().lower()
    if pos in SIDE_POSITIONS:
        bottom_side = "b" if flip else "w"
        if pos == "bottom":
            return bottom_side
        return "w" if bottom_side == "b" else "b"
    color = str(cfg.get("player_color") or "").strip().lower()
    if color:
        return "b" if color.startswith("b") else "w"
    return "b" if flip else "w"


def capture_size(cfg):
    """设备端抓图的目标尺寸；None 表示按手机原生分辨率抓。

    降采样能省掉大半 PNG 编码时间（实测 3.0s → 1.6s），代价是四角要等比换算，
    由 scale_corners 处理。screen_size 是标定 board_corners 时的截图尺寸。
    """
    ref = cfg.get("screen_size")
    scale = cfg.get("capture_scale")
    if not ref or not scale:
        return None
    scale = float(scale)
    if scale >= 1.0:
        return None
    return (int(round(ref[0] * scale)), int(round(ref[1] * scale)))


def _default_grab(cfg):
    size = capture_size(cfg)

    def _grab():
        return capture.grab(cfg["hdc_path"], cfg["remote_path"], cfg["local_path"], size=size)

    return _grab


def _read_image(path):
    """读截图，兼容中文路径。

    cv2.imread 在 Windows 上处理非 ASCII 路径会失败（并往 stderr 打警告），
    所以先用 np.fromfile + imdecode，失败再退回 imread。
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except (OSError, ValueError):
        data = None
    if data is not None and data.size:
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    return cv2.imread(path)


def detect_first_mover(cfg, img, grab_fn, my_side, settle_s=1.2):
    """靠「谁的头像计时器在跳」判断当前轮到哪一方走，判不出返回 None。

    比让用户手填 start_side 可靠：程序常在残局中途接管，这时红方可能已经
    走过一步，硬编码「红先」会一直等对方、永远不出手。JJ象棋 当前行棋方的
    头像计时器在跳，抓两帧比对应区域的像素差就能分出来。无时限对局两边都
    不跳，返回 None，由调用方按「开局红先／中局我方先」兜底。
    """
    regions = cfg.get("timer_regions") or {}
    self_reg, opp_reg = regions.get("self"), regions.get("opponent")
    ref = cfg.get("screen_size")
    if not (self_reg and opp_reg and ref):
        return None
    try:
        time.sleep(settle_s)
        img2 = _read_image(grab_fn())
    except Exception:  # noqa: BLE001 - 判先后失败不该影响启动
        return None
    if img2 is None or img2.shape[:2] != img.shape[:2]:
        return None

    h, w = img.shape[:2]
    sx, sy = w / float(ref[0]), h / float(ref[1])

    def _scaled(reg):
        x, y, rw, rh = reg
        return (int(x * sx), int(y * sy), max(1, int(rw * sx)), max(1, int(rh * sy)))

    d_self = region_diff(img, img2, _scaled(self_reg))
    d_opp = region_diff(img, img2, _scaled(opp_reg))
    side = to_fen_side(active_side(d_self, d_opp), "red" if my_side == "w" else "black")
    if side:
        print(f"判先后：己方计时器变化 {d_self:.1f}，对方 {d_opp:.1f} → 轮到{'红' if side == 'w' else '黑'}方")
    return side


def main(
    config_path="config.json",
    *,
    max_iterations=None,
    grab=None,
    sleep=None,
    engine_factory=None,
    show=False,
    debug=False,
    out_path="bestmove.png",
    interval=None,
    movetime_ms=None,
    on_state=None,
    resync_event=None,
):
    """主循环。返回运行摘要 dict（供测试/调用方检查）。

    grab / sleep / engine_factory 可注入（测试用）；show 开实时窗口；
    debug 打印每帧追踪细节。max_iterations 限制循环轮数（测试用）。
    interval / movetime_ms 覆盖配置文件里的值（命令行用，不落盘）。
    """
    global _RUN_STARTED
    _RUN_STARTED = time.time()
    # 供 Web/UI 层订阅的运行状态（on_state 回调，每帧变化时推一次）
    state = {
        "status": "starting",        # starting/waiting/ready/frozen/error
        "message": "正在建立盘面…",
        "board": [],
        "side": None,                # 轮到哪方走 w/b
        "my_side": None,
        "flip": False,
        "move": None,                # 本次建议 {from,to,uci,chinese,score}
        "last_move": None,           # 最近一步 {chinese,uci,board_before}
        "frames": 0,
    }

    def push(**kw):
        state.update(kw)
        # 帧号/时间戳让界面能判断「后端还在推进」还是真卡了（原来是空操作，
        # 界面永远看到 frames=0，没法区分）。
        state["frames"] = iterations
        state["updated_at"] = time.time()
        if on_state is None:
            return
        try:
            snapshot = dict(state)
            snapshot["board"] = [row for row in state["board"]] if state["board"] else []
            on_state(snapshot)
        except Exception:  # noqa: BLE001 - UI 推送失败不该影响主循环
            pass


    cfg = resolve_paths(validate_config(load_config(config_path)), config_path)
    sleep = sleep or time.sleep
    grab = grab or _default_grab(cfg)

    # ---- 引擎（带重试启动）----
    engine = None
    last_engine_error = None
    for attempt in range(3):
        try:
            engine = (engine_factory or (lambda: PikafishEngine(
                cfg["engine_path"],
                cfg.get("engine_threads", 1),
                cfg.get("engine_hash_mb", 16),
            )))()
            engine.start()
            break
        except Exception as e:  # noqa: BLE001 - 启动失败要重试，最后统一报错
            last_engine_error = e
            engine = None
            print(f"引擎启动失败（第 {attempt + 1}/3 次）: {e}")
            time.sleep(1.0)
    if engine is None:
        raise SystemExit(f"引擎无法启动: {last_engine_error}\n请检查 config.json 的 engine_path（当前: {cfg['engine_path']}）")

    templates = load_templates(cfg["templates_dir"])
    if not templates:
        engine.quit()
        raise SystemExit(
            f"模板目录 {cfg['templates_dir']} 里没有可用模板。\n"
            "请用初始局面截图生成：python -m xiangqi.retemplate <截图路径>"
        )

    # 我方颜色要等启动时判定出屏幕方向（flip）后才能确定，见 resolve_my_side
    my_side = None
    interval = float(cfg.get("poll_interval", 0.4) if interval is None else interval)
    movetime_ms = int(cfg.get("movetime_ms", 3000) if movetime_ms is None else movetime_ms)
    view = None
    if show:
        try:
            view = LiveView()
        except Exception as e:  # noqa: BLE001 - 无显示环境时退化为只写文件
            print(f"实时窗口不可用（{e}），改为只写 {out_path}")

    summary = {
        "board": None,
        "side": None,
        "my_side": None,
        "flip": None,
        "grid_refined": False,
        "moves": [],
        "searches": 0,
        "resyncs": 0,
        "frames": 0,
        "quit_reason": "unknown",
    }

    tracker = None
    guard = TrustGuard()
    mguard = MaterialGuard()
    # 生效的四角。坐标系跟着实际图像尺寸走：screen_size 是标定 board_corners
    # 时的截图尺寸，抓图降采样或手机换分辨率时按比例换算，矫正才不跑偏。
    corner_size = tuple(cfg["screen_size"]) if cfg.get("screen_size") else ()
    corners = [list(p) for p in cfg["board_corners"]]
    grid_refined = False
    grid_next_retry = 0      # 精修失败后，下次允许重试的帧号
    last_fen = None
    last_move_at = None   # 最近一次识别到走子的时刻，用于统计「走完到出建议」的延迟
    iterations = 0
    try:
        while True:
            if max_iterations is not None and iterations >= max_iterations:
                summary["quit_reason"] = "max_iterations"
                break
            iterations += 1
            summary["frames"] = iterations
            guard.tick()

            # ---- 采集 ----
            try:
                local = grab()
                img = _read_image(local)
                if img is None:
                    print(f"读取截图失败: {local}（设备是否还在录屏/截图？）")
                    sleep(interval)
                    continue
            except capture.CaptureError as e:
                print(f"截屏失败: {e}")
                if not capture.device_connected(cfg["hdc_path"]):
                    print("提示：hdc 看不到设备，请检查 USB 连接与调试授权。")
                sleep(interval)
                continue

            # ---- 人工请求重新识别（界面上点了「重新识别」按钮）----
            # 程序自己判断不出来的偏差，人来兜底最快：抛掉当前盘面与追踪状态，
            # 下一帧按「重新建盘」的流程整体重识别一遍。
            if resync_event is not None and resync_event.is_set():
                resync_event.clear()
                tracker = None
                push(status="starting", message="正在重新识别盘面…")
                print(f"[{_clock()}] 收到界面请求：重新识别盘面")
                sleep(interval)
                continue

            # ---- 矫正（首帧尝试网格精修：标定四角往往含棋盘外框，
            #      比真实交叉点网格约大半格，不修则采样整体偏心）----
            h_img, w_img = img.shape[:2]
            if corner_size and (w_img, h_img) != tuple(corner_size):
                corners = scale_corners(corners, corner_size, (w_img, h_img))
                corner_size = (w_img, h_img)
            if tracker is None and not grid_refined and iterations >= grid_next_retry:
                rect, cell, margin, corners, fit = refine_rectify(img, corners)
                if fit is None:
                    # 关键：失败也要排重试。棋子太少（残局）时拟合不出网格，
                    # 但下一局棋子满了就能成功——只试一次会永久用偏掉的标定四角。
                    grid_next_retry = iterations + 15
                    print("警告：网格自动精修失败（棋子太少？），先按标定四角继续，30 秒后自动重试。")
                else:
                    grid_refined = True
                    print(f"网格已自动精修（修正四角: {[[round(x), round(y)] for x, y in corners]}）")
            else:
                rect, cell, margin = rectify(img, corners)

            # ---- 开局：识别 + 自动定向，建立可信盘面 ----
            if tracker is None:
                # 精修给出的四角仍可能差一两像素：换皮肤后实测「离线 90/90、
                # 运行时 87/90」，差的就是顶部一行三个子（将+双象）。建盘只有
                # 一次机会，所以在精修值附近枚举几组微调，取与开局最吻合的。
                best = None
                for dy0, dy1 in ((0, 0), (-2, 0), (2, 0), (0, -2), (0, 2),
                                 (-2, -2), (2, 2), (-1, 0), (1, 0)):
                    trial = [[x, y + (dy0 if i < 2 else dy1)] for i, (x, y) in enumerate(corners)]
                    r2, c2, m2 = rectify(img, trial)
                    s2, f2, mt = detect_orientation(
                        r2, templates, c2, m2, prefer_flip=cfg.get("flip_board")
                    )
                    if best is None or mt > best[2]:
                        best = (s2, f2, mt, trial, r2, c2, m2)
                std, flip, matches, corners, rect, cell, margin = best
                # 中局与开局的匹配数会随步数自然下降（每步差 2~4 格），
                # 只有低到「方向都判不准」的程度才值得告警。
                # 用「棋子清点」判识别是否可信：中局本来就离开局很远（实测真机
                # 中局仅 69/90 匹配却完全正确），拿开局匹配数当门槛只会误报。
                problems = board_problems(std)
                if problems:
                    print(f"警告：识别结果不像一副真棋（{'；'.join(problems)}），可跑 diagnose 看每格分数。")
                    guard.penalize(f"启动盘面可疑（{'；'.join(problems)}）")
                board = [row[:] for row in std]
                my_side = resolve_my_side(cfg, flip)
                # 启动时轮到谁：先看谁的头像计时器在跳（自动），判不出再兜底——
                # 开局必然红先；中途接管时默认「轮到我方」（玩家重启程序通常
                # 就是等着自己走）。配置里的 start_side 可强制覆盖。
                side0 = str(cfg.get("start_side") or "").strip().lower()
                if side0 not in ("w", "b"):
                    side0 = detect_first_mover(cfg, img, grab, my_side)
                if side0 not in ("w", "b"):
                    # 只有「与开局完全一致」才敢断定红方还没动过。差一格就说明
                    # 已经有子离位（红先走完一步就差 2 格），此时该我方走——
                    # 用 85 当门槛会把「红方刚走完」误判成开局，程序白白等对方。
                    side0 = "w" if matches >= 90 else my_side
                tracker = BoardTracker(board, flip, cell, margin, side_to_move=side0)
                tracker.update(rect)  # 建立基准帧
                summary["flip"] = flip
                summary["my_side"] = my_side
                summary["grid_refined"] = fit is not None
                bottom_side = "b" if flip else "w"
                where = "下方" if my_side == bottom_side else "上方"
                if matches >= 85:
                    print(f"初始局面已建立（识别匹配 {matches}/90，屏幕{'红方在上' if flip else '红方在下'}）：")
                else:
                    print(f"中局盘面已建立（与开局匹配 {matches}/90，屏幕{'红方在上' if flip else '红方在下'}）：")
                print_board(tracker.board)
                print(f"我方为{side_names(my_side)}方（截屏{where}）")
                print(f"[{_clock()}] 开始记录 —— 每步走子与建议都会带时间点")
                push(
                    status="waiting" if tracker.side_to_move != my_side else "ready",
                    message=(
                        f"等待{side_names(tracker.side_to_move)}方走子"
                        if tracker.side_to_move != my_side
                        else f"轮到我方（{side_names(my_side)}），正在算棋…"
                    ),
                    board=tracker.board, side=tracker.side_to_move,
                    my_side=my_side, flip=flip, move=None,
                )
                if tracker.side_to_move != my_side:
                    print(f"等待{side_names(tracker.side_to_move)}方（对方）走子…")
                sleep(interval)
                continue

            # ---- 网格精修重试 ----
            # 启动时若是残局，圆盘太少拟合不出网格，只能先用标定四角（可能偏掉）。
            # 这里低频重试：等棋子变多（开新局）拟合成功后，按新网格重建整盘。
            if not grid_refined and iterations >= grid_next_retry:
                _, _, _, used, fit = refine_rectify(img, corners)
                if fit is not None:
                    corners = used
                    grid_refined = True
                    print(f"网格精修补成功（修正四角: {[[round(x), round(y)] for x, y in corners]}），按新网格重建盘面。")
                    tracker = None
                    sleep(interval)
                    continue
                grid_next_retry = iterations + 15

            # ---- 追踪走子 ----
            event = tracker.update(rect)
            if debug:
                print(f"[帧 {iterations}] {event.kind} {event.detail if event.kind != 'moved' else event.moves}")

            if event.kind == "moved":
                for piece, src, dst in event.moves:
                    try:
                        chinese = move_to_chinese(event.board_before, src[0], src[1], dst[0], dst[1])
                    except Exception:  # noqa: BLE001 - 记谱失败不影响追踪
                        chinese = "(记谱失败)"
                    print(f"[{_clock()}] 检测到走子：{chinese}  {piece} {src} -> {dst}")
                    last_move_at = time.time()
                    summary["moves"].append((piece, src, dst, chinese))
                    push(last_move={"chinese": chinese, "piece": piece,
                                    "from": list(src), "to": list(dst)})
                turn = tracker.side_to_move
                print(f"轮到{side_names(turn)}方{'（我方）' if turn == my_side else '（对方）'}")
                push(
                    status="ready" if turn == my_side else "waiting",
                    message=(f"轮到我方（{side_names(my_side)}），正在算棋…"
                             if turn == my_side else f"等待{side_names(turn)}方走子"),
                    board=tracker.board, side=turn, move=None,
                )
                print_board(tracker.board)
                # 走子后的盘面必须仍然「像一副真棋」。理论上 moved 只是把子
                # 从 A 挪到 B，数量不会变；但实测走过一段后会出现「黑方 3 个
                # 车、红方 1 个车」这种自相矛盾的局面（某帧把红车读成黑车、
                # 或落点高亮让同一格被数两次），之后 FEN 非法、算棋被静默跳过，
                # 表现为「走了却不给方案」。所以每步都查一次，一旦不自洽就
                # 立刻按多帧投票重建，而不是等追踪彻底失联。
                problems = board_problems(tracker.board)
                if problems:
                    res = _resync_once(
                        tracker, rect, templates, cell, margin, summary,
                        reason=f"盘面不自洽（{'；'.join(problems)}）",
                        mguard=mguard,
                    )
                    # 重建成功说明盘面已经刷新干净，不必再冻结——冻结是给
                    # 「没修好」的情况留的，滥用只会让提示卡刷屏。
                    if res is None or res.kind not in ("resynced", "moved"):
                        guard.penalize(f"盘面不自洽（{'；'.join(problems)}）")

            elif event.kind == "need_resync":
                res = _resync_once(tracker, rect, templates, cell, margin, summary, mguard=mguard)
                if res is None or res.kind not in ("resynced", "moved"):
                    guard.penalize("追踪失联")

            elif event.kind == "settled" and debug:
                print(f"[帧 {iterations}] 纯噪声已吸收为新基准（候选 {event.detail.get('candidates')} 格）")

            # ---- 轮到我方且局面未算过：算棋 ----
            if tracker.side_to_move == my_side:
                fen = board_to_fen(tracker.board, tracker.side_to_move)
                if fen != last_fen:
                    if not fen_is_sane(fen):
                        last_fen = fen
                        guard.penalize("盘面读出来不合法")
                        print(f"FEN 异常，跳过算棋: {fen}")
                    elif not guard.ok:
                        # 这里**不能**更新 last_fen：冷却结束后还要按同一个局面重算，
                        # 提前吃掉 FEN 会让程序「永远」不再出声（实测用户以为卡死）。
                        _freeze_notice(guard.reason, img, out_path, notify=push)
                    else:
                        # ---- 出建议前独立复核 ----
                        # 追踪盘面会「静默漏子」：少一个车在规则上完全合法，
                        # board_problems 抓不到。真机第 5 局就是这样——重同步时
                        # 把左边那辆车读丢了，引擎拿着「单车」的假盘面算出 621 的
                        # 炮退一步，而真实局面双车下底已经是杀。所以每次要开口
                        # 之前，先自己重新认一遍整盘，对不上就闭嘴。
                        verify = recognize_board(rect, templates, cell, margin)[0]
                        if tracker.flip:
                            verify = flip_board(verify)
                        verify, _ = sanitize_positions(verify, strict=True)
                        diff = _board_diff(tracker.board, verify)
                        if diff:
                            for r, f, a, b in diff[:5]:
                                print(f"  复核差异 ({r},{f}): 追踪 {a!r} → 重识别 {b!r}")
                            jump = material_jump(tracker.board, verify)
                            if jump and is_new_game_restart(tracker.board, verify):
                                print("盘面从残局跳回开局规模（用户开了新局），按重识别结果重建。")
                                tracker.side_to_move = "w"   # 新局红先，别沿用上一局的走子方
                                mguard.clear()
                                jump = []
                            if jump and not mguard.confirm(jump):
                                # 一次少两个车这类跳变，先认定是识别糊了，守住原盘面
                                guard.penalize(f"子力跳变（{'；'.join(jump)}）")
                                _freeze_notice(
                                    f"重识别与追踪的子力不符（{'；'.join(jump)}）", img, out_path, notify=push
                                )
                                sleep(interval)
                                continue
                            if jump:
                                print(f"注意：子力跳变（{'；'.join(jump)}）反复出现，强制采纳重识别结果。")
                            mguard.clear()
                            tracker.set_board(verify, rect)   # 以重识别为准
                            guard.penalize(f"追踪盘面与重识别差 {len(diff)} 格")
                            _freeze_notice(f"追踪盘面与重识别差 {len(diff)} 格", img, out_path, notify=push)
                            sleep(interval)
                            continue
                        last_fen = fen
                        result = _engine_search(engine, fen, movetime_ms)
                        summary["searches"] += 1
                        move = result.get("bestmove") if result else None
                        reason = (
                            "引擎没有给出走法"
                            if not move or move in ("(none)", "0000")
                            else check_engine_move(tracker.board, tracker.side_to_move, move)
                        )
                        if reason:
                            print(f"这次不算走法：{reason}（{move}）")
                        else:
                            (fr, ff), (tr, tf) = parse_move(move)
                            try:
                                chinese = move_to_chinese(tracker.board, fr, ff, tr, tf)
                            except Exception:  # noqa: BLE001
                                chinese = "(记谱失败)"
                            spend = "" if last_move_at is None else f"  [走子后 {_elapsed(time.time() - last_move_at)}]"
                            print(f"[{_clock()}] 我方最佳走法: {chinese}  ({move})  评估: {_score_text(result.get('score'))}{spend}")
                            push(
                                status="ready", message=f"建议：{chinese}",
                                move={
                                    "from": [fr, ff], "to": [tr, tf], "uci": move,
                                    "chinese": chinese, "score": result.get("score"),
                                    "score_text": str(_score_text(result.get("score"))),
                                },
                                board=tracker.board, side=tracker.side_to_move,
                            )
                            # 箭头画在原始截图上，标准坐标需换算成屏幕坐标
                            sfr = tracker.to_screen(fr, ff)
                            str_ = tracker.to_screen(tr, tf)
                            annotated = draw_move(img, corners, (sfr, str_))
                            try:
                                cv2.imwrite(out_path, annotated)
                            except Exception as e:  # noqa: BLE001
                                print(f"写 {out_path} 失败: {e}")
                            if view is not None:
                                try:
                                    view.update(annotated)
                                except Exception:  # noqa: BLE001
                                    print("实时窗口更新失败，后续只写文件")
                                    view = None

            push()          # 每帧一次心跳，界面据此判断后端是否还在跑
            sleep(interval)
    except KeyboardInterrupt:
        summary["quit_reason"] = "keyboard_interrupt"
    finally:
        if tracker is not None:
            summary["board"] = tracker.board
            summary["side"] = tracker.side_to_move
        if view is not None:
            view.close()
        engine.quit()
        print(
            f"[{_clock()}] 监控结束，运行 {_elapsed(time.time() - _RUN_STARTED)}："
            f"识别走子 {len(summary['moves'])} 步，出建议 {summary['searches']} 次，"
            f"重同步 {summary['resyncs']} 次"
        )

    return summary


MATE_SCORE = 99000   # parse_score 把 mate N 折成 ±100000


class TrustGuard:
    """盘面可信度守门：识别刚出过问题时，暂停给出建议。

    实测教训（真机第 3 局）：识别一旦开始漂移（FEN 非法、追踪失联、多帧
    票数掉到 2），引擎就在「假棋盘」上算出看起来极美的着法——它曾建议红马
    跳开、把停在黑车直线上的红车让出来，结果白丢一车而崩盘。这类错误的
    建议比没有建议更坏，所以只要近期出过问题就冻结输出，等画面连续干净
    几帧再恢复。
    """

    def __init__(self, cooldown_frames=2):
        self.cooldown_frames = cooldown_frames
        self._left = 0
        self.reason = ""

    def penalize(self, reason):
        """记一次「这盘面可能不值得信」，重新开始冷却。"""
        self._left = self.cooldown_frames
        self.reason = reason

    def tick(self):
        """每帧调用一次，冷却到期自动恢复。"""
        if self._left > 0:
            self._left -= 1
            if self._left == 0:
                self.reason = ""

    @property
    def ok(self):
        return self._left == 0


def _score_text(score):
    """评估分的可读形式。

    直接把 100000 甩给用户，看不出是「我要绝杀对方」还是「我要被绝杀」，
    而这两个结论对下棋是完全相反的意思，所以 mate 分一律换成文字。
    """
    if score is None:
        return "?"
    if score >= MATE_SCORE:
        return "绝杀（对方无解）"
    if score <= -MATE_SCORE:
        return "被绝杀（我方无解）"
    return score


_RUN_STARTED = time.time()


def _clock():
    """墙钟时间 HH:MM:SS，用来给对局记时间点。"""
    return time.strftime("%H:%M:%S")


def _elapsed(seconds):
    """秒数 → '1m23s' 这种好读的形式。"""
    seconds = int(max(0, seconds))
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m{seconds % 60:02d}s"


class MaterialGuard:
    """子力突变的二次确认。

    单帧识别出「红方少两个车」这种跳变时先不采纳（一步最多吃一个子），
    但也不能死守：如果连着几次识别都稳定地给出同一个盘面，那多半是
    **基准本身早就错了**（比如程序中途接管时盘面就带错），这时必须校准，
    否则会永久卡在「拒绝采纳」上。所以连着 `strikes` 次一致就放行并告警。
    """

    def __init__(self, strikes=3):
        self.strikes = strikes
        self._count = 0

    def confirm(self, reasons):
        """记一次跳变，返回 True 表示「连着被拦太多次，应当强制采纳」。

        不比较 reasons 文本：识别每帧抖出来的差异文案可能都不一样
        （「车多出 2 个」→「车多出 2 个、马多出 1 个」），按文本比对会让
        计数不断归零、永远凑不满，于是永久卡在拒绝采纳上。只数次数。
        """
        self._count += 1
        return self._count >= self.strikes

    def clear(self):
        self._count = 0


def _freeze_notice(reason, img, out_path, notify=None):
    """「这帧不给建议」的统一出口：写一行原因，并把箭头图覆盖成作废样式。"""
    print(f"[{_clock()}] 暂不给出建议：{reason}，等画面稳定几帧再算。")
    if notify is not None:
        notify(status="frozen", message=f"暂不给出建议：{reason}", move=None)
    try:
        cv2.imwrite(out_path, draw_void(img))
    except Exception as e:  # noqa: BLE001
        print(f"写 {out_path} 失败: {e}")


def _resync_once(tracker, rect, templates, cell, margin, summary,
                 reason="追踪失联", mguard=None):
    """按多帧投票重新识别并重建盘面。

    两个入口共用：追踪彻底失联（need_resync），以及走子后盘面自相矛盾
    （黑方多出一个车之类）。后者不会让追踪报错——差分照样能凑出「一步」，
    只是拼出来的盘面根本不存在，最终 FEN 非法、算棋被静默跳过。

    重建结果还要过一道**子力校验**：一步最多吃一个子，若新盘面凭空多子
    或一次少两个以上，先拒绝采纳（很可能只是这一批帧糊了），连续几次都
    这么认才强制校准——那说明基准本身就错了，再守下去只会永久卡死。
    """
    board_before_resync = [row[:] for row in tracker.board]
    frames = tracker.recent_frames()[-3:] or [rect]
    recognized, votes = majority_board(frames, templates, cell, margin)
    recognized = flip_board(recognized) if tracker.flip else recognized
    jump = material_jump(tracker.board, recognized)
    new_game = False
    if jump and is_new_game_restart(tracker.board, recognized):
        new_game = True
        print(f"{reason}：盘面从残局跳回开局规模（用户开了新局），按新对局重建。")
        # 关键：新局红先。上一局结束时 side_to_move 停在某方，不重置的话
        # 程序会一直「等对方走」，盘面明明是对的却不出建议（真机实测）。
        tracker.side_to_move = "w"
        if mguard is not None:
            mguard.clear()
        jump = []
    if jump and mguard is not None and not mguard.confirm(jump):
        summary["resyncs"] += 1
        print(f"{reason}，但重建出的盘面子力跳变（{'；'.join(jump)}），拒绝采纳，保留原盘面。")
        tracker.set_board(tracker.board, rect)   # 只刷新基准帧，别让同一批糊帧反复触发
        return None
    if jump:
        print(f"注意：子力跳变（{'；'.join(jump)}）反复出现，强制校准盘面，请核对屏幕。")
        if mguard is not None:
            mguard.clear()
    min_votes = min(min(v) for v in votes)
    res = tracker.resync(recognized, rect)
    summary["resyncs"] += 1
    if res.kind == "resynced":
        print(f"{reason}，已按识别结果重建盘面（多帧最低票数 {min_votes}）：")
        print_board(tracker.board)
        # 走子没被解析成「一步」时也会走到这里（比如落点抖动、双步同帧）。
        # 盘面已经更新了，但回合还停在原方，会一直「等对方走」而不给建议，
        # 所以有实际变化就切到对方——宁可多给一条建议，也别彻底哑掉。
        if not new_game and _board_diff(board_before_resync, tracker.board):
            tracker.side_to_move = "b" if tracker.side_to_move == "w" else "w"
            print(f"（该步未能解析为走子，已按走子处理，轮到{side_names(tracker.side_to_move)}方）")
    elif res.kind == "moved":
        piece, src, dst = res.moves[0]
        print(f"{reason}，识别补救走子：{piece} {src} -> {dst}")
        summary["moves"].append((piece, src, dst, "(识别补救)"))
        print_board(tracker.board)
    elif res.kind == "settled":
        print(f"{reason}，重新识别后无差异，已刷新基准帧。")
    else:
        print(f"重新识别未通过 sanity 检查（{res.detail.get('resync')}），保留原盘面继续观察。")
    return res


def _engine_search(engine, fen, movetime_ms):
    """算棋，引擎崩溃时重启重试一次。"""
    try:
        engine.set_position(fen)
        return engine.search(movetime_ms)
    except Exception as e:  # noqa: BLE001 - 引擎任何异常都尝试重启一次
        print(f"引擎通信失败（{e}），尝试重启引擎…")
        try:
            engine.restart()
            engine.set_position(fen)
            return engine.search(movetime_ms)
        except Exception as e2:  # noqa: BLE001
            print(f"引擎重启后仍失败: {e2}")
            return None


def _cli(argv=None):
    p = argparse.ArgumentParser(description="鸿蒙手机象棋实时识别 + 最佳走法")
    p.add_argument("--config", default="config.json", help="配置文件路径（默认 config.json）")
    p.add_argument("--interval", type=float, default=None, help="轮询间隔秒数（覆盖 config 的 poll_interval）")
    p.add_argument("--movetime", type=int, default=None, help="引擎思考毫秒数（覆盖 config 的 movetime_ms）")
    p.add_argument("--show", action="store_true", help="打开实时窗口（无显示环境时自动退化为只写文件）")
    p.add_argument("--debug", action="store_true", help="打印每帧追踪细节")
    args = p.parse_args(argv)

    summary = main(
        args.config,
        show=args.show,
        debug=args.debug,
        interval=args.interval,
        movetime_ms=args.movetime,
    )
    mine = summary.get("my_side")
    print(
        f"\n运行结束（{summary['quit_reason']}）：{summary['frames']} 帧，"
        f"{len(summary['moves'])} 步走子，{summary['searches']} 次算棋，"
        f"{summary['resyncs']} 次重同步。"
    )
    if mine:
        bottom_side = "b" if summary.get("flip") else "w"
        where = "下方" if mine == bottom_side else "上方"
        print(f"我方为{side_names(mine)}方（截屏{where}）")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
