import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# PRESETS  (anode chemistry parameter sets)
# ============================================================
#
# "si_v4"  : literature-supported core model
#            params_Si_prelithiation.m v4
#            Anchor  [A] Zhao 2025, JES 172 080512
#            Rate    [B] Li 2023, J Energy Storage 64 107149
#
# "sic_v3" : SUPERSEDED DRAFT, not a different validated chemistry.
#            params_SiC_prelithiation.m v3 originally targeted a
#            Si/C composite anode, but no literature satisfying the
#            model's own criteria was found for that chemistry, so
#            scope was narrowed to pure Si (-> si_v4). This preset
#            is kept only as a low-confidence reference: it uses
#            the same [A]/[B] sources but older, partially
#            "TODO"-flagged digitizations, an extrapolated R_SEI at
#            Q_PL = 0, and no on/off switches (rate + duty corrections
#            were always-on in v3). Do not present this as a second
#            validated material system.
#
# ============================================================

PRESETS = {
    "si_v4": {
        "label": "Si (순수 실리콘) · v4 — 문헌 지지 core 모델",
        "confidence_badge": "High / Medium",
        "note": (
            "Zhao 2025 [A] 앵커(같은 셀, Q_PL 6수준) + Li 2023 [B] rate 보정. "
            "현재 앱의 기본 모델."
        ),
        "Q_ref": 3579.0,
        "Q_target_default": 3000.0,
        "Q_target_min": 500.0,
        "R0_fixed": 3.5,
        "Q_PL_bp": np.array([0, 500, 1050, 1500, 2000, 2500, 3000], dtype=float),
        "ICE_base": np.array([51.0, 69.2, 69.2, 84.6, 83.0, 89.0, 88.2]),
        "Rct_base": np.array([35.5, 24.6, 18.5, 13.2, 8.4, 5.9, 6.9]),
        "Q_PL_RSEI_bp": np.array([500, 1050, 1500, 2000, 2500, 3000], dtype=float),
        "RSEI_base": np.array([80.1, 75.9, 68.5, 59.1, 50.9, 54.3]),
        "RSEI_gate_min": 500.0,
        "RSEI_extrapolated_at_zero": False,
        "C_rate_bp": np.array([0.02, 0.05, 0.08, 0.10]),
        "f_rate_RSEI": np.array([94.0, 86.6, 93.6, 112.4]) / 86.6,
        "f_rate_ICE": np.array([99.8, 99.5, 98.8, 99.2]) / 99.5,
    },
    "sic_v3": {
        "label": "SiC (Si/C 초안) · v3 — Low confidence, 참고용",
        "confidence_badge": "Low (미완성 초안 · 폐기된 버전)",
        "note": (
            "원래 Si/C 복합체를 다루려 했으나 조건에 맞는 문헌을 찾지 못해 "
            "si_v4에서 순수 Si로 범위가 좁혀졌습니다. 이 프리셋은 그 이전 "
            "초안으로, 일부 값이 논문 그림에서 대략 읽은 값(TODO 표기 있음)이고 "
            "R_SEI(Q_PL=0)은 외삽값입니다. 화학종 비교용이 아니라 이력 참고용입니다."
        ),
        "Q_ref": 3579.0,
        "Q_target_default": 3000.0,
        "Q_target_min": 0.0,
        "R0_fixed": 3.5,
        "Q_PL_bp": np.array([0, 500, 1050, 1500, 2000, 2500, 3000], dtype=float),
        "ICE_base": np.array([51.0, 62.0, 71.0, 85.0, 87.0, 89.0, 89.0]),
        "Rct_base": np.array([35.5, 28.0, 22.0, 16.0, 11.0, 7.0, 6.0]),
        "Q_PL_RSEI_bp": np.array([0, 500, 1050, 1500, 2000, 2500, 3000], dtype=float),
        "RSEI_base": np.array([95.0, 80.1, 72.0, 63.0, 56.0, 51.0, 50.0]),
        "RSEI_gate_min": None,
        "RSEI_extrapolated_at_zero": True,
        "C_rate_bp": np.array([0.02, 0.05, 0.08, 0.10]),
        # already normalized to 0.05C in the source file
        "f_rate_RSEI": np.array([1.69, 1.00, 1.10, 1.15]),
        "f_rate_ICE": np.array([1.003, 1.000, 0.993, 0.997]),
    },
}

w_ICE = 0.5
w_RSEI = 0.5
ICE_overPL_threshold = 100.0

# ------------------------------------------------------------
# DUTY-CYCLE FACTORS — EXPLORATORY, OFF BY DEFAULT
#
# params_prelithiation.m 4절과 같은 배열이다. 논문 그림에서 digitize한
# 값이 아니라, 일반 펄스 충전 문헌 [C]의 정성적 경향(휴지 구간이 있으면
# 표면 Li 농도구배가 완화되어 국부 과전압이 낮아지고, 그 결과 전해액
# 환원이 덜 일어나 SEI가 더 얇고 균일해진다)을 숫자로 옮겨 적은
# 플레이스홀더다. Si 프리리튬화의 pulse-vs-continuous 실측 데이터는
# 아직 없으므로 기본 가정 강도는 0 (= 보정 없음)이다.
#
# 실측이 생기면 이 네 줄만 교체하고 기본 가정 강도를 1.0으로 올리면 된다.
# ------------------------------------------------------------
DUTY_BP     = np.array([0.10, 0.25, 0.50, 0.75, 1.00])
F_DUTY_RSEI = np.array([0.90, 0.83, 0.86, 0.93, 1.00])
F_DUTY_RCT  = np.array([0.95, 0.91, 0.93, 0.97, 1.00])
F_DUTY_ICE  = np.array([1.014, 1.020, 1.017, 1.009, 1.000])
DUTY_CONF   = "Low"

# ------------------------------------------------------------
# Nyquist / Randles circuit — EXPLORATORY, NOT literature-digitized
#
# Neither params_Si_prelithiation.m nor the original Simulink build
# script (build_Si_Prelithiation_V3.m) ever simulated an actual R-C
# circuit: R_SEI / R_ct are scalar EIS-fit outputs taken directly
# from the source papers, not values produced by a simulated
# circuit. No C_SEI / C_dl value exists anywhere in the literature
# ledger for this project.
#
# To draw a Nyquist curve at all, characteristic frequencies
# (representative of where an SEI arc vs. a charge-transfer arc
# typically appears in Si-anode EIS) are assumed, and capacitance is
# backed out from the literature-supported R value:
#
#       C = 1 / (2*pi*f_char*R)
#
# This keeps the semicircle SIZE (R-axis) literature-anchored while
# clearly flagging the semicircle WIDTH (frequency axis) as a
# representative assumption, not a digitized fit.
# ------------------------------------------------------------
F_SEI_CHAR_HZ = 2000.0   # representative SEI-arc characteristic frequency (thin film -> small C -> high f)
F_CT_CHAR_HZ  = 100.0    # representative charge-transfer-arc characteristic frequency
                         # ([A]'s electrode is mesoporous, BET ~230 m^2/g, so a larger
                         #  double-layer capacitance than a dense film is physically expected)


# ============================================================
# SIMULATION CORE
# ============================================================

def interp_clip(x, xp, fp):
    """Linear interpolation with clipping at boundaries."""
    return float(np.interp(x, xp, fp, left=fp[0], right=fp[-1]))


def duty_factor(duty, arr, belief):
    """
    펄스 이득 가정을 belief(0~1)만큼만 인정한 duty 보정계수.

    belief = 0  -> 1.0        (보정 없음. 지금까지의 기본 동작)
    belief = 1  -> arr 값 그대로 (params_prelithiation.m 4절 플레이스홀더)

    실측 데이터가 아니므로 belief > 0 인 결과는 화면에서 '가정 포함'으로
    표시된다.
    """
    if belief <= 0.0:
        return 1.0
    return 1.0 + belief * (interp_clip(duty, DUTY_BP, arr) - 1.0)


def simulate(preset, C_rate, pl_mode, t_on, t_off, Q_tgt, duty_belief=0.0):
    """
    Returns dict with ICE, RSEI, Rct, R0, time_h, Q_PL,
    RSEI_valid, overPL at the moment Q_PL reaches Q_tgt.
    """
    Q_ref = preset["Q_ref"]

    duty = 1.0 if pl_mode == 0 else t_on / (t_on + t_off)

    I_peak = C_rate * Q_ref
    I_avg = I_peak * duty

    t_needed = Q_tgt / I_avg * 3600.0

    f_ICE = interp_clip(C_rate, preset["C_rate_bp"], preset["f_rate_ICE"])
    f_RSEI = interp_clip(C_rate, preset["C_rate_bp"], preset["f_rate_RSEI"])
    f_Rct = 1.0  # rate -> Rct disabled in both presets

    # duty 보정은 펄스 조건에만, 그리고 가정 강도만큼만 적용된다
    if pl_mode == 1 and duty_belief > 0.0:
        d_ICE = duty_factor(duty, F_DUTY_ICE, duty_belief)
        d_RSEI = duty_factor(duty, F_DUTY_RSEI, duty_belief)
        d_Rct = duty_factor(duty, F_DUTY_RCT, duty_belief)
    else:
        d_ICE = d_RSEI = d_Rct = 1.0

    f_ICE *= d_ICE
    f_RSEI *= d_RSEI
    f_Rct *= d_Rct

    ICE_val = interp_clip(Q_tgt, preset["Q_PL_bp"], preset["ICE_base"]) * f_ICE
    Rct_val = interp_clip(Q_tgt, preset["Q_PL_bp"], preset["Rct_base"]) * f_Rct

    gate = preset["RSEI_gate_min"]
    RSEI_valid = True if gate is None else (Q_tgt >= gate)

    if RSEI_valid:
        RSEI_val = interp_clip(Q_tgt, preset["Q_PL_RSEI_bp"], preset["RSEI_base"]) * f_RSEI
    else:
        RSEI_val = float("nan")

    RSEI_is_extrapolated = bool(preset.get("RSEI_extrapolated_at_zero")) and (Q_tgt <= preset["Q_PL_RSEI_bp"][0] + 1e-9)

    time_h = t_needed / 3600.0

    return {
        "ICE": ICE_val,
        "RSEI": RSEI_val,
        "Rct": Rct_val,
        "R0": preset["R0_fixed"],
        "time_h": time_h,
        "Q_PL": Q_tgt,
        "RSEI_valid": RSEI_valid,
        "RSEI_is_extrapolated": RSEI_is_extrapolated,
        "overPL": ICE_val >= ICE_overPL_threshold,
        "duty": duty,
        "duty_belief": duty_belief if pl_mode == 1 else 0.0,
        "duty_applied": (pl_mode == 1 and duty_belief > 0.0),
    }


def get_curves(preset, C_rate, Q_target):
    """Return ICE, RSEI, Rct vs Q_PL curves for plotting."""
    Q_arr = np.linspace(0, Q_target, 200)

    f_ICE = interp_clip(C_rate, preset["C_rate_bp"], preset["f_rate_ICE"])
    f_RSEI = interp_clip(C_rate, preset["C_rate_bp"], preset["f_rate_RSEI"])

    ICE_arr = np.array([interp_clip(q, preset["Q_PL_bp"], preset["ICE_base"]) * f_ICE for q in Q_arr])
    Rct_arr = np.array([interp_clip(q, preset["Q_PL_bp"], preset["Rct_base"]) for q in Q_arr])

    gate = preset["RSEI_gate_min"]
    q0 = preset["Q_PL_RSEI_bp"][0] if gate is None else gate
    Q_rsei = Q_arr[Q_arr >= q0]
    RSEI_arr = np.array([interp_clip(q, preset["Q_PL_RSEI_bp"], preset["RSEI_base"]) * f_RSEI for q in Q_rsei])

    return Q_arr, ICE_arr, Rct_arr, Q_rsei, RSEI_arr


def sweep(preset, Q_target, duty_belief=0.0, w_time=0.0):
    """
    20조건 스윕.

    duty_belief : 펄스 이득 가정 강도 0~1. 0이면 펄스의 ICE/R_SEI/Rct는
                  같은 C-rate의 연속 조건과 완전히 같은 값이 된다.
    w_time      : 공정시간 가중치 0~1. 0이면 점수식이 기존과 동일하다.
                  0보다 크면 품질 가중치가 (1-w_time)/2씩으로 재배분된다.

    펄스 조건도 이제 점수를 받는다. duty_belief = 0 일 때 펄스의 점수는
    같은 C-rate 연속 조건과 동점이 되는데, 이는 '펄스가 열등하다'가 아니라
    '구분할 데이터가 아직 없다'는 뜻을 그대로 보여주는 것이다.
    """
    C_list = [0.02, 0.05, 0.08, 0.10]
    t_on_s = [10 * 60, 10 * 60, 20 * 60, 30 * 60]
    t_off_s = [60 * 60, 30 * 60, 20 * 60, 10 * 60]

    rows = []

    for cr in C_list:
        r = simulate(preset, cr, 0, 0, 0, Q_target)
        r["label"] = f"{cr:.2f}C continuous"
        r["C_rate"] = cr
        r["mode"] = "Continuous"
        r["t_on"] = 0
        r["t_off"] = 0
        r["support"] = "Core"
        rows.append(r)

    for cr in C_list:
        for ton, toff in zip(t_on_s, t_off_s):
            r = simulate(preset, cr, 1, ton, toff, Q_target, duty_belief=duty_belief)
            r["label"] = f"{cr:.2f}C pulse {ton // 60}/{toff // 60}min"
            r["C_rate"] = cr
            r["mode"] = "Pulse"
            r["t_on"] = ton // 60
            r["t_off"] = toff // 60
            r["support"] = "Exploratory"
            rows.append(r)

    # ── 정규화 기준 ────────────────────────────────────────────
    # ICE / R_SEI 는 지금까지와 똑같이 '연속 조건 4개'의 범위로만 정규화한다.
    # 그래야 기존에 발표한 연속 조건 점수(0.8566 / 0.8500 / 0.3643 / 0.2000)가
    # 그대로 재현되고, 펄스는 그 자와 비교되는 형태가 된다.
    core = [r for r in rows if r["mode"] == "Continuous" and r["RSEI_valid"]]
    ICE_min, ICE_max = min(c["ICE"] for c in core), max(c["ICE"] for c in core)
    RSEI_min, RSEI_max = min(c["RSEI"] for c in core), max(c["RSEI"] for c in core)

    # 공정시간은 가정이 아니라 정의(t = Q / (duty·C·Q_ref))이므로
    # 순위에 들어가는 모든 조건의 실제 범위로 정규화한다. 폭이 6.99h ~ 293h로
    # 매우 넓어 로그 스케일을 쓴다.
    ranked = [r for r in rows if r["RSEI_valid"]]
    lt = [np.log(r["time_h"]) for r in ranked]
    lt_min, lt_max = min(lt), max(lt)

    wT = float(np.clip(w_time, 0.0, 1.0))
    wI = wR = (1.0 - wT) / 2.0

    for r in rows:
        if not r["RSEI_valid"]:
            r["score"] = float("nan")
            r["s_ice"] = r["s_rsei"] = r["s_time"] = float("nan")
            continue
        r["s_ice"] = (r["ICE"] - ICE_min) / (ICE_max - ICE_min) if ICE_max > ICE_min else 1.0
        r["s_rsei"] = (RSEI_max - r["RSEI"]) / (RSEI_max - RSEI_min) if RSEI_max > RSEI_min else 1.0
        r["s_time"] = ((lt_max - np.log(r["time_h"])) / (lt_max - lt_min)) if lt_max > lt_min else 1.0
        r["score"] = wI * r["s_ice"] + wR * r["s_rsei"] + wT * r["s_time"]

    # ── 손익분기: 가정 없이도 펄스 행을 읽을 수 있게 만드는 값 ──────
    # "같은 공정시간 안에 돌릴 수 있는 최선의 연속 조건"을 이기려면
    # ICE를 몇 %p 올리거나 R_SEI를 몇 % 낮춰야 하는가. 순수한 역산이라
    # 어떤 가정도 들어가지 않는다.
    conts = [r for r in rows if r["mode"] == "Continuous" and r["RSEI_valid"]]
    for r in rows:
        r["be"] = None
        if r["mode"] != "Pulse" or not r["RSEI_valid"]:
            continue
        elig = [c for c in conts if c["time_h"] <= r["time_h"] + 1e-9]
        if not elig:
            continue
        rival = max(elig, key=lambda c: c["score"])
        be = {"rival": rival["label"],
              "rival_time_h": rival["time_h"],
              "extra_h": r["time_h"] - rival["time_h"]}
        gap = rival["score"] - r["score"]
        be["gap"] = gap
        if wI > 0 and ICE_max > ICE_min:
            s_need = (rival["score"] - wR * r["s_rsei"] - wT * r["s_time"]) / wI
            be["d_ice_pp"] = max(0.0, ICE_min + s_need * (ICE_max - ICE_min) - r["ICE"])
        if wR > 0 and RSEI_max > RSEI_min and r["RSEI"] > 0:
            s_need = (rival["score"] - wI * r["s_ice"] - wT * r["s_time"]) / wR
            need = RSEI_max - s_need * (RSEI_max - RSEI_min)
            be["d_rsei_pct"] = max(0.0, (1.0 - need / r["RSEI"]) * 100.0)
        r["be"] = be

    return rows


def randles_impedance(R0, R_SEI, C_SEI, R_ct, C_dl, freqs_hz):
    """Complex impedance of R0 + (R_SEI || C_SEI) + (R_ct || C_dl)."""
    w = 2.0 * np.pi * np.asarray(freqs_hz)
    Z_sei = (R_SEI / (1.0 + 1j * w * R_SEI * C_SEI)) if R_SEI > 0 else 0.0
    Z_ct = (R_ct / (1.0 + 1j * w * R_ct * C_dl)) if R_ct > 0 else 0.0
    return R0 + Z_sei + Z_ct


def derive_C(R, f_char_hz):
    return 1.0 / (2.0 * np.pi * f_char_hz * R)


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(page_title="Si Pre-lithiation Optimizer", layout="wide")

st.title("Si Anode Pre-lithiation Optimizer")
st.caption("Based on Zhao 2025 [A] (anchor) · Li 2023 [B] (rate correction)")

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.header("음극 프리셋")

    preset_key = st.selectbox(
        "화학종 / 데이터셋",
        options=list(PRESETS.keys()),
        format_func=lambda k: PRESETS[k]["label"],
        index=0,
    )
    preset = PRESETS[preset_key]

    if preset_key == "sic_v3":
        st.warning(f"**{preset['confidence_badge']}**\n\n{preset['note']}")
    else:
        st.caption(f"신뢰도: {preset['confidence_badge']} · {preset['note']}")

    st.divider()
    st.header("Conditions")

    Q_tgt_input = st.number_input(
        "Pre-lithiation target [mAh/g]",
        min_value=preset["Q_target_min"], max_value=3000.0,
        value=preset["Q_target_default"], step=100.0,
        key=f"qtgt_{preset_key}",
    )

    C_rate_input = st.number_input(
        "C-rate  (0.02 – 0.10)",
        min_value=0.019, max_value=0.101,
        value=0.05, step=0.01, format="%.3f")
    # 0.03 - 0.01 등은 이진 부동소수점으로 정확히 0.02가 아니라
    # 0.019999999999999997이 되어, min_value=0.02와 같으면 그 값이
    # "범위 밖"으로 판정되어 −버튼이 0.02에 도달하기 직전에 멈춰버림.
    # min/max를 한 스텝 폭 안에서 살짝 넓혀 그 경계 판정을 피하고,
    # 계산에 쓰는 값은 다시 소수 3자리로 반올림해 깨끗하게 만든다.
    C_rate_input = round(C_rate_input, 3)

    mode_input = st.radio("Mode", ["Continuous", "Pulse"])

    if mode_input == "Pulse":
        t_on_input = st.number_input("t_on  [min]", min_value=1, max_value=120, value=10)
        t_off_input = st.number_input("t_off [min]", min_value=1, max_value=120, value=30)
    else:
        t_on_input = 0
        t_off_input = 0

    show_nyquist = st.checkbox("Nyquist(EIS) 곡선 보기 — 탐색적", value=False)

    st.divider()
    with st.expander("순위 기준 (Ranking)", expanded=False):
        w_time_pct = st.slider(
            "공정시간 가중치", 0, 60, 0, step=5, format="%d %%",
            help="0 %면 지금까지와 같은 점수식(ICE 50 : R_SEI 50)입니다. "
                 "올리면 그만큼 품질 가중치가 줄고 공정시간이 순위에 반영됩니다. "
                 "시간은 t = Q /(duty·C·Q_ref)로 정확히 계산되는 값이라 "
                 "여기에는 아무 가정도 들어가지 않습니다.",
        )
        duty_belief_pct = st.slider(
            "펄스 이득 가정", 0, 100, 0, step=5, format="%d %%",
            help="펄스의 휴지 구간이 SEI를 더 얇고 균일하게 만든다는 효과를 "
                 "어느 정도로 인정할지 정합니다. 0 %면 보정 없음(현재 기본값), "
                 "100 %면 params_prelithiation.m 4절의 duty 플레이스홀더를 "
                 "그대로 적용합니다. Si 프리리튬화의 실측 pulse-vs-continuous "
                 "데이터는 아직 없으므로, 0 %보다 크게 두면 결과는 '가정 포함'입니다.",
        )
    w_time_input = w_time_pct / 100.0
    duty_belief_input = duty_belief_pct / 100.0

    run_btn = st.button("▶  Run", use_container_width=True)
    sweep_btn = st.button("⟳  Sweep (20 conditions)", use_container_width=True)

# ── Run single condition ──────────────────────────────────────
if run_btn:
    pl_mode = 1 if mode_input == "Pulse" else 0
    res = simulate(preset, C_rate_input, pl_mode,
                   t_on_input * 60, t_off_input * 60,
                   Q_tgt=Q_tgt_input, duty_belief=duty_belief_input)

    if res["RSEI_is_extrapolated"]:
        st.info("R_SEI 값은 Q_PL = 0에서 외삽된 값입니다 (실측 아님).")

    if res["duty_applied"]:
        st.warning(
            f"**가정 포함 결과** — 펄스 이득 가정 {duty_belief_pct} % 적용 "
            f"(duty {res['duty']:.3f}). 이 보정은 Si 프리리튬화 실측이 아니라 "
            f"일반 펄스 충전 문헌의 정성적 경향을 옮겨 적은 플레이스홀더입니다 "
            f"(신뢰도 {DUTY_CONF}). 문헌 지지 값을 보려면 가정을 0 %로 두세요."
        )
    elif pl_mode == 1:
        st.info(
            "펄스 보정이 꺼져 있어(가정 0 %) ICE·R_SEI·Rct는 같은 C-rate의 "
            "연속 조건과 같은 값입니다. 달라지는 것은 공정시간뿐입니다 — "
            "펄스가 열등하다는 뜻이 아니라, 아직 구분할 실측 데이터가 없다는 뜻입니다."
        )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ICE", f"{res['ICE']:.2f} %",
                delta="⚠ over-prelithiation" if res["overPL"] else None)
    col2.metric("R_SEI", f"{res['RSEI']:.2f} Ω" if res["RSEI_valid"] else "N/A (Q_PL too low)")
    col3.metric("Rct", f"{res['Rct']:.2f} Ω")
    col4.metric("Time", f"{res['time_h']:.2f} h")

    Q_arr, ICE_arr, Rct_arr, Q_rsei, RSEI_arr = get_curves(preset, C_rate_input, Q_tgt_input)

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["ICE [%]", "R_SEI [Ω]", "Rct [Ω]"])

    fig.add_trace(go.Scatter(x=Q_arr, y=ICE_arr, name="ICE",
                             line=dict(color="#2563EB", width=2)), row=1, col=1)
    fig.add_vline(x=Q_tgt_input, line_dash="dash", line_color="gray", row=1, col=1)

    fig.add_trace(go.Scatter(x=Q_rsei, y=RSEI_arr, name="R_SEI",
                             line=dict(color="#DC2626", width=2)), row=1, col=2)
    fig.add_vline(x=Q_tgt_input, line_dash="dash", line_color="gray", row=1, col=2)

    fig.add_trace(go.Scatter(x=Q_arr, y=Rct_arr, name="Rct",
                             line=dict(color="#16A34A", width=2)), row=1, col=3)
    fig.add_vline(x=Q_tgt_input, line_dash="dash", line_color="gray", row=1, col=3)

    fig.update_xaxes(title_text="Q_PL [mAh/g]")
    fig.update_layout(height=340, showlegend=False, margin=dict(t=40, b=30))
    st.plotly_chart(fig, use_container_width=True)

    # ── Nyquist (Randles circuit) — exploratory ──────────────
    if show_nyquist:
        st.subheader("Nyquist 곡선 (Randles 등가회로) — 탐색적")
        st.caption(
            "반원 크기(R축)는 문헌 지지 R_SEI/R_ct 값을 그대로 씁니다. "
            f"반원 폭(주파수축)은 대표 특성주파수(SEI {F_SEI_CHAR_HZ:.0f} Hz, "
            f"전하이동 {F_CT_CHAR_HZ:.0f} Hz)를 가정해 C = 1/(2πf·R)로 역산한 것으로, "
            "[A]/[B]에서 직접 디지타이징한 값이 아닙니다."
        )

        R0 = res["R0"]
        R_SEI_for_c = res["RSEI"] if res["RSEI_valid"] else 0.0
        Rct = res["Rct"]
        C_SEI = derive_C(R_SEI_for_c, F_SEI_CHAR_HZ) if R_SEI_for_c > 0 else 0.0
        C_dl = derive_C(Rct, F_CT_CHAR_HZ)

        freqs = np.logspace(5, -2, 400)  # 100 kHz -> 10 mHz, finer for a clean color split
        Z = randles_impedance(R0, R_SEI_for_c, C_SEI, Rct, C_dl, freqs)

        # The curve is really TWO semicircles in series (SEI arc, then charge-transfer
        # arc) but when one resistance is much bigger than the other they visually
        # blend into what looks like a single rounded arc. Color-split the SAME curve
        # at the geometric-mean frequency between the two characteristic frequencies so
        # a viewer can see which half of the curve each resistor is responsible for,
        # and label each segment's real-axis width with its Ω value directly.
        x0, x1, x2 = R0, R0 + R_SEI_for_c, R0 + R_SEI_for_c + Rct
        apex = float(np.max(-Z.imag)) if len(Z) else 1.0

        fig_nyq = go.Figure()

        if R_SEI_for_c > 0:
            f_split = np.sqrt(F_SEI_CHAR_HZ * F_CT_CHAR_HZ)
            idx_split = int(np.clip(np.searchsorted(-freqs, -f_split), 1, len(freqs) - 1))
            sei_sl, ct_sl = slice(0, idx_split + 1), slice(idx_split, None)  # share boundary point, no gap

            # 위 ICE/R_SEI/Rct 3분할 그래프에서 R_SEI는 빨강(#DC2626), Rct는 초록(#16A34A)을
            # 이미 쓰고 있음 — 새 색을 끌어오는 대신 앱 안에서 이미 통용되는 그 색을 그대로 재사용.
            fig_nyq.add_trace(go.Scatter(
                x=Z.real[sei_sl], y=-Z.imag[sei_sl], mode="lines+markers",
                marker=dict(size=4), line=dict(color="#DC2626", width=3),
                name=f"SEI 저항 성분 (R_SEI ≈ {R_SEI_for_c:.1f} Ω)",
                customdata=freqs[sei_sl],
                hovertemplate="f ≈ %{customdata:.3g} Hz<br>Z' = %{x:.2f} Ω<br>-Z'' = %{y:.2f} Ω<extra></extra>",
            ))
            fig_nyq.add_trace(go.Scatter(
                x=Z.real[ct_sl], y=-Z.imag[ct_sl], mode="lines+markers",
                marker=dict(size=4), line=dict(color="#16A34A", width=3),
                name=f"전하이동 저항 성분 (R_ct ≈ {Rct:.1f} Ω)",
                customdata=freqs[ct_sl],
                hovertemplate="f ≈ %{customdata:.3g} Hz<br>Z' = %{x:.2f} Ω<br>-Z'' = %{y:.2f} Ω<extra></extra>",
            ))
        else:
            fig_nyq.add_trace(go.Scatter(
                x=Z.real, y=-Z.imag, mode="lines+markers",
                marker=dict(size=4), line=dict(color="#16A34A", width=3),
                name=f"전하이동 저항 성분 (R_ct ≈ {Rct:.1f} Ω)",
                customdata=freqs,
                hovertemplate="f ≈ %{customdata:.3g} Hz<br>Z' = %{x:.2f} Ω<br>-Z'' = %{y:.2f} Ω<extra></extra>",
            ))

        # dotted guide lines + inline Ω labels at each segment's real-axis boundary
        for xv in ([x0, x1, x2] if R_SEI_for_c > 0 else [x0, x2]):
            fig_nyq.add_vline(x=xv, line_dash="dot", line_color="#9CA3AF", line_width=1)

        label_y = apex * 0.06
        if R_SEI_for_c > 0:
            fig_nyq.add_annotation(x=(x0 + x1) / 2, y=label_y, text=f"R_SEI ≈ {R_SEI_for_c:.1f} Ω",
                                    showarrow=False, bgcolor="rgba(255,255,255,0.85)",
                                    font=dict(size=12, color="#DC2626"))
            fig_nyq.add_annotation(x=(x1 + x2) / 2, y=label_y, text=f"R_ct ≈ {Rct:.1f} Ω",
                                    showarrow=False, bgcolor="rgba(255,255,255,0.85)",
                                    font=dict(size=12, color="#16A34A"))
        else:
            fig_nyq.add_annotation(x=(x0 + x2) / 2, y=label_y, text=f"R_ct ≈ {Rct:.1f} Ω",
                                    showarrow=False, bgcolor="rgba(255,255,255,0.85)",
                                    font=dict(size=12, color="#16A34A"))
        fig_nyq.add_annotation(x=x0, y=apex * 0.22, text=f"R0 ≈ {R0:.1f} Ω",
                                showarrow=True, arrowhead=2, ax=-40, ay=-30,
                                bgcolor="rgba(255,255,255,0.85)", font=dict(size=11, color="#374151"))

        fig_nyq.update_xaxes(title_text="Z' [Ω]")
        fig_nyq.update_yaxes(title_text="-Z'' [Ω]", scaleanchor="x", scaleratio=1,
                              range=[-apex * 0.1, apex * 1.15])
        fig_nyq.update_layout(
            height=460,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(t=20, b=30),
        )
        st.plotly_chart(fig_nyq, use_container_width=True)

        st.caption(
            "**읽는 법** — 반원의 가로 폭이 곧 그 저항 성분의 크기입니다. "
            f"원점 → 첫 지점({R0:.1f} Ω)은 배선·접촉 저항 R0, "
            + (f"빨간색 구간의 폭({R_SEI_for_c:.1f} Ω)은 SEI 저항, " if R_SEI_for_c > 0 else "")
            + f"초록색 구간의 폭({Rct:.1f} Ω)은 전하이동 저항입니다 (위 R_SEI·Rct 그래프와 같은 색). "
            + ("두 반원이 이어 붙어 하나처럼 보이지만, 실제로는 직렬로 연결된 두 개의 서로 다른 저항 성분입니다."
               if R_SEI_for_c > 0 else "")
        )

        ncol1, ncol2 = st.columns(2)
        ncol1.metric("C_SEI (역산)", f"{C_SEI*1e6:.2f} µF" if C_SEI > 0 else "N/A")
        ncol2.metric("C_dl (역산)", f"{C_dl*1e6:.2f} µF")

# ── Sweep ─────────────────────────────────────────────────────
if sweep_btn:
    with st.spinner("Running 20 conditions..."):
        rows = sweep(preset, Q_tgt_input,
                     duty_belief=duty_belief_input, w_time=w_time_input)

    wT = w_time_input
    wq = (1.0 - wT) / 2.0
    st.subheader("Sweep Results  (20 conditions, pulse included in ranking)")
    st.caption(
        f"점수식: ICE {wq*100:.0f} : R_SEI {wq*100:.0f} : 공정시간 {wT*100:.0f} "
        f"· 펄스 이득 가정 {duty_belief_pct} % "
        + ("· **가정 포함 결과**" if duty_belief_pct > 0 else "· 문헌 지지 값만 사용")
    )

    if duty_belief_pct > 0:
        st.warning(
            f"펄스 행에 duty 보정이 {duty_belief_pct} % 강도로 적용되어 있습니다. "
            f"이 보정계수는 Si 프리리튬화 실측이 아니라 플레이스홀더입니다 "
            f"(신뢰도 {DUTY_CONF}). 아래 순위는 '이 정도 효과가 실제로 있다면' 이라는 "
            f"조건부 결과로만 읽어야 합니다."
        )
    else:
        st.info(
            "펄스 이득 가정이 0 %이므로 펄스의 ICE·R_SEI는 같은 C-rate 연속 조건과 "
            "같은 값입니다. 그래서 펄스는 동점으로 나오며, 이는 열등하다는 뜻이 아니라 "
            "구분할 실측 데이터가 아직 없다는 뜻입니다. 오른쪽 두 열은 "
            "'같은 시간 안에 돌릴 수 있는 최선의 연속 조건을 이기려면 얼마나 좋아져야 하는가'를 "
            "가정 없이 역산한 값입니다."
        )

    import pandas as pd
    rows_sorted = sorted(
        rows, key=lambda r: (-r["score"] if not np.isnan(r["score"]) else 1e9))

    def _be(r, key, fmt):
        if r["mode"] != "Pulse" or not r.get("be") or key not in r["be"]:
            return "—"
        if abs(r["be"]["gap"]) < 1e-9:
            return "동점"
        if r["be"]["gap"] < 0:
            return "이미 우세"
        return fmt.format(r["be"][key])

    df = pd.DataFrame([{
        "Condition": r["label"],
        "Support": r["support"],
        "C-rate": r["C_rate"],
        "Mode": r["mode"],
        "t_on [min]": r["t_on"],
        "t_off [min]": r["t_off"],
        "Duty": f"{r['duty']:.3f}",
        "Time [h]": f"{r['time_h']:.2f}",
        "ICE [%]": f"{r['ICE']:.2f}",
        "R_SEI [Ω]": f"{r['RSEI']:.2f}" if r["RSEI_valid"] else "N/A",
        "Rct [Ω]": f"{r['Rct']:.2f}",
        "Score": f"{r['score']:.4f}" if not np.isnan(r["score"]) else "—",
        "이기려면 ICE [%p]": _be(r, "d_ice_pp", "{:+.2f}"),
        "이기려면 R_SEI ↓ [%]": _be(r, "d_rsei_pct", "{:.1f}"),
        "비교 대상": (r["be"]["rival"] if r["mode"] == "Pulse" and r.get("be") else "—"),
        "OverPL": "⚠" if r["overPL"] else "",
    } for r in rows_sorted])

    st.dataframe(df, use_container_width=True, height=420)

    ranked = [r for r in rows
              if r["RSEI_valid"] and not r["overPL"] and not np.isnan(r["score"])]
    if ranked:
        # 동점이면 문헌 지지 조건(연속)을, 그래도 같으면 짧은 쪽을 택한다
        best = max(ranked, key=lambda r: (round(r["score"], 9),
                                          r["mode"] == "Continuous",
                                          -r["time_h"]))
        tag = "  ·  ⚠ 가정 포함" if best["mode"] == "Pulse" and duty_belief_pct > 0 else ""
        st.success(f"✅ Best condition: **{best['label']}**  |  "
                   f"ICE = {best['ICE']:.2f}%  |  "
                   f"R_SEI = {best['RSEI']:.2f} Ω  |  "
                   f"Time = {best['time_h']:.2f} h  |  "
                   f"Score = {best['score']:.4f}{tag}")

        core_only = [r for r in ranked if r["mode"] == "Continuous"]
        if core_only:
            bc = max(core_only, key=lambda r: r["score"])
            if bc["label"] != best["label"]:
                st.caption(f"문헌 지지 조건(연속)만 볼 때의 1위: **{bc['label']}** "
                           f"(Score {bc['score']:.4f}, {bc['time_h']:.2f} h)")

    # 실험 우선순위 — 가정 없이 계산되는 값만 사용
    cand = [r for r in rows
            if r["mode"] == "Pulse" and r.get("be") and "d_ice_pp" in r["be"]
            and r["be"]["gap"] > 0]
    if cand:
        # 실험을 실제로 돌릴 수 있느냐(추가 시간)가 먼저고,
        # 그 다음이 판가름 나기 쉬우냐(필요 효과 크기)다.
        cand.sort(key=lambda r: (r["be"]["extra_h"], r["be"]["d_ice_pp"]))
        top = cand[:3]
        lines = " · ".join(
            f"**{r['label']}** (추가 {r['be']['extra_h']:.2f} h, "
            f"ICE +{r['be']['d_ice_pp']:.2f} %p면 역전)" for r in top)
        st.markdown(f"🔬 **먼저 측정할 펄스 조건** — {lines}")
        st.caption("추가 공정시간이 적은 순, 같으면 역전에 필요한 효과가 작은 순입니다. "
                   "가정이 아니라 역산 값이라 실험 계획에 그대로 쓸 수 있습니다.")

    if any((not np.isnan(r["score"])) and r["score"] > 1.0 for r in rows):
        st.caption("※ 점수가 1을 넘는 행이 있습니다 — 정규화 기준이 연속 조건 4개의 "
                   "범위라서, 그 범위를 벗어나면 1을 초과합니다. 오류가 아니라 "
                   "'문헌으로 확인된 구간 밖'이라는 표시입니다.")

    cont = sorted([r for r in rows if r["mode"] == "Continuous"], key=lambda r: r["C_rate"])
    cr_vals = [r["C_rate"] for r in cont]
    ice_vals = [r["ICE"] for r in cont]
    rsei_vals = [r["RSEI"] for r in cont]
    rct_vals = [r["Rct"] for r in cont]

    fig2 = make_subplots(rows=1, cols=3, subplot_titles=["ICE [%]", "R_SEI [Ω]", "Rct [Ω]"])
    fig2.add_trace(go.Scatter(x=cr_vals, y=ice_vals, mode="lines+markers",
                              line=dict(color="#2563EB", width=2)), row=1, col=1)
    fig2.add_trace(go.Scatter(x=cr_vals, y=rsei_vals, mode="lines+markers",
                              line=dict(color="#DC2626", width=2)), row=1, col=2)
    fig2.add_trace(go.Scatter(x=cr_vals, y=rct_vals, mode="lines+markers",
                              line=dict(color="#16A34A", width=2)), row=1, col=3)

    fig2.update_xaxes(title_text="C-rate", type="log")
    fig2.update_layout(height=340, showlegend=False,
                       title_text="Continuous conditions at matched Q_PL",
                       margin=dict(t=60, b=30))
    st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.caption("Model scope: 0.02–0.10 C · Q_PL up to 3000 mAh/g · "
           "Pulse mode exploratory only · Nyquist curves are exploratory "
           "(characteristic-frequency assumption, not digitized from source literature)")
