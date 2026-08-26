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


def simulate(preset, C_rate, pl_mode, t_on, t_off, Q_tgt):
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


def sweep(preset, Q_target):
    """Run 20-condition sweep. Returns list of result dicts."""
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
            r = simulate(preset, cr, 1, ton, toff, Q_target)
            r["label"] = f"{cr:.2f}C pulse {ton // 60}/{toff // 60}min"
            r["C_rate"] = cr
            r["mode"] = "Pulse"
            r["t_on"] = ton // 60
            r["t_off"] = toff // 60
            r["support"] = "Exploratory"
            rows.append(r)

    core = [r for r in rows if r["mode"] == "Continuous" and r["RSEI_valid"]]
    ICE_vals = [r["ICE"] for r in core]
    RSEI_vals = [r["RSEI"] for r in core]

    ICE_min, ICE_max = min(ICE_vals), max(ICE_vals)
    RSEI_min, RSEI_max = min(RSEI_vals), max(RSEI_vals)

    for r in rows:
        if not r["RSEI_valid"] or r["mode"] == "Pulse":
            r["score"] = float("nan")
            continue
        s_ice = (r["ICE"] - ICE_min) / (ICE_max - ICE_min) if ICE_max > ICE_min else 1.0
        s_rsei = (RSEI_max - r["RSEI"]) / (RSEI_max - RSEI_min) if RSEI_max > RSEI_min else 1.0
        r["score"] = w_ICE * s_ice + w_RSEI * s_rsei

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

    run_btn = st.button("▶  Run", use_container_width=True)
    sweep_btn = st.button("⟳  Sweep (20 conditions)", use_container_width=True)

# ── Run single condition ──────────────────────────────────────
if run_btn:
    pl_mode = 1 if mode_input == "Pulse" else 0
    res = simulate(preset, C_rate_input, pl_mode,
                   t_on_input * 60, t_off_input * 60,
                   Q_tgt=Q_tgt_input)

    if res["RSEI_is_extrapolated"]:
        st.info("R_SEI 값은 Q_PL = 0에서 외삽된 값입니다 (실측 아님).")

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
        rows = sweep(preset, Q_tgt_input)

    st.subheader("Sweep Results  (sorted by score, continuous only ranked)")

    import pandas as pd
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
        "OverPL": "⚠" if r["overPL"] else "",
    } for r in rows])

    st.dataframe(df, use_container_width=True, height=420)

    core_rows = [r for r in rows
                 if r["mode"] == "Continuous"
                 and r["RSEI_valid"]
                 and not r["overPL"]
                 and not np.isnan(r["score"])]
    if core_rows:
        best = max(core_rows, key=lambda r: r["score"])
        st.success(f"✅ Best condition: **{best['label']}**  |  "
                   f"ICE = {best['ICE']:.2f}%  |  "
                   f"R_SEI = {best['RSEI']:.2f} Ω  |  "
                   f"Score = {best['score']:.4f}")

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
