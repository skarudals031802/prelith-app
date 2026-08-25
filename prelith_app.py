import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# PARAMETERS  (params_Si_prelithiation.m v4)
# ============================================================

Q_ref    = 3579.0   # [mAh/g]
Q_target = 3000.0   # [mAh/g]
R0_fixed = 3.5      # [Ohm]

Q_PL_bp  = np.array([0, 500, 1050, 1500, 2000, 2500, 3000], dtype=float)
ICE_base = np.array([51.0, 69.2, 69.2, 84.6, 83.0, 89.0, 88.2])
Rct_base = np.array([35.5, 24.6, 18.5, 13.2,  8.4,  5.9,  6.9])

Q_PL_RSEI_bp = np.array([500, 1050, 1500, 2000, 2500, 3000], dtype=float)
RSEI_base    = np.array([80.1, 75.9, 68.5, 59.1, 50.9, 54.3])

C_rate_bp  = np.array([0.02, 0.05, 0.08, 0.10])
f_rate_RSEI = np.array([94.0, 86.6, 93.6, 112.4]) / 86.6
f_rate_ICE  = np.array([99.8, 99.5, 98.8,  99.2]) / 99.5
f_rate_Rct  = np.ones(4)

w_ICE  = 0.5
w_RSEI = 0.5

ICE_overPL_threshold = 100.0


# ============================================================
# SIMULATION CORE
# ============================================================

def interp_clip(x, xp, fp):
    """Linear interpolation with clipping at boundaries."""
    return float(np.interp(x, xp, fp,
                           left=fp[0], right=fp[-1]))


def simulate(C_rate, pl_mode, t_on, t_off, Q_tgt=Q_target):
    """
    Returns dict with ICE, RSEI, Rct, R0, time_h, Q_PL,
    RSEI_valid, overPL at the moment Q_PL reaches Q_tgt.
    """
    duty = 1.0 if pl_mode == 0 else t_on / (t_on + t_off)

    I_peak = C_rate * Q_ref   # [mA/g]
    I_avg  = I_peak * duty    # average current

    # Time to reach Q_target [s]
    t_needed = Q_tgt / I_avg * 3600.0

    # Rate factors (clipped at boundary)
    f_ICE  = interp_clip(C_rate, C_rate_bp, f_rate_ICE)
    f_RSEI = interp_clip(C_rate, C_rate_bp, f_rate_RSEI)
    f_Rct  = 1.0  # disabled (use_rate_Rct = false)

    # Base values at Q_tgt
    ICE_val  = interp_clip(Q_tgt, Q_PL_bp, ICE_base)  * f_ICE
    Rct_val  = interp_clip(Q_tgt, Q_PL_bp, Rct_base)  * f_Rct

    RSEI_valid = (Q_tgt >= Q_PL_RSEI_bp[0])
    if RSEI_valid:
        RSEI_val = interp_clip(Q_tgt, Q_PL_RSEI_bp, RSEI_base) * f_RSEI
    else:
        RSEI_val = float('nan')

    time_h = t_needed / 3600.0

    return {
        'ICE':        ICE_val,
        'RSEI':       RSEI_val,
        'Rct':        Rct_val,
        'R0':         R0_fixed,
        'time_h':     time_h,
        'Q_PL':       Q_tgt,
        'RSEI_valid': RSEI_valid,
        'overPL':     ICE_val >= ICE_overPL_threshold,
        'duty':       duty,
    }


def get_curves(C_rate, pl_mode, t_on, t_off):
    """Return ICE, RSEI, Rct vs Q_PL curves for plotting."""
    Q_arr = np.linspace(0, Q_target, 200)

    f_ICE  = interp_clip(C_rate, C_rate_bp, f_rate_ICE)
    f_RSEI = interp_clip(C_rate, C_rate_bp, f_rate_RSEI)

    ICE_arr  = np.array([interp_clip(q, Q_PL_bp, ICE_base) * f_ICE  for q in Q_arr])
    Rct_arr  = np.array([interp_clip(q, Q_PL_bp, Rct_base)           for q in Q_arr])

    Q_rsei = Q_arr[Q_arr >= Q_PL_RSEI_bp[0]]
    RSEI_arr = np.array([interp_clip(q, Q_PL_RSEI_bp, RSEI_base) * f_RSEI for q in Q_rsei])

    return Q_arr, ICE_arr, Rct_arr, Q_rsei, RSEI_arr


def sweep():
    """Run 20-condition sweep. Returns list of result dicts."""
    C_list   = [0.02, 0.05, 0.08, 0.10]
    t_on_s   = [10*60, 10*60, 20*60, 30*60]
    t_off_s  = [60*60, 30*60, 20*60, 10*60]

    rows = []

    # Continuous
    for cr in C_list:
        r = simulate(cr, 0, 0, 0)
        r['label']   = f'{cr:.2f}C continuous'
        r['C_rate']  = cr
        r['mode']    = 'Continuous'
        r['t_on']    = 0
        r['t_off']   = 0
        r['support'] = 'Core'
        rows.append(r)

    # Pulse
    for cr in C_list:
        for ton, toff in zip(t_on_s, t_off_s):
            r = simulate(cr, 1, ton, toff)
            r['label']   = f'{cr:.2f}C pulse {ton//60}/{toff//60}min'
            r['C_rate']  = cr
            r['mode']    = 'Pulse'
            r['t_on']    = ton // 60
            r['t_off']   = toff // 60
            r['support'] = 'Exploratory'
            rows.append(r)

    # Scoring (core conditions only for normalization)
    core = [r for r in rows if r['mode'] == 'Continuous' and r['RSEI_valid']]
    ICE_vals  = [r['ICE']  for r in core]
    RSEI_vals = [r['RSEI'] for r in core]

    ICE_min, ICE_max   = min(ICE_vals),  max(ICE_vals)
    RSEI_min, RSEI_max = min(RSEI_vals), max(RSEI_vals)

    for r in rows:
        if not r['RSEI_valid'] or r['mode'] == 'Pulse':
            r['score'] = float('nan')
            continue
        s_ice  = (r['ICE']  - ICE_min)  / (ICE_max  - ICE_min)  if ICE_max  > ICE_min  else 1.0
        s_rsei = (RSEI_max - r['RSEI']) / (RSEI_max - RSEI_min) if RSEI_max > RSEI_min else 1.0
        r['score'] = w_ICE * s_ice + w_RSEI * s_rsei

    return rows


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(page_title="Si Pre-lithiation Optimizer",
                   layout="wide")

st.title("Si Anode Pre-lithiation Optimizer")
st.caption("Based on Zhao 2025 [A] (anchor) · Li 2023 [B] (rate correction)")

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.header("Conditions")

    Q_tgt_input = st.number_input(
        "Pre-lithiation target [mAh/g]",
        min_value=500.0, max_value=3000.0,
        value=3000.0, step=100.0)

    C_rate_input = st.number_input(
        "C-rate  (0.02 – 0.10)",
        min_value=0.02, max_value=0.10,
        value=0.05, step=0.01, format="%.3f")

    mode_input = st.radio("Mode", ["Continuous", "Pulse"])

    if mode_input == "Pulse":
        t_on_input  = st.number_input("t_on  [min]",  min_value=1, max_value=120, value=10)
        t_off_input = st.number_input("t_off [min]",  min_value=1, max_value=120, value=30)
    else:
        t_on_input  = 0
        t_off_input = 0

    run_btn   = st.button("▶  Run", use_container_width=True)
    sweep_btn = st.button("⟳  Sweep (20 conditions)", use_container_width=True)

# ── Run single condition ──────────────────────────────────────
if run_btn:
    pl_mode = 1 if mode_input == "Pulse" else 0
    res = simulate(C_rate_input, pl_mode,
                   t_on_input * 60, t_off_input * 60,
                   Q_tgt=Q_tgt_input)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ICE",        f"{res['ICE']:.2f} %",
                delta="⚠ over-prelithiation" if res['overPL'] else None)
    col2.metric("R_SEI",      f"{res['RSEI']:.2f} Ω" if res['RSEI_valid'] else "N/A (Q_PL too low)")
    col3.metric("Rct",        f"{res['Rct']:.2f} Ω")
    col4.metric("Time",       f"{res['time_h']:.2f} h")

    # Curves
    Q_arr, ICE_arr, Rct_arr, Q_rsei, RSEI_arr = get_curves(
        C_rate_input, pl_mode, t_on_input*60, t_off_input*60)

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["ICE [%]", "R_SEI [Ω]", "Rct [Ω]"])

    fig.add_trace(go.Scatter(x=Q_arr, y=ICE_arr,  name="ICE",
                             line=dict(color="#2563EB", width=2)), row=1, col=1)
    fig.add_vline(x=Q_tgt_input, line_dash="dash",
                  line_color="gray", row=1, col=1)

    fig.add_trace(go.Scatter(x=Q_rsei, y=RSEI_arr, name="R_SEI",
                             line=dict(color="#DC2626", width=2)), row=1, col=2)
    fig.add_vline(x=Q_tgt_input, line_dash="dash",
                  line_color="gray", row=1, col=2)

    fig.add_trace(go.Scatter(x=Q_arr, y=Rct_arr,  name="Rct",
                             line=dict(color="#16A34A", width=2)), row=1, col=3)
    fig.add_vline(x=Q_tgt_input, line_dash="dash",
                  line_color="gray", row=1, col=3)

    fig.update_xaxes(title_text="Q_PL [mAh/g]")
    fig.update_layout(height=340, showlegend=False,
                      margin=dict(t=40, b=30))
    st.plotly_chart(fig, use_container_width=True)

# ── Sweep ─────────────────────────────────────────────────────
if sweep_btn:
    with st.spinner("Running 20 conditions..."):
        rows = sweep()

    st.subheader("Sweep Results  (sorted by score, continuous only ranked)")

    import pandas as pd
    df = pd.DataFrame([{
        'Condition':  r['label'],
        'Support':    r['support'],
        'C-rate':     r['C_rate'],
        'Mode':       r['mode'],
        't_on [min]': r['t_on'],
        't_off [min]':r['t_off'],
        'Duty':       f"{r['duty']:.3f}",
        'Time [h]':   f"{r['time_h']:.2f}",
        'ICE [%]':    f"{r['ICE']:.2f}",
        'R_SEI [Ω]':  f"{r['RSEI']:.2f}" if r['RSEI_valid'] else 'N/A',
        'Rct [Ω]':    f"{r['Rct']:.2f}",
        'Score':      f"{r['score']:.4f}" if not np.isnan(r['score']) else '—',
        'OverPL':     '⚠' if r['overPL'] else '',
    } for r in rows])

    df_sorted = df.copy()
    st.dataframe(df_sorted, use_container_width=True, height=420)

    # Best core condition
    core_rows = [r for r in rows
                 if r['mode'] == 'Continuous'
                 and r['RSEI_valid']
                 and not r['overPL']
                 and not np.isnan(r['score'])]
    if core_rows:
        best = max(core_rows, key=lambda r: r['score'])
        st.success(f"✅ Best condition: **{best['label']}**  |  "
                   f"ICE = {best['ICE']:.2f}%  |  "
                   f"R_SEI = {best['RSEI']:.2f} Ω  |  "
                   f"Score = {best['score']:.4f}")

    # C-rate comparison plot (continuous)
    cont = sorted([r for r in rows if r['mode'] == 'Continuous'],
                  key=lambda r: r['C_rate'])
    cr_vals   = [r['C_rate'] for r in cont]
    ice_vals  = [r['ICE']    for r in cont]
    rsei_vals = [r['RSEI']   for r in cont]
    rct_vals  = [r['Rct']    for r in cont]

    fig2 = make_subplots(rows=1, cols=3,
                         subplot_titles=["ICE [%]", "R_SEI [Ω]", "Rct [Ω]"])
    fig2.add_trace(go.Scatter(x=cr_vals, y=ice_vals,
                              mode='lines+markers',
                              line=dict(color="#2563EB", width=2)), row=1, col=1)
    fig2.add_trace(go.Scatter(x=cr_vals, y=rsei_vals,
                              mode='lines+markers',
                              line=dict(color="#DC2626", width=2)), row=1, col=2)
    fig2.add_trace(go.Scatter(x=cr_vals, y=rct_vals,
                              mode='lines+markers',
                              line=dict(color="#16A34A", width=2)), row=1, col=3)

    fig2.update_xaxes(title_text="C-rate", type="log")
    fig2.update_layout(height=340, showlegend=False,
                       title_text="Continuous conditions at matched Q_PL",
                       margin=dict(t=60, b=30))
    st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.caption("Model scope: 0.02–0.10 C · Q_PL up to 3000 mAh/g · "
           "Pulse mode exploratory only (no direct literature validation)")
