# app.py
import streamlit as st
import time
import math
import matplotlib.pyplot as plt
from dorm_backend import DORMFramework
import numpy as np

st.set_page_config(page_title="DORM - Integrated", layout="wide")

# ---------- App State ----------
if "dorm" not in st.session_state:
    st.session_state.dorm = DORMFramework(n_satellites=6)
    st.session_state.running = False
    st.session_state.time = 0
    st.session_state.logs = []

dorm = st.session_state.dorm

# ---------- Controls ----------
st.title("🛰 DORM Simulator — Integrated Backend + UI")
col_ctrl, col_vis = st.columns([1, 3])

with col_ctrl:
    st.header("Controls")
    if st.button("Prepare & Train (quick)"):
        with st.spinner("Generating dataset and training ML... (quick mode)"):
            info = dorm.prepare_and_train(dataset_size=2000, n_estimators=50)
        st.success(f"Trained. Accuracy: {info['accuracy']:.3f}, train time: {info['train_time_s']:.1f}s")
        st.session_state.logs.insert(0, f"Trained ML model — acc {info['accuracy']:.3f}")

    if st.button("Inject random attack"):
        info = dorm.inject_random_attack()
        st.session_state.logs.insert(0, f"Injected {info['type']} on S{info['sat']}")

    start_stop = st.button("Start / Stop Simulation")
    if start_stop:
        st.session_state.running = not st.session_state.running

    if st.button("Reset network"):
        st.session_state.dorm = DORMFramework(n_satellites=6)
        st.session_state.running = False
        st.session_state.time = 0
        st.session_state.logs.insert(0, "Network reset")

    st.markdown("---")
    st.write("Logs (recent):")
    for l in st.session_state.logs[:12]:
        st.write(l)

# ---------- Visualization & step ----------
with col_vis:
    st.subheader("Orbital Visualization")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_facecolor("#0f172a")

    # draw Earth
    earth = plt.Circle((0, 0), 0.5, color="royalblue")
    ax.add_artist(earth)

    # orbit circle
    orbit = plt.Circle((0, 0), 2.0, color="gray", fill=False, linestyle="--")
    ax.add_artist(orbit)

    # update simulation if running
    if st.session_state.running:
        st.session_state.time += 1
        # AUTO inject occasional random attack
        if np.random.rand() < 0.05:
            a = dorm.inject_random_attack()
            st.session_state.logs.insert(0, f"[t{st.session_state.time}] Injected {a['type']} on S{a['sat']}")

        # For each satellite, run a sample detection using a synthetic sample (quick)
        for sat in dorm.network.satellites:
            # build a "normal" sample baseline
            sample = np.array([30, 1, -70, 0, 50, 100, 300, 1, 2, 0])
            try:
                res = dorm.detect_and_mitigate_sample(sample, sat.id)
                if res['detected']:
                    st.session_state.logs.insert(0, f"[t{st.session_state.time}] Detected {res['type']} on S{sat.id} (mitigated={res['mitigated']})")
            except Exception as e:
                # model may not be trained yet
                pass

        time.sleep(0.2)

    # draw satellites
    for s in dorm.network.satellites:
        angle = (s.id / len(dorm.network.satellites)) * 2 * math.pi + (st.session_state.time * 0.08)
        x = math.cos(angle) * 2.0
        y = math.sin(angle) * 2.0
        color = "red" if s.status == 'under_attack' else "limegreen"
        ax.scatter(x, y, s=140, color=color, edgecolors='w')
        ax.text(x, y+0.15, f"S{s.id}", color="white", fontsize=9, ha='center')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.axis("off")
    st.pyplot(fig)

# ---------- Metrics ----------
st.markdown("---")
st.header("Metrics & Network State")
state = dorm.get_network_state()
cols = st.columns(4)
cols[0].metric("Satellites", len(state))
cols[1].metric("Total Active Threats", sum(len(s['threats']) for s in state))
cols[2].metric("Time", st.session_state.time)
cols[3].metric("ML trained", "Yes" if dorm.ml.model is not None else "No")

st.subheader("Satellite details")
for s in state:
    st.write(f"S{s['id']}: status={s['status']} threats={s['threats']} resources P{round(s['resources']['power'],1)} B{round(s['resources']['bandwidth'],1)} C{round(s['resources']['compute'],1)}")
