"""
app.py
------
Interactive Streamlit dashboard for the Smart Parking AI project.
Run with: streamlit run app.py
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from parking_model import ParkingLot, create_vehicles
from bfs_algorithm import bfs_allocate
from astar_algorithm import astar_allocate
from hill_climbing import hill_climbing_allocate
from comparison import (
    run_comparison,
    results_to_dataframe,
    run_benchmark_suite,
    create_comparison_charts,
)

st.set_page_config(
    page_title="Smart Parking AI",
    page_icon="🅿️",
    layout="wide",
)

ALL_SLOT_IDS = [f"P{i}" for i in range(1, 21)]


# ---------- Session state defaults ----------
if "occupied_slots" not in st.session_state:
    st.session_state.occupied_slots = ["P2", "P7", "P12"]
if "num_cars" not in st.session_state:
    st.session_state.num_cars = 2
if "num_bikes" not in st.session_state:
    st.session_state.num_bikes = 1


def sort_slot_ids(slot_ids):
    return sorted(set(slot_ids), key=lambda sid: int(sid[1:]))


def apply_preset(occupied, cars, bikes):
    st.session_state.occupied_slots = sort_slot_ids(occupied)
    st.session_state.num_cars = cars
    st.session_state.num_bikes = bikes


def commit_allocation(new_slot_ids):
    updated = list(st.session_state.occupied_slots) + list(new_slot_ids)
    st.session_state.occupied_slots = sort_slot_ids(updated)


# ---------- Visual parking grid helper ----------
def render_lot_grid(lot, allocation=None, compact=False):
    """Render the 4x5 parking lot as a color-coded HTML grid."""
    slot_to_vehicle = {}
    if allocation:
        slot_to_vehicle = {sid: vid for vid, sid in allocation.items()}

    pad = "6px" if compact else "10px"
    title_size = "13px" if compact else "15px"
    sub_size = "11px" if compact else "12px"

    parts = [
        "<div style='margin-bottom:8px; display:flex; align-items:center; gap:8px;'>",
        "<span style='background:#1E3A8A; color:#FFFFFF; padding:4px 10px; "
        "border-radius:6px; font-weight:700; font-size:12px;'>ENTRANCE → (Next to P1)</span>",
        "</div>",
        "<div style='display:grid; grid-template-columns:repeat(5, minmax(0, 1fr)); gap:8px;'>",
    ]

    for r in range(1, lot.rows + 1):
        for c in range(1, lot.cols + 1):
            sid = f"P{(r - 1) * lot.cols + c}"
            slot = lot.slots[sid]

            if sid in slot_to_vehicle:
                bg, border, text_col = "#DBEAFE", "#2563EB", "#1E3A8A"
                status = f"NEW: {slot_to_vehicle[sid]}"
            elif slot.occupied:
                bg, border, text_col = "#FEE2E2", "#DC2626", "#991B1B"
                status = "Occupied"
            elif slot.slot_type == "bike":
                bg, border, text_col = "#FEF3C7", "#D97706", "#92400E"
                status = "Vacant (Bike)"
            else:
                bg, border, text_col = "#DCFCE7", "#16A34A", "#166534"
                status = "Vacant (Car)"

            zone_tag = "Bike" if slot.slot_type == "bike" else "Car"
            card = (
                f"<div style='background:{bg}; border:2px solid {border}; "
                f"border-radius:10px; padding:{pad}; text-align:center; color:{text_col};'>"
                f"<div style='display:flex; justify-content:space-between; font-size:{sub_size}; "
                f"opacity:0.85; font-weight:600;'>"
                f"<span>{zone_tag}</span><span>Cost {slot.distance_cost}</span></div>"
                f"<div style='font-size:{title_size}; font-weight:800; margin:4px 0 2px 0;'>{sid}</div>"
                f"<div style='font-size:{sub_size}; font-weight:700;'>{status}</div>"
                f"</div>"
            )
            parts.append(card)

    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_legend():
    st.markdown(
        "<div style='display:flex; flex-wrap:wrap; gap:12px; margin:10px 0 16px 0; font-size:13px;'>"
        "<span><span style='display:inline-block;width:12px;height:12px;background:#DCFCE7;border:2px solid #16A34A;border-radius:3px;margin-right:5px;'></span>Vacant Car (Rows 1–3)</span>"
        "<span><span style='display:inline-block;width:12px;height:12px;background:#FEF3C7;border:2px solid #D97706;border-radius:3px;margin-right:5px;'></span>Vacant Bike (Row 4)</span>"
        "<span><span style='display:inline-block;width:12px;height:12px;background:#FEE2E2;border:2px solid #DC2626;border-radius:3px;margin-right:5px;'></span>Occupied</span>"
        "<span><span style='display:inline-block;width:12px;height:12px;background:#DBEAFE;border:2px solid #2563EB;border-radius:3px;margin-right:5px;'></span>Newly Allocated</span>"
        "</div>",
        unsafe_allow_html=True,
    )


def build_current_lot_and_vehicles():
    lot = ParkingLot()
    lot.set_occupied_slots(st.session_state.occupied_slots)
    vehicles = create_vehicles(st.session_state.num_cars, st.session_state.num_bikes)
    return lot, vehicles


def run_selected_algorithm(algo_name, lot, vehicles, hc_seed, hc_neighborhood):
    if algo_name == "BFS":
        return bfs_allocate(lot, vehicles)
    if algo_name == "A*":
        return astar_allocate(lot, vehicles)
    return hill_climbing_allocate(
        lot, vehicles, seed=hc_seed, neighborhood=hc_neighborhood
    )


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Parking Lot Setup")

    st.caption("Quick Presets")
    c1, c2 = st.columns(2)
    c1.button("Light Load", on_click=apply_preset, args=(["P2", "P7", "P12"], 2, 1))
    c2.button("BFS Trap", on_click=apply_preset, args=(["P1", "P2", "P3", "P4"], 2, 0))

    c3, c4 = st.columns(2)
    c3.button("Heavy Load", on_click=apply_preset, args=(["P2", "P4", "P8", "P12", "P17"], 4, 2))
    c4.button("No Bike Slots", on_click=apply_preset, args=(["P16", "P17", "P18", "P19", "P20"], 1, 1))

    st.button("Reset Empty Lot", on_click=apply_preset, args=([], 2, 1))

    st.divider()

    st.multiselect(
        "Occupied Slots",
        options=ALL_SLOT_IDS,
        key="occupied_slots",
        help="Rows 1–3 (P1–P15) are car slots; Row 4 (P16–P20) are bike slots.",
    )

    st.subheader("Incoming Vehicles")
    st.number_input("Cars (C1, C2, ...)", min_value=0, max_value=15, step=1, key="num_cars")
    st.number_input("Bikes (B1, B2, ...)", min_value=0, max_value=5, step=1, key="num_bikes")

    st.divider()
    st.subheader("Algorithm Settings")
    selected_algo = st.selectbox("Algorithm (Simulator Tab)", ["A*", "BFS", "Hill Climbing"])
    hc_seed = st.number_input("Hill Climbing Seed", min_value=0, max_value=9999, value=42, step=1)
    hc_neighborhood = st.selectbox(
        "Hill Climbing Moves",
        ["any", "adjacent"],
        help="'any' moves to any free compatible slot; 'adjacent' restricts moves to touching slots.",
    )


# ---------- Main Header ----------
st.title("🅿️ Smart Parking Slot Allocation — AI Search")
st.caption("Comparing Breadth-First Search (BFS), A* Search, and Hill Climbing on a 4×5 multi-vehicle lot.")

lot, vehicles = build_current_lot_and_vehicles()

tab1, tab2, tab3 = st.tabs([
    "Live Simulator",
    "Compare Algorithms",
    "Benchmark Suite & Local Optimum Demo",
])


# ---------- TAB 1: Live Simulator ----------
with tab1:
    result = run_selected_algorithm(selected_algo, lot, vehicles, hc_seed, hc_neighborhood)

    # Top metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Algorithm", result["algorithm"])
    m2.metric("Total Cost", result["total_cost"] if result["success"] else "N/A")
    m3.metric("Explored States", f"{result['explored_states']:,}")
    m4.metric("Time (ms)", f"{result['time_ms']:.3f}")

    if result["success"]:
        st.success(result["message"])
    else:
        st.error(result["message"])

    left_col, right_col = st.columns([3, 2], gap="large")

    with left_col:
        st.subheader("Parking Lot Map")
        render_lot_grid(lot, result["allocation"])
        render_legend()

    with right_col:
        st.subheader("Allocation Details")
        vacant_cars = sum(1 for s in lot.get_vacant_slots() if s.slot_type == "car")
        vacant_bikes = sum(1 for s in lot.get_vacant_slots() if s.slot_type == "bike")
        st.write(
            f"**Vacant before allocation:** {vacant_cars} car slots, {vacant_bikes} bike slots  \n"
            f"**Waiting vehicles:** {st.session_state.num_cars} car(s), {st.session_state.num_bikes} bike(s)"
        )

        if result["success"] and result["allocation"]:
            alloc_rows = []
            for v in vehicles:
                sid = result["allocation"][v.vehicle_id]
                slot = lot.get_slot(sid)
                alloc_rows.append({
                    "Vehicle": v.vehicle_id,
                    "Type": v.vehicle_type.capitalize(),
                    "Assigned Slot": sid,
                    "Row, Col": f"({slot.row}, {slot.col})",
                    "Distance Cost": slot.distance_cost,
                })
            st.dataframe(pd.DataFrame(alloc_rows), hide_index=True)

            st.button(
                "Commit Allocation to Lot",
                on_click=commit_allocation,
                args=(list(result["allocation"].values()),),
                help="Marks the newly assigned slots as occupied so you can simulate the next wave of arrivals.",
            )
        elif result["success"]:
            st.info("No vehicles are waiting to be parked.")


# ---------- TAB 2: Compare Algorithms (Current Setup) ----------
with tab2:
    st.subheader("Side-by-Side Comparison on Current Lot Setup")
    comp_results = [
        bfs_allocate(lot, vehicles),
        astar_allocate(lot, vehicles),
        hill_climbing_allocate(lot, vehicles, seed=hc_seed, neighborhood=hc_neighborhood),
    ]
    comp_df = results_to_dataframe(comp_results)
    st.dataframe(comp_df, hide_index=True)

    succeeded = [r for r in comp_results if r["success"]]
    if succeeded:
        min_cost = min(r["total_cost"] for r in succeeded)
        optimal_algos = [r["algorithm"] for r in succeeded if r["total_cost"] == min_cost]
        st.info(f"**Lowest cost on this setup:** {min_cost} — achieved by {', '.join(optimal_algos)}.")

        # Quick live bar charts for Cost and Explored States
        ch1, ch2 = st.columns(2)
        algos = [r["algorithm"] for r in succeeded]
        costs = [r["total_cost"] for r in succeeded]
        states = [r["explored_states"] for r in succeeded]
        bar_colors = ["#4C78A8", "#54A24B", "#F58518"][:len(algos)]

        with ch1:
            fig_c, ax_c = plt.subplots(figsize=(5, 3.4))
            bars = ax_c.bar(algos, costs, color=bar_colors, width=0.5)
            ax_c.set_title("Total Distance Cost (Lower is Better)")
            ax_c.set_ylabel("Total Cost")
            ax_c.grid(axis="y", linestyle="--", alpha=0.4)
            ax_c.bar_label(bars, padding=3)
            fig_c.tight_layout()
            st.pyplot(fig_c)
            plt.close(fig_c)

        with ch2:
            fig_s, ax_s = plt.subplots(figsize=(5, 3.4))
            bars = ax_s.bar(algos, states, color=bar_colors, width=0.5)
            ax_s.set_title("Explored States")
            ax_s.set_ylabel("States")
            ax_s.grid(axis="y", linestyle="--", alpha=0.4)
            if max(states) > 20 * max(1, min(states)):
                ax_s.set_yscale("log")
            ax_s.bar_label(bars, padding=3)
            fig_s.tight_layout()
            st.pyplot(fig_s)
            plt.close(fig_s)

        st.subheader("Visual Layouts Chosen by Each Algorithm")
        g1, g2, g3 = st.columns(3)
        for col, res in zip([g1, g2, g3], comp_results):
            with col:
                cost_label = res["total_cost"] if res["success"] else "Failed"
                st.markdown(f"**{res['algorithm']}** — Cost: `{cost_label}` | States: `{res['explored_states']:,}`")
                render_lot_grid(lot, res["allocation"], compact=True)
    else:
        st.warning("No algorithm could find a valid allocation for the current setup.")


# ---------- TAB 3: Benchmark Suite & Local Optimum Demo ----------
with tab3:
    st.subheader("5-Scenario Benchmark Suite (from comparison.py)")
    bench_df = run_benchmark_suite(hc_seed=hc_seed)
    st.dataframe(
        bench_df[["Scenario", "Algorithm", "Success", "Total Cost", "Explored States", "Time (ms)", "Allocation"]],
        hide_index=True,
    )

    csv_bytes = bench_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Benchmark Results (CSV)",
        data=csv_bytes,
        file_name="comparison_results.csv",
        mime="text/csv",
    )

    st.subheader("Benchmark Charts")
    charts = create_comparison_charts(bench_df)
    for filename, fig in charts.items():
        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    st.subheader("Viva Demo: Hill Climbing Local Optimum vs. A* Global Optimum")
    st.write(
        "Scenario: **P6** is occupied, 1 car (**C1**) starts at **P11** (cost 2), and Hill Climbing can only move to "
        "**adjacent** slots. Its only touching slots are P6 (occupied), P12 (cost 3, worse), and P16 (bike slot), "
        "so it gets trapped at cost 2 while A* finds the global optimum **P1** (cost 0)."
    )

    demo_lot = ParkingLot()
    demo_lot.set_occupied_slots(["P6"])
    demo_vehicles = create_vehicles(1, 0)
    demo_hc = hill_climbing_allocate(
        demo_lot, demo_vehicles, start={"C1": "P11"}, neighborhood="adjacent"
    )
    demo_astar = astar_allocate(demo_lot, demo_vehicles)

    d1, d2 = st.columns(2, gap="large")
    with d1:
        st.markdown(f"**Hill Climbing (Adjacent Moves)** — Stuck at `{demo_hc['allocation']}` | Cost: **{demo_hc['total_cost']}**")
        render_lot_grid(demo_lot, demo_hc["allocation"], compact=True)
    with d2:
        st.markdown(f"**A* Search (Global Optimum)** — Finds `{demo_astar['allocation']}` | Cost: **{demo_astar['total_cost']}**")
        render_lot_grid(demo_lot, demo_astar["allocation"], compact=True)
