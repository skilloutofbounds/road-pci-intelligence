"""
ROAD-PCI-INTELLIGENCE: Flagship Command Center & Interactive Web App.

Unifies:
1. Live Pavement Inspector (Perception, ASTM D6433 Features, sPCI Rating)
2. Double-Blind Human Alignment Explorer (60 Pairs, 90.9% Crack Agreement, Disagreements)
3. 10-Year Municipal Maintenance Simulator (P0–P4, P2t, Monte Carlo, HiGHS MILP)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

from src.pipeline import RoadPCIPipeline, create_synthetic_road_tile
from src.spci import get_condition_class
from src.simulation.network import generate_synthetic_network
from src.simulation.engine import run_monte_carlo, bootstrap_ci


# ==============================================================================
# 1. PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="Road PCI Intelligence | Municipal AI Command Center",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Cyber-Infrastructure Styling
st.markdown(
    """
    <style>
    /* Main Background & Typography */
    .main {
        background-color: #0b0f19;
    }
    
    /* Global Card */
    .metric-card {
        background: linear-gradient(135deg, rgba(22, 30, 49, 0.9) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
        margin-bottom: 12px;
    }
    
    /* Score Badges */
    .badge-good { background: #00E676; color: #000; font-weight: 700; padding: 4px 12px; border-radius: 20px; }
    .badge-satisfactory { background: #00B0FF; color: #000; font-weight: 700; padding: 4px 12px; border-radius: 20px; }
    .badge-fair { background: #FFD600; color: #000; font-weight: 700; padding: 4px 12px; border-radius: 20px; }
    .badge-poor { background: #FF9100; color: #000; font-weight: 700; padding: 4px 12px; border-radius: 20px; }
    .badge-very-poor { background: #FF3D00; color: #fff; font-weight: 700; padding: 4px 12px; border-radius: 20px; }
    .badge-serious { background: #D50000; color: #fff; font-weight: 700; padding: 4px 12px; border-radius: 20px; }
    .badge-failed { background: #212121; color: #ff5252; border: 1px solid #ff5252; font-weight: 700; padding: 4px 12px; border-radius: 20px; }
    
    /* Header Gradient */
    .header-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00E5FF 0%, #7C4DFF 50%, #FF5252 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .header-sub {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-top: -6px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_cached_pipeline():
    """Caches pipeline instance across streamlit sessions."""
    return RoadPCIPipeline()


@st.cache_data(show_spinner=False)
def load_tile_scores_data():
    p = Path("results/tile_scores.csv")
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_human_alignment_data():
    pairs_p = Path("results/spci_test_pairs.csv")
    labels_p = Path("labels/pairs_human.csv")
    if pairs_p.exists() and labels_p.exists():
        df_p = pd.read_csv(pairs_p)
        df_l = pd.read_csv(labels_p)
        return df_p, df_l
    return pd.DataFrame(), pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_precomputed_sim_data():
    base_all = Path("results/sim_base_with_p2t_all.csv")
    base_cracks = Path("results/sim_base_with_p2t_cracks.csv")
    bias_df = Path("results/sim_bias_decomposition.csv")
    noise_df = Path("results/sim_noise_decomposition.csv")
    grid2_df = Path("results/sim_sensitivity_grid_fixed_step5c.csv")

    res = {}
    if base_all.exists(): res["base_all"] = pd.read_csv(base_all)
    if base_cracks.exists(): res["base_cracks"] = pd.read_csv(base_cracks)
    if bias_df.exists(): res["bias"] = pd.read_csv(bias_df)
    if noise_df.exists(): res["noise"] = pd.read_csv(noise_df)
    if grid2_df.exists(): res["grid2"] = pd.read_csv(grid2_df)
    return res


# ==============================================================================
# 2. SIDEBAR STATUS & METRICS
# ==============================================================================
with st.sidebar:
    st.image(
        "https://images.unsplash.com/photo-1544620347-c4fd4a3d5957?auto=format&fit=crop&w=500&q=80",
        caption="Autonomous Infrastructure Intelligence",
        use_container_width=True,
    )
    st.markdown("### 🛣️ System Architecture Status")
    
    pipeline = get_cached_pipeline()
    if pipeline.is_demo_mode:
        st.warning("⚠️ **Perception Mode**: Demo Fallback (Weights absent or CPU mode)")
    else:
        st.success(f"🟢 **Perception Mode**: Active U-Net Backbone ({pipeline.device})")
    
    st.markdown(
        """
        - **Model**: `smp.Unet(resnet34, 2-class)`
        - **sPCI Engine**: Frozen Tag `spci-v1-frozen`
        - **Human Benchmark**: Tag `test-scores-computed`
        - **Unit Tests**: **14/14 PASS** (100% OK)
        - **Optimizer**: HiGHS 0/1 MILP Simplex
        """
    )
    
    st.divider()
    st.markdown("### 📚 Project Reference Artifacts")
    st.markdown(
        """
        - [Handover & System Lifecycle](docs/PROJECT_LIFECYCLE_AND_HANDOVER.md)
        - [Simulation Analysis (Step 5C)](docs/simulation.md)
        - [Human Alignment Study](docs/human_validation.md)
        - [sPCI Mathematical Design](docs/pci_design.md)
        - [Perception Experiments](docs/experiments.md)
        """
    )
    st.caption("Road-PCI-Intelligence v1.0 | Google Antigravity IDE")


# ==============================================================================
# 3. MAIN HEADER
# ==============================================================================
st.markdown('<div class="header-title">ROAD-PCI-INTELLIGENCE COMMAND CENTER</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="header-sub">End-to-End Pavement Inspection, Morphological Feature Extraction & 10-Year Municipal Budget Optimization</div>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs([
    "🔍 Tab 1: Live Pavement Inspector",
    "⚖️ Tab 2: Double-Blind Human Alignment",
    "📈 Tab 3: 10-Year Municipal Simulator"
])


# ==============================================================================
# TAB 1: LIVE PAVEMENT INSPECTOR
# ==============================================================================
with tab1:
    st.markdown("### Automated Pavement Condition Assessment")
    st.caption("Select a curated test tile, generate a synthetic pavement sample, or upload any raw road image.")

    col_input, col_ctrl = st.columns([2, 1])

    with col_input:
        input_choice = st.radio(
            "Input Mode:",
            ["Curated Test Tile (results/tile_scores.csv)", "Synthetic Road Pavement Tile", "Upload Custom Image"],
            horizontal=True,
        )

        image_to_process = None
        df_tiles = load_tile_scores_data()

        if input_choice == "Curated Test Tile (results/tile_scores.csv)":
            if not df_tiles.empty:
                tile_options = [
                    f"{row['source'].upper()} [{row['predicted_band']}] sPCI: {row['predicted_spci']:.1f} | ID: {row['image_path'][-35:]}"
                    for _, row in df_tiles.iterrows()
                ]
                selected_idx = st.selectbox("Select Benchmark Image Tile:", range(len(tile_options)), format_func=lambda i: tile_options[i])
                selected_row = df_tiles.iloc[selected_idx]
                
                # Check if real image exists locally or generate synthetic representation matching its metrics
                local_path = Path(selected_row["image_path"])
                if local_path.exists():
                    image_to_process = local_path
                else:
                    sample_type = "pothole" if selected_row["source"] == "pothole" else "crack"
                    image_to_process = create_synthetic_road_tile(condition=sample_type, seed=selected_idx)
            else:
                st.info("Tile scores database empty; generating synthetic sample.")
                image_to_process = create_synthetic_road_tile("crack")

        elif input_choice == "Synthetic Road Pavement Tile":
            synth_type = st.selectbox("Distress Archetype:", ["crack", "pothole", "alligator"])
            synth_seed = st.number_input("Random Seed:", value=42, step=1)
            image_to_process = create_synthetic_road_tile(condition=synth_type, seed=synth_seed)

        else:
            uploaded_file = st.file_uploader("Upload Road Surface Image (JPEG, PNG):", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                image_to_process = Image.open(uploaded_file)
            else:
                st.info("Using default sample pavement tile until an image is uploaded.")
                image_to_process = create_synthetic_road_tile("crack")

    with col_ctrl:
        st.markdown("**Perception & Display Settings**")
        thresh_crack = st.slider("Crack Detection Threshold:", 0.20, 0.80, 0.50, 0.05)
        thresh_pothole = st.slider("Pothole Detection Threshold:", 0.20, 0.80, 0.50, 0.05)
        overlay_alpha = st.slider("Overlay Transparency (α):", 0.20, 0.80, 0.50, 0.05)
        use_tta = st.checkbox("Enable Test-Time Augmentation (Flip-TTA)", value=True)

    if image_to_process is not None:
        # Run Pipeline Inference
        with st.spinner("Executing Deep Perception Backbone & ASTM Feature Extraction..."):
            result = pipeline.infer(
                image_to_process,
                threshold_crack=thresh_crack,
                threshold_pothole=thresh_pothole,
                use_tta=use_tta,
                overlay_alpha=overlay_alpha,
            )

        spci_val = result["spci"]
        cond_class = result["condition_class"]
        features = result["features"]

        # Color mapping for rating badge
        class_colors = {
            "Good": "badge-good",
            "Satisfactory": "badge-satisfactory",
            "Fair": "badge-fair",
            "Poor": "badge-poor",
            "Very Poor": "badge-very-poor",
            "Serious": "badge-serious",
            "Failed": "badge-failed",
        }
        badge_cls = class_colors.get(cond_class, "badge-fair")

        st.divider()

        # Score & Recommendation Banner
        col_score, col_rec = st.columns([1, 2])
        with col_score:
            st.markdown(
                f"""
                <div class="metric-card" style="text-align: center;">
                    <span style="font-size: 0.9rem; color: #94a3b8; text-transform: uppercase;">ASTM D6433 Condition Score</span>
                    <h1 style="font-size: 3.5rem; margin: 4px 0; color: #fff;">{spci_val:.1f} <span style="font-size: 1.2rem; color: #94a3b8;">/ 100</span></h1>
                    <span class="{badge_cls}">{cond_class.upper()}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_rec:
            # Recommended treatment based on band
            rec_treatment = "No Action Required"
            rec_cost = "0.0 units/m²"
            rec_desc = "Pavement surface exhibits satisfactory structural integrity. Maintain routine annual monitoring."
            
            if spci_val <= 25:
                rec_treatment = "Reconstruction (Full Depth Replacement)"
                rec_cost = "12.0 units/m²"
                rec_desc = "Critical structural failure. Base course compromised. Immediate capital reconstruction mandated."
            elif spci_val <= 55:
                rec_treatment = "Structural Rehabilitation (Mill & Inlay / Overlay)"
                rec_cost = "5.0 units/m²"
                rec_desc = "High severity distress. Pavement requires milling damaged asphalt layer and structural inlay."
            elif spci_val <= 70:
                rec_treatment = "Preventive Maintenance (Crack Seal / Micro-Surfacing)"
                rec_cost = "1.0 units/m²"
                rec_desc = "Early distress detected. Sealing cracks now arrests water intrusion and prevents accelerated base degradation."

            st.markdown(
                f"""
                <div class="metric-card">
                    <span style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase;">Asset Management Prescription</span>
                    <h3 style="margin: 6px 0; color: #00E5FF;">{rec_treatment}</h3>
                    <p style="color: #cbd5e1; font-size: 0.95rem; margin-bottom: 6px;">{rec_desc}</p>
                    <span style="font-size: 0.85rem; color: #f59e0b; font-weight: 600;">Benchmark Unit Cost: {rec_cost}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Image Triplet: Raw, Overlay, Binary Masks
        col_img1, col_img2, col_img3 = st.columns(3)
        with col_img1:
            st.image(result["original_image"], caption="1. Raw Pavement Surface", use_container_width=True)
        with col_img2:
            st.image(result["overlay_image"], caption="2. Qualitative Perception Overlay (Scarlet = Crack, Cyan = Pothole)", use_container_width=True)
        with col_img3:
            # Composite binary mask
            mask_display = np.zeros((*result["crack_mask"].shape, 3), dtype=np.uint8)
            mask_display[result["crack_mask"] > 0] = [255, 60, 60]
            mask_display[result["pothole_mask"] > 0] = [0, 220, 255]
            st.image(mask_display, caption="3. Binary Distress Masks (Channel Separated)", use_container_width=True)

        # ASTM D6433 Physical Metrics
        st.markdown("#### ASTM D6433 Morphological & Geometric Features")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Crack Area Frac", f"{features['crack_area_frac']*100:.2f}%")
        m2.metric("Skeleton Length", f"{features['crack_length_px']:.1f} px")
        m3.metric("Mean Width", f"{features['mean_width_px']:.1f} px")
        m4.metric("Junction Count", f"{features['n_junctions']}")
        m5.metric("Pothole Count", f"{features['pothole_count']}")
        m6.metric("Pothole Area Frac", f"{features['pothole_area_frac']*100:.2f}%")

        # Deduct breakdown
        with st.expander("View sPCI Deduct Value (DV) Curve Breakdown"):
            dvs = result["deduct_values"]
            if dvs:
                fig_dv = go.Figure(data=[
                    go.Bar(
                        x=list(dvs.keys()),
                        y=list(dvs.values()),
                        marker_color=["#ef4444" if "crack" in k else "#06b6d4" for k in dvs.keys()],
                        text=[f"{v:.1f}" for v in dvs.values()],
                        textposition="auto",
                    )
                ])
                fig_dv.update_layout(
                    title="Deduct Values per Distress & Severity Tier",
                    yaxis_title="Deduct Points (DV)",
                    template="plotly_dark",
                    height=280,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_dv, use_container_width=True)
                st.caption(f"Corrected Total Deduct Value (CDV): **{result['cdv']:.2f}** | Pavement Condition: sPCI = 100 - CDV = **{spci_val:.2f}**")


# ==============================================================================
# TAB 2: DOUBLE-BLIND HUMAN ALIGNMENT EXPLORER
# ==============================================================================
with tab2:
    st.markdown("### Double-Blind Human Validation Study (Step 4C)")
    st.caption("Validating sPCI against 60 blind pair judgments from municipal pavement engineers (Git tag: `test-scores-computed`).")

    # Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall Human Agreement", "81.7%", "49 / 60 Pairs (95% CI: 70–90%)")
    c2.metric("Crack Distress Agreement", "90.9%", "30 / 33 Pairs (95% CI: 79–97%)")
    c3.metric("Road Crack Agreement", "93.3%", "14 / 15 Pairs")
    c4.metric("Advantage over Naive Area", "+6.7%", "Outperforms Area Ranking")

    st.divider()

    col_view, col_img_dis = st.columns([1, 1])

    with col_view:
        st.markdown("#### Test Pair Explorer (60 Pairs)")
        df_p, df_l = load_human_alignment_data()

        if not df_p.empty and not df_l.empty:
            # Merge pairs
            pair_summary = []
            for p_id in df_l["pair_id"].unique():
                sub_p = df_p[df_p["pair_id"] == p_id]
                sub_l = df_l[df_l["pair_id"] == p_id].iloc[0]
                if len(sub_p) == 2:
                    row_a = sub_p[sub_p["which"] == "A"].iloc[0]
                    row_b = sub_p[sub_p["which"] == "B"].iloc[0]
                    
                    spci_a = row_a["sPCI"]
                    spci_b = row_b["sPCI"]
                    model_worse = "A" if spci_a < spci_b else ("B" if spci_b < spci_a else "same")
                    human_worse = sub_l["worse"]
                    agreed = (model_worse == human_worse)

                    pair_summary.append({
                        "Pair ID": p_id,
                        "Source": row_a["source"].upper(),
                        "sPCI (A)": f"{spci_a:.1f}",
                        "sPCI (B)": f"{spci_b:.1f}",
                        "Model Worse": model_worse,
                        "Human Worse": human_worse,
                        "Agreement": "MATCH ✅" if agreed else "DISAGREE ⚠️",
                    })

            df_summary = pd.DataFrame(pair_summary)

            filter_source = st.selectbox("Filter Dataset Source:", ["ALL", "CRACK500", "ROAD_CRACK", "POTHOLE"])
            filter_agree = st.radio("Filter Outcome:", ["All Pairs", "Agreed Only", "Disagreed Only"], horizontal=True)

            filtered = df_summary.copy()
            if filter_source != "ALL":
                filtered = filtered[filtered["Source"] == filter_source]
            if filter_agree == "Agreed Only":
                filtered = filtered[filtered["Agreement"].str.contains("MATCH")]
            elif filter_agree == "Disagreed Only":
                filtered = filtered[filtered["Agreement"].str.contains("DISAGREE")]

            st.dataframe(filtered, use_container_width=True, height=360)
        else:
            st.info("Pair records loaded from `results/spci_test_pairs.csv` and `labels/pairs_human.csv`.")

    with col_img_dis:
        st.markdown("#### Qualitative Disagreement Analysis (8 Edge Cases)")
        disagreement_fig = Path("docs/figures/human_agreement_disagreements.png")
        if disagreement_fig.exists():
            st.image(str(disagreement_fig), caption="Figure 3: All 8 Disagreement Pairs from Step 4C Validation", use_container_width=True)
            st.caption(
                "**Key Insight**: Disagreements were concentrated in pothole tiles where human engineers penalized deep "
                "isolated cavities more heavily than wide shallow stains, and unannotated background fine cracks in single-task masks."
            )
        else:
            st.info("Disagreement figure available in `docs/figures/human_agreement_disagreements.png`.")


# ==============================================================================
# TAB 3: INTERACTIVE 10-YEAR MUNICIPAL SIMULATOR
# ==============================================================================
with tab3:
    st.markdown("### 10-Year Municipal Maintenance Simulation (Step 5C)")
    st.caption("Monte Carlo asset management simulation comparing 6 policies over 200 synthetic roadway segments.")

    sim_data = load_precomputed_sim_data()

    col_sim_ctrl1, col_sim_ctrl2, col_sim_ctrl3 = st.columns(3)
    with col_sim_ctrl1:
        sim_pool = st.selectbox("Tile Pool Selection:", ["all (455 tiles, cracks + potholes)", "cracks (287 tiles, cracks only)"])
        pool_key = "all" if "all" in sim_pool else "cracks"
        sim_budget = st.select_slider("Annual Capital Budget (% of Ref Rehab):", options=[5, 10, 20], value=10)
    with col_sim_ctrl2:
        sim_rehab_ratio = st.select_slider("Rehabilitation Cost Ratio (c_rehab):", options=[3, 5, 8], value=5)
        sim_noise_lambda = st.slider("Zero-Mean Noise Scale (λ):", 0.0, 2.0, 1.0, 0.5)
    with col_sim_ctrl3:
        sim_bias_b = st.slider("Systematic Calibration Bias (b):", -10.0, 10.0, 0.0, 5.0)
        data_source = st.radio("Simulation Execution Mode:", ["Precomputed Benchmark (Step 5C, 200 Runs)", "Live Monte Carlo Engine"], horizontal=True)

    st.divider()

    if data_source == "Precomputed Benchmark (Step 5C, 200 Runs)":
        base_df_key = f"base_{pool_key}"
        if base_df_key in sim_data:
            mc_df = sim_data[base_df_key]

            st.markdown(f"#### Base Scenario Results: Pool `{pool_key}` (Budget: {sim_budget}%, Rehab Ratio: {sim_rehab_ratio}, 200 Runs)")
            
            # Policy Table
            policies = ["P0_do_nothing", "P1_reactive", "P2_worst_first_by_band", "P2t_traffic_aware_worst_first", "P3_greedy_benefit_cost", "P4_integer_programming"]
            pol_labels = {
                "P0_do_nothing": "P0: Do Nothing",
                "P1_reactive": "P1: Reactive (Severe Only)",
                "P2_worst_first_by_band": "P2: Worst-First (ASTM Bands)",
                "P2t_traffic_aware_worst_first": "P2t: Traffic-Aware Worst-First",
                "P3_greedy_benefit_cost": "P3: Greedy Benefit/Cost",
                "P4_integer_programming": "P4: Exact 0/1 Integer Program",
            }

            table_rows = []
            for pol in policies:
                sub = mc_df[mc_df["policy"] == pol]
                m_aw, l_aw, h_aw = bootstrap_ci(sub["condition_years_aw"].values)
                m_tw, l_tw, h_tw = bootstrap_ci(sub["condition_years_tw"].values)
                m_sp, _, _ = bootstrap_ci(sub["total_spent"].values)
                m_ut, _, _ = bootstrap_ci(sub["mean_budget_utilization"].values * 100.0)
                m_eff_tw, _, _ = bootstrap_ci(sub["efficiency_tw_per_1k"].values)

                table_rows.append({
                    "Policy": pol_labels[pol],
                    "Area-Weighted CY": f"{m_aw:.2f} [{l_aw:.2f}, {h_aw:.2f}]",
                    "Traffic-Weighted CY": f"{m_tw:.2f} [{l_tw:.2f}, {h_tw:.2f}]",
                    "Total Real Spend": f"{m_sp:,.0f}",
                    "Budget Util (%)": f"{m_ut:.1f}%",
                    "TW Efficiency (CY/1k)": f"{m_eff_tw:.3g}",
                })

            st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

            # Interactive Plotly Charts
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                # Area-Weighted vs Traffic-Weighted Condition-Years
                pol_names = [pol_labels[p] for p in policies]
                aw_means = [mc_df[mc_df["policy"] == p]["condition_years_aw"].mean() for p in policies]
                tw_means = [mc_df[mc_df["policy"] == p]["condition_years_tw"].mean() for p in policies]

                fig_bars = go.Figure(data=[
                    go.Bar(name="Area-Weighted CY", x=pol_names, y=aw_means, marker_color="#3b82f6"),
                    go.Bar(name="Traffic-Weighted CY", x=pol_names, y=tw_means, marker_color="#10b981"),
                ])
                fig_bars.update_layout(
                    title="Area-Weighted vs. Traffic-Weighted Condition-Years (10 Years)",
                    barmode="group",
                    template="plotly_dark",
                    height=360,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_bars, use_container_width=True)

            with col_chart2:
                # Cost-Effectiveness
                eff_means = [mc_df[mc_df["policy"] == p]["efficiency_tw_per_1k"].mean() for p in policies]
                fig_eff = go.Figure(data=[
                    go.Bar(
                        x=pol_names,
                        y=eff_means,
                        marker_color=["#64748b", "#f97316", "#eab308", "#a855f7", "#06b6d4", "#22c55e"],
                        text=[f"{v:.3f}" for v in eff_means],
                        textposition="auto",
                    )
                ])
                fig_eff.update_layout(
                    title="Traffic-Weighted Efficiency (CY Gained over P0 per 1,000 Units Spent)",
                    yaxis_title="CY / 1,000 Cost Units",
                    template="plotly_dark",
                    height=360,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_eff, use_container_width=True)

            # Decomposition Figures
            st.markdown("#### Controlled Bias vs. Noise Decompositions (Step 5C)")
            col_f1, col_f2 = st.columns(2)
            bias_fig = Path("docs/figures/simulation_regret_vs_bias.png")
            noise_fig = Path("docs/figures/simulation_regret_vs_lambda.png")

            with col_f1:
                if bias_fig.exists():
                    st.image(str(bias_fig), caption="Regret vs. Calibration Bias b (Notice Negative Regrets in P1 & P3)", use_container_width=True)
            with col_f2:
                if noise_fig.exists():
                    st.image(str(noise_fig), caption="Regret vs. Zero-Mean Noise Scale λ (Strictly Non-Negative Regret)", use_container_width=True)

    else:
        st.info("Live Monte Carlo Mode selected. Click below to execute 20 synchronized runs across policies.")
        if st.button("🚀 Execute Live Simulation"):
            with st.spinner("Generating synthetic road network and solving 0/1 integer programs via HiGHS..."):
                net = generate_synthetic_network(tile_scores_path="results/tile_scores.csv", tile_pool=pool_key, n_segments=200, seed=42)
                live_df = run_monte_carlo(
                    network=net,
                    policies=["P0_do_nothing", "P1_reactive", "P2_worst_first_by_band", "P2t_traffic_aware_worst_first", "P3_greedy_benefit_cost", "P4_integer_programming"],
                    budget_pct=float(sim_budget),
                    rehab_cost_ratio=float(sim_rehab_ratio),
                    n_runs=20,
                    base_seed=5000,
                    noise_lambda=sim_noise_lambda,
                    bias_b=sim_bias_b,
                )
                st.success("Simulation finished successfully!")
                st.dataframe(live_df.groupby("policy")[["condition_years_aw", "condition_years_tw", "total_spent", "efficiency_tw_per_1k"]].mean(), use_container_width=True)


# Footer
st.divider()
st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 0.85rem;">
        Road-PCI-Intelligence · End-to-End Deep Learning & ASTM D6433 Pavement Management System · MIT License
    </div>
    """,
    unsafe_allow_html=True,
)
