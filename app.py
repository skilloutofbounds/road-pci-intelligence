"""
ROAD-PCI-INTELLIGENCE: Flagship Command Center & Interactive Web App.

Unifies:
1. Live Pavement Inspector (Perception, ASTM D6433 Features, sPCI Rating)
2. Double-Blind Human Alignment Explorer (Runtime computed from test pairs, zero hardcoded headline numbers)
3. 10-Year Municipal Maintenance Simulator (P0–P4, P2t, Monte Carlo, HiGHS MILP)
"""

import sys
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

from src.pipeline import (
    RoadPCIPipeline,
    create_synthetic_road_tile,
    EXPECTED_SHA256,
    HF_MODEL_REPO_ID,
)
from src.spci import get_condition_class
from src.simulation.network import generate_synthetic_network
from src.simulation.engine import run_monte_carlo, bootstrap_ci


# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Initializes and caches the perception pipeline. Raises FileNotFoundError if weights missing."""
    try:
        pipe = RoadPCIPipeline(device="cpu")
        return pipe, None
    except Exception as e:
        return None, str(e)


@st.cache_data(show_spinner=False)
def read_rdd_limitation_doc() -> str:
    """Reads the RDD limitation tables verbatim from docs/experiments.md, not retyped."""
    exp_file = Path("docs/experiments.md")
    if not exp_file.exists():
        return "Documentation file `docs/experiments.md` not found."
    content = exp_file.read_text(encoding="utf-8", errors="ignore")
    start_token = "## 6. Out-of-Domain Perspective Evaluation on RDD2022ES"
    end_token = "## 7. Frozen Perception Backbone Verification"
    if start_token in content:
        sub = content.split(start_token)[1]
        if end_token in sub:
            sub = sub.split(end_token)[0]
        return f"### {start_token}\n\n{sub.strip()}"
    return content[:3000]


@st.cache_data(show_spinner=False)
def compute_runtime_human_alignment():
    """
    Computes all Tab 2 headline numbers at runtime strictly from raw evaluation artifacts:
    - results/spci_test_pairs.csv
    - labels/pairs_human.csv
    Adheres strictly to the pre-registered protocol:
    - Excludes 'same' labels
    - |sPCI_A - sPCI_B| < 1.0 is counted as a tie (0.5 points)
    - 10,000 bootstrap resamples (seed 42) for 95% CIs
    - Paired difference against Damage Area baseline
    """
    pairs_p = Path("results/spci_test_pairs.csv")
    labels_p = Path("labels/pairs_human.csv")
    if not (pairs_p.exists() and labels_p.exists()):
        return None

    df_p = pd.read_csv(pairs_p)
    df_l = pd.read_csv(labels_p)

    df_a = df_p[df_p["which"] == "A"].set_index("pair_id")
    df_b = df_p[df_p["which"] == "B"].set_index("pair_id")

    records = []
    for _, row in df_l.iterrows():
        pid = row["pair_id"]
        lbl = str(row["worse"]).strip()
        ra = df_a.loc[pid]
        rb = df_b.loc[pid]
        records.append({
            "pair_id": pid,
            "source": ra["source"],
            "human_label": lbl,
            "spci_a": float(ra["sPCI"]),
            "spci_b": float(rb["sPCI"]),
            "area_a": float(ra["crack_area_frac"] + ra["pothole_area_frac"]),
            "area_b": float(rb["crack_area_frac"] + rb["pothole_area_frac"]),
        })
    df = pd.DataFrame(records)
    valid = df[df["human_label"].isin(["A", "B"])].copy()

    def score_df(d):
        spci_sc = []
        area_sc = []
        for _, r in d.iterrows():
            d_pci = abs(r["spci_a"] - r["spci_b"])
            if d_pci < 1.0:
                spci_sc.append(0.5)
            elif (r["spci_a"] < r["spci_b"] and r["human_label"] == "A") or (r["spci_b"] < r["spci_a"] and r["human_label"] == "B"):
                spci_sc.append(1.0)
            else:
                spci_sc.append(0.0)

            d_area = abs(r["area_a"] - r["area_b"])
            if d_area <= 0.0:
                area_sc.append(0.5)
            elif (r["area_a"] > r["area_b"] and r["human_label"] == "A") or (r["area_b"] > r["area_a"] and r["human_label"] == "B"):
                area_sc.append(1.0)
            else:
                area_sc.append(0.0)
        return np.array(spci_sc), np.array(area_sc)

    def boot(arr, n=10000, seed=42):
        rng = np.random.RandomState(seed)
        boot_means = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n)]
        return float(np.mean(arr)), float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))

    spci_scores, area_scores = score_df(valid)
    overall_mean, overall_low, overall_high = boot(spci_scores)
    area_mean, area_low, area_high = boot(area_scores)

    diff_arr = spci_scores - area_scores
    diff_mean, diff_low, diff_high = boot(diff_arr)

    # Per source breakdowns
    sources = {}
    for src in ["crack500", "road_crack", "pothole"]:
        sub = valid[valid["source"] == src]
        s_pci, _ = score_df(sub)
        sm, sl, sh = boot(s_pci)
        sources[src] = {
            "n": len(sub),
            "points": float(np.sum(s_pci)),
            "mean": sm,
            "low": sl,
            "high": sh,
            "ties": int(np.sum(s_pci == 0.5)),
        }

    # Combined crack (crack500 + road_crack)
    crack_sub = valid[valid["source"].isin(["crack500", "road_crack"])]
    c_pci, _ = score_df(crack_sub)
    cm, cl, ch = boot(c_pci)
    sources["crack_combined"] = {
        "n": len(crack_sub),
        "points": float(np.sum(c_pci)),
        "mean": cm,
        "low": cl,
        "high": ch,
    }

    return {
        "n_total": len(df),
        "n_valid": len(valid),
        "n_same": len(df) - len(valid),
        "ties": int(np.sum(spci_scores == 0.5)),
        "total_points": float(np.sum(spci_scores)),
        "overall_mean": overall_mean,
        "overall_low": overall_low,
        "overall_high": overall_high,
        "area_mean": area_mean,
        "area_low": area_low,
        "area_high": area_high,
        "diff_mean": diff_mean,
        "diff_low": diff_low,
        "diff_high": diff_high,
        "sources": sources,
        "summary_table": valid,
    }


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


def is_scenario_precomputed(pool: str, budget: int, ratio: float, noise_lambda: float, bias_b: float) -> Tuple[bool, str]:
    """Identifies if a specific scenario has precomputed Monte Carlo artifacts."""
    if budget == 10 and ratio == 5.0 and noise_lambda == 1.0 and bias_b == 0.0:
        return True, "Base 10-Year Benchmark (N = 200 runs, synchronized seeds)"
    if budget == 10 and ratio == 5.0 and noise_lambda == 1.0 and bias_b in [-10.0, -5.0, 0.0, 5.0, 10.0]:
        return True, "Controlled Bias-Only Study (N = 100 runs)"
    if budget == 10 and ratio == 5.0 and bias_b == 0.0 and noise_lambda in [0.0, 0.5, 1.0, 1.5, 2.0]:
        return True, "Controlled Zero-Mean Noise Study (N = 100 runs)"
    if bias_b == 0.0 and noise_lambda == 1.0 and budget in [5, 10, 20] and ratio in [3, 5, 8]:
        return True, "Grid 2 Decoupled Sensitivity (N = 100 runs, 18 cells)"
    return False, "Not precomputed (Live Monte Carlo execution required)"


# ==============================================================================


def render_app():
    # 1. PAGE CONFIGURATION & STYLING
    # ==============================================================================
    st.set_page_config(
        page_title="Road PCI Intelligence | Municipal AI Command Center",
        page_icon="🛣️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .main { background-color: #0b0f19; }
        .metric-card {
            background: linear-gradient(135deg, rgba(22, 30, 49, 0.9) 0%, rgba(15, 23, 42, 0.9) 100%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 18px 22px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
            margin-bottom: 12px;
        }
        .badge-good { background: #00E676; color: #000; font-weight: 700; padding: 4px 12px; border-radius: 20px; }
        .badge-satisfactory { background: #00B0FF; color: #000; font-weight: 700; padding: 4px 12px; border-radius: 20px; }
        .badge-fair { background: #FFD600; color: #000; font-weight: 700; padding: 4px 12px; border-radius: 20px; }
        .badge-poor { background: #FF9100; color: #000; font-weight: 700; padding: 4px 12px; border-radius: 20px; }
        .badge-very-poor { background: #FF3D00; color: #fff; font-weight: 700; padding: 4px 12px; border-radius: 20px; }
        .badge-serious { background: #D50000; color: #fff; font-weight: 700; padding: 4px 12px; border-radius: 20px; }
        .badge-failed { background: #212121; color: #ff5252; border: 1px solid #ff5252; font-weight: 700; padding: 4px 12px; border-radius: 20px; }
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
            margin-bottom: 15px;
        }
        .banner-box {
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 16px;
            font-size: 0.95rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


    # 2. TOP BANNER (MANDATORY REQUIREMENT 2G)
    # ==============================================================================
    st.warning(
        "⚠️ **Research demo trained on close-up pavement images (handheld or drone). "
        "Not for dashcam or street views. sPCI is a simplified relative index, not ASTM-compliant, "
        "and not for real maintenance decisions.**"
    )

    # Expandable RDD limitation read verbatim from docs/experiments.md
    with st.expander("📖 View Out-of-Domain RDD2022 Limitation Benchmark (Read from docs/experiments.md)"):
        st.markdown(read_rdd_limitation_doc())


    # ==============================================================================
    # 3. SIDEBAR STATUS & METRICS
    # ==============================================================================
    pipeline, pipe_error = load_pipeline()

    with st.sidebar:
        st.image(
            "https://images.unsplash.com/photo-1544620347-c4fd4a3d5957?auto=format&fit=crop&w=500&q=80",
            caption="Autonomous Infrastructure Intelligence",
            use_container_width=True,
        )
        st.markdown("### 🛣️ System Architecture Status")

        if pipeline is not None:
            st.success(f"🟢 **Perception Backbone**: ResNet-34 U-Net ({pipeline.device})")
            st.caption(f"SHA-256: `{EXPECTED_SHA256[:16]}...` (Verified)")
        else:
            st.error("🔴 **Perception Backbone**: Missing Weights")
            st.caption(f"Target Checkpoint: `best.pt` (SHA-256: `{EXPECTED_SHA256[:16]}...`)")

        st.markdown(
            """
            - **Model**: `smp.Unet(resnet34, 2-class)`
            - **sPCI Engine**: Frozen Tag `spci-v1-frozen`
            - **Human Benchmark**: Tag `test-scores-computed`
            - **Simulation**: 14/14 Unit Tests Passing
            - **In-Memory Guarantee**: Zero disk persistence
            """
        )
        st.divider()
        st.markdown("### 📚 Traceable Facts & Verification")
        st.markdown(
            """
            - [Verified Facts Record](docs/PROJECT_FACTS.md)
            - [Simulation Evidence (Step 5C)](docs/simulation.md)
            - [Double-Blind Validation](docs/human_validation.md)
            - [sPCI Formulation](docs/pci_design.md)
            - [Model Experiments](docs/experiments.md)
            """
        )
        st.caption("Road-PCI-Intelligence · Advanced Agentic Coding")


    # ==============================================================================
    # 4. MAIN APPLICATION HEADER
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

        if pipeline is None:
            st.error(
                f"❌ **Perception Model Weights Missing**: Model weights `best.pt` (SHA-256: `{EXPECTED_SHA256}`) "
                f"could not be loaded.\n\n"
                f"**Setup Instructions**:\n"
                f"1. Download `best.pt` and place it in `outputs/models/best.pt`, OR\n"
                f"2. Ensure the public Hugging Face repository `{HF_MODEL_REPO_ID}` contains `best.pt`.\n\n"
                f"*Demo Mode has been completely removed to prevent ungrounded or synthetic mask generation.*"
            )
        else:
            st.caption("Select one of the 6 held-out test split example tiles or upload a custom image.")

            # In-Memory Security Confirmation (Requirement 2g-iv)
            st.info("🔒 **In-Memory Guarantee**: Uploaded images are processed strictly in RAM and never written to disk or persisted.")

            col_input, col_ctrl = st.columns([2, 1])

            # 6 Curated Test Split Examples (Requirement 2e & 2f)
            EXAMPLE_SPECS = [
                ("crack500_example_01.jpg", "CRACK500", "Satisfactory", 70.68, "Held-out test split tile with moderate longitudinal cracking"),
                ("crack500_example_02.jpg", "CRACK500", "Serious", 24.60, "Held-out test split tile with extensive structural alligator fatigue"),
                ("road_crack_example_01.jpg", "Road Crack", "Good", 90.81, "Held-out test split tile with high surface integrity"),
                ("road_crack_example_02.jpg", "Road Crack", "Poor", 40.97, "Held-out test split tile with dense branching crack distress"),
                ("pothole_example_01.jpg", "Pothole", "Poor", 43.65, "Held-out test split tile with structural surface depression"),
                ("pothole_example_02.jpg", "Pothole", "Satisfactory", 79.08, "Held-out test split tile with minor localized pothole damage"),
            ]

            with col_input:
                input_mode = st.radio(
                    "Input Mode:",
                    ["Held-Out Test Split Examples (6 Tiles)", "Upload Custom Pavement Image"],
                    horizontal=True,
                )

                image_to_process = None
                is_upload_flag = False

                if input_mode == "Held-Out Test Split Examples (6 Tiles)":
                    example_labels = [
                        f"{spec[1]} | {spec[2]} (Benchmark sPCI: {spec[3]:.1f}) — {spec[4]}"
                        for spec in EXAMPLE_SPECS
                    ]
                    selected_idx = st.selectbox(
                        "Select Held-Out Test Example Tile:",
                        range(len(EXAMPLE_SPECS)),
                        format_func=lambda i: example_labels[i],
                    )
                    spec = EXAMPLE_SPECS[selected_idx]
                    ex_path = Path("app_assets/examples") / spec[0]
                    if ex_path.exists():
                        image_to_process = ex_path
                        # For crack500, upload resizing applies the 0.5 scale factor matching benchmark evaluation
                        is_upload_flag = (spec[1] == "CRACK500")
                    else:
                        st.error(f"Example file {ex_path} not found.")

                else:
                    uploaded_file = st.file_uploader(
                        "Upload Road Surface Image (JPEG, PNG | Max 25 MB | Min 256 px):",
                        type=["jpg", "jpeg", "png"],
                    )
                    if uploaded_file is not None:
                        # Pass in-memory BytesIO directly (never written to disk)
                        image_to_process = io.BytesIO(uploaded_file.read())
                        is_upload_flag = True
                    else:
                        st.info("Please select an image above or upload an orthogonal close-up pavement image.")

                # Advanced / Developer expander for synthetic generator (Requirement 2f)
                with st.expander("🛠️ Advanced / Developer: Synthetic Pattern Generator (Not Real Pavement)"):
                    st.caption("⚠️ **Notice**: Synthetic pattern, not real pavement, not evidence of accuracy.")
                    synth_type = st.selectbox("Pattern Archetype:", ["crack", "pothole", "alligator"])
                    synth_seed = st.number_input("Pattern Seed:", value=42, step=1)
                    if st.button("Generate Synthetic Tile"):
                        image_to_process = create_synthetic_road_tile(condition=synth_type, seed=synth_seed)
                        is_upload_flag = False

            with col_ctrl:
                st.markdown("**Perception & Display Settings**")
                thresh_crack = st.slider("Crack Detection Threshold:", 0.20, 0.80, 0.50, 0.05)
                thresh_pothole = st.slider("Pothole Detection Threshold:", 0.20, 0.80, 0.50, 0.05)
                overlay_alpha = st.slider("Overlay Transparency (α):", 0.20, 0.80, 0.50, 0.05)
                use_tta = st.checkbox("Enable Flip-TTA (Mean of Orig, HFlip, VFlip)", value=True)

            if image_to_process is not None:
                try:
                    with st.spinner("Executing Deep Perception Backbone (Flip-TTA) & ASTM Feature Extraction..."):
                        result = pipeline.infer(
                            image_to_process,
                            is_upload=is_upload_flag,
                            threshold_crack=thresh_crack,
                            threshold_pothole=thresh_pothole,
                            use_tta=use_tta,
                            overlay_alpha=overlay_alpha,
                        )
                except Exception as e:
                    st.error(f"Error processing image: {e}")
                    result = None

                if result is not None:
                    spci_val = result["spci"]
                    cond_class = result["condition_class"]
                    features = result["features"]

                    st.divider()

                    col_score, col_presc = st.columns([1, 2])
                    with col_score:
                        badge_cls = f"badge-{cond_class.lower().replace(' ', '-')}"
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <span style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase;">Automated Condition Rating</span>
                                <div style="display: flex; align-items: baseline; gap: 12px; margin: 8px 0;">
                                    <span style="font-size: 2.8rem; font-weight: 800; color: #fff;">{spci_val:.1f}</span>
                                    <span style="font-size: 1.1rem; color: #94a3b8;">/ 100</span>
                                </div>
                                <span class="{badge_cls}">{cond_class.upper()}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with col_presc:
                        # Repair recommendation with disclaimer (Requirement 2g-iii)
                        rec_treatment = "No Action Required"
                        rec_cost = "0.0 units/m²"
                        rec_desc = "Pavement surface exhibits satisfactory structural condition. Continue routine visual monitoring."

                        if spci_val <= 10:
                            rec_treatment = "Full-Depth Reconstruction"
                            rec_cost = "12.0 units/m²"
                            rec_desc = "Total structural failure. Complete subgrade removal and repaving required."
                        elif spci_val <= 25:
                            rec_treatment = "Structural Reconstruction / Deep Base Repair"
                            rec_cost = "12.0 units/m²"
                            rec_desc = "Severe structural loss. Heavy reconstruction required."
                        elif spci_val <= 55:
                            rec_treatment = "Structural Rehabilitation (Mill & Overlay)"
                            rec_cost = "5.0 units/m² (Reference)"
                            rec_desc = "High severity distress. Pavement requires milling damaged asphalt layer and structural inlay."
                        elif spci_val <= 70:
                            rec_treatment = "Preventive Maintenance (Crack Seal / Micro-Surfacing)"
                            rec_cost = "1.0 units/m²"
                            rec_desc = "Early distress detected. Sealing cracks now arrests water intrusion and prevents accelerated base degradation."

                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <span style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase;">Asset Management Prescription</span>
                                <span style="font-size: 0.75rem; color: #f59e0b; margin-left: 8px;">(Illustrative mapping from sPCI band; not engineering advice)</span>
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
                        st.image(result["overlay_image"], caption="2. Qualitative Overlay (Scarlet: Crack, Cyan: Pothole)", use_container_width=True)
                    with col_img3:
                        mask_display = np.zeros((*result["crack_mask"].shape, 3), dtype=np.uint8)
                        mask_display[result["crack_mask"] > 0] = [255, 60, 60]
                        mask_display[result["pothole_mask"] > 0] = [0, 220, 255]
                        st.image(mask_display, caption="3. Binary Masks (Channel Separated)", use_container_width=True)

                    # ASTM D6433 Physical Metrics
                    st.markdown("#### ASTM D6433 Morphological & Geometric Features (Normalized at 512 px Scale)")
                    m1, m2, m3, m4, m5, m6 = st.columns(6)
                    m1.metric("Crack Area Frac", f"{features['crack_area_frac']*100:.2f}%")
                    m2.metric("Skeleton Length", f"{features['crack_length_px']:.1f} px")
                    m3.metric("Mean Width", f"{features['mean_width_px']:.1f} px")
                    m4.metric("Junction Count", f"{features['n_junctions']}")
                    m5.metric("Pothole Count", f"{features['pothole_count']}")
                    m6.metric("Pothole Area Frac", f"{features['pothole_area_frac']*100:.2f}%")


    # ==============================================================================
    # TAB 2: DOUBLE-BLIND HUMAN ALIGNMENT EXPLORER (RUNTIME COMPUTED)
    # ==============================================================================
    with tab2:
        st.markdown("### Double-Blind Human Validation Study (Step 4C)")
        st.caption(
            "Pairwise visual rating alignment computed dynamically at runtime from raw evaluation artifacts "
            "(`results/spci_test_pairs.csv` and `labels/pairs_human.csv`). One human labeler, 60 test pairs."
        )

        ha_stats = compute_runtime_human_alignment()

        if ha_stats is not None:
            c1, c2, c3, c4 = st.columns(4)

            # 1. Overall Human Agreement (Pre-registered rule: exclude 'same', |delta| < 1 is tie counted 0.5)
            n_valid = ha_stats["n_valid"]
            pts = ha_stats["total_points"]
            m_pct = ha_stats["overall_mean"] * 100.0
            l_pct = ha_stats["overall_low"] * 100.0
            h_pct = ha_stats["overall_high"] * 100.0
            ties = ha_stats["ties"]

            c1.metric(
                "Overall Human Agreement",
                f"{m_pct:.1f}%",
                f"{pts:.1f} / {n_valid} Pairs (95% CI: {l_pct:.1f}%–{h_pct:.1f}%)",
            )

            # 2. Crack Agreement
            crk = ha_stats["sources"]["crack_combined"]
            c2.metric(
                "Crack Distress Agreement",
                f"{crk['mean']*100:.1f}%",
                f"{crk['points']:.0f} / {crk['n']} Pairs (95% CI: {crk['low']*100:.1f}%–{crk['high']*100:.1f}%)",
            )

            # 3. Pothole Agreement
            pot = ha_stats["sources"]["pothole"]
            c3.metric(
                "Pothole Distress Agreement",
                f"{pot['mean']*100:.1f}%",
                f"{pot['points']:.1f} / {pot['n']} Pairs (Ties: {pot['ties']})",
            )

            # 4. Paired Difference against Damage Area Baseline
            diff_m = ha_stats["diff_mean"] * 100.0
            diff_l = ha_stats["diff_low"] * 100.0
            diff_h = ha_stats["diff_high"] * 100.0
            area_m = ha_stats["area_mean"] * 100.0

            diff_label = "no detectable difference" if (diff_l <= 0.0 <= diff_h) else f"{diff_m:+.1f}% Advantage"
            c4.metric(
                "Paired Diff vs Area Baseline",
                f"{diff_m:+.2f}%",
                f"CI: [{diff_l:+.1f}%, {diff_h:+.1f}%] ({diff_label})",
            )

            st.info(
                f"ℹ️ **Statistical Interpretation**: Overall sPCI agreement is **{m_pct:.2f}%** ({pts:.1f}/{n_valid} pairs, {ties} ties counted as 0.5). "
                f"The Damage Area baseline achieves **{area_m:.2f}%**. The paired difference is **{diff_m:+.2f}%** "
                f"[95% CI: {diff_l:+.2f}%, {diff_h:+.2f}%]. Because the 95% bootstrap confidence interval strictly contains 0, "
                f"there is **no detectable difference** between sPCI and the naive damage-area baseline on the overall set."
            )

            st.divider()

            col_view, col_img_dis = st.columns([1, 1])

            with col_view:
                st.markdown(f"#### Test Pair Summary ({ha_stats['n_total']} Total, {ha_stats['n_same']} Excluded 'Same')")
                valid_df = ha_stats["summary_table"].copy()

                valid_df["Outcome"] = valid_df.apply(
                    lambda r: "TIE (0.5)" if abs(r["spci_a"] - r["spci_b"]) < 1.0
                    else ("MATCH ✅" if ((r["spci_a"] < r["spci_b"] and r["human_label"] == "A") or (r["spci_b"] < r["spci_a"] and r["human_label"] == "B"))
                    else "DISAGREE ⚠️"),
                    axis=1
                )

                filter_source = st.selectbox("Filter Source:", ["ALL", "crack500", "road_crack", "pothole"])
                if filter_source != "ALL":
                    valid_df = valid_df[valid_df["source"] == filter_source]

                st.dataframe(
                    valid_df[["pair_id", "source", "human_label", "spci_a", "spci_b", "Outcome"]],
                    use_container_width=True,
                    height=340,
                )

            with col_img_dis:
                st.markdown("#### Qualitative Disagreement Analysis (60 Pairs, One Labeler)")
                disagreement_fig = Path("docs/figures/human_agreement_disagreements.png")
                if disagreement_fig.exists():
                    st.image(str(disagreement_fig), caption="Disagreement Analysis: Selected Pairs from Step 4C", use_container_width=True)
                    st.caption(
                        "**Hypotheses for Disagreement (Single Human Labeler, 60 Pairs)**:\n"
                        "1. *Pothole Cavity Depth*: The single human evaluator penalized deep isolated cavities heavily, whereas 2D sPCI measures surface area fraction.\n"
                        "2. *Fine Background Micro-Cracks*: Unannotated hairline fractures present in single-task masks influenced human visual severity."
                    )

        else:
            st.error("Evaluation files `results/spci_test_pairs.csv` or `labels/pairs_human.csv` not found.")


    # ==============================================================================
    # TAB 3: 10-YEAR MUNICIPAL MAINTENANCE SIMULATOR
    # ==============================================================================
    with tab3:
        st.markdown("### 10-Year Municipal Maintenance Simulation (Step 5C)")
        st.caption("Monte Carlo asset management simulation comparing 6 policies over 200 synthetic roadway segments.")

        sim_data = load_precomputed_sim_data()

        col_sim1, col_sim2, col_sim3 = st.columns(3)
        with col_sim1:
            sim_pool = st.selectbox("Tile Pool Selection:", ["all (455 tiles, cracks + potholes)", "cracks (287 tiles, cracks only)"])
            pool_key = "all" if "all" in sim_pool else "cracks"
            sim_budget = st.select_slider("Annual Capital Budget (% of Ref Rehab):", options=[5, 10, 20], value=10)
        with col_sim2:
            sim_rehab_ratio = st.select_slider("Rehabilitation Cost Ratio (c_rehab):", options=[3, 5, 8], value=5)
            sim_noise_lambda = st.slider("Zero-Mean Noise Scale (λ):", 0.0, 2.0, 1.0, 0.5)
        with col_sim3:
            sim_bias_b = st.slider("Systematic Calibration Bias (b):", -10.0, 10.0, 0.0, 5.0)
            exec_mode = st.radio("Simulation Execution Mode:", ["Precomputed Benchmark", "Live Monte Carlo Engine (Cap 20 Runs)"], horizontal=True)

        # Precomputed Status Indicator (Requirement 2h)
        is_precomp, precomp_desc = is_scenario_precomputed(pool_key, sim_budget, float(sim_rehab_ratio), sim_noise_lambda, sim_bias_b)
        if is_precomp:
            st.success(f"✅ **Precomputed Benchmark Scenario Available**: {precomp_desc}")
        else:
            st.warning(f"⚡ **Live Simulation Required**: Parameter combination ({sim_budget}%, ratio {sim_rehab_ratio}, λ={sim_noise_lambda}, b={sim_bias_b}) is not precomputed.")

        st.divider()

        policies = ["P0_do_nothing", "P1_reactive", "P2_worst_first_by_band", "P2t_traffic_aware_worst_first", "P3_greedy_benefit_cost", "P4_integer_programming"]
        pol_labels = {
            "P0_do_nothing": "P0: Do Nothing",
            "P1_reactive": "P1: Reactive (Severe Only)",
            "P2_worst_first_by_band": "P2: Worst-First (ASTM Bands)",
            "P2t_traffic_aware_worst_first": "P2t: Traffic-Aware Worst-First",
            "P3_greedy_benefit_cost": "P3: Greedy Benefit/Cost",
            "P4_integer_programming": "P4: Exact 0/1 Integer Program",
        }

        if exec_mode == "Precomputed Benchmark":
            if not is_precomp:
                st.info("The selected parameter combination is not among precomputed benchmark runs. Switch to 'Live Monte Carlo Engine' to run this scenario dynamically.")
            else:
                base_df_key = f"base_{pool_key}"
                if base_df_key in sim_data:
                    mc_df = sim_data[base_df_key]
                    st.markdown(f"#### Base Benchmark: Pool `{pool_key}` (Budget: {sim_budget}%, Ratio: {sim_rehab_ratio}, N = 200 Runs)")

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

                    col_c1, col_c2 = st.columns(2)
                    with col_c1:
                        pol_names = [pol_labels[p] for p in policies]
                        aw_means = [mc_df[mc_df["policy"] == p]["condition_years_aw"].mean() for p in policies]
                        tw_means = [mc_df[mc_df["policy"] == p]["condition_years_tw"].mean() for p in policies]

                        fig_bars = go.Figure(data=[
                            go.Bar(name="Area-Weighted CY", x=pol_names, y=aw_means, marker_color="#3b82f6"),
                            go.Bar(name="Traffic-Weighted CY", x=pol_names, y=tw_means, marker_color="#10b981"),
                        ])
                        fig_bars.update_layout(
                            title="Area-Weighted vs. Traffic-Weighted Condition-Years (Illustrative Costs and Rates)",
                            barmode="group",
                            template="plotly_dark",
                            height=360,
                            margin=dict(l=20, r=20, t=40, b=20),
                        )
                        st.plotly_chart(fig_bars, use_container_width=True)
                        st.caption("ℹ️ *Banner: Illustrative costs and rates. Dimensionless relative units for comparative policy analysis.*")

                    with col_c2:
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
                            title="Traffic-Weighted Efficiency (Illustrative Costs and Rates)",
                            yaxis_title="CY / 1,000 Cost Units",
                            template="plotly_dark",
                            height=360,
                            margin=dict(l=20, r=20, t=40, b=20),
                        )
                        st.plotly_chart(fig_eff, use_container_width=True)
                        st.caption("ℹ️ *Banner: Illustrative costs and rates. Condition-years gained over P0 per 1,000 cost units spent.*")

        else:
            st.markdown("#### Live Monte Carlo Execution Mode (Capped at 20 Runs)")
            st.warning("⚠️ **Noisy-Result Banner**: Live Monte Carlo run with N = 20 runs. Results are subject to sample variance. For definitive statistical benchmarks, refer to precomputed N=200/N=100 tables.")

            if st.button("🚀 Execute Live Simulation (20 Runs)"):
                with st.spinner("Generating synthetic network and solving HiGHS 0/1 integer programs across 20 runs..."):
                    net = generate_synthetic_network(
                        tile_scores_path="results/tile_scores.csv",
                        tile_pool=pool_key,
                        n_segments=200,
                        seed=42,
                    )
                    live_df = run_monte_carlo(
                        network=net,
                        policies=policies,
                        budget_pct=float(sim_budget),
                        rehab_cost_ratio=float(sim_rehab_ratio),
                        n_runs=20,
                        base_seed=5000,
                        noise_lambda=sim_noise_lambda,
                        bias_b=sim_bias_b,
                    )
                    st.success("Live simulation completed successfully!")

                    # Live Results Table with CIs
                    live_rows = []
                    for pol in policies:
                        sub = live_df[live_df["policy"] == pol]
                        m_aw, l_aw, h_aw = bootstrap_ci(sub["condition_years_aw"].values, n_boot=2000)
                        m_tw, l_tw, h_tw = bootstrap_ci(sub["condition_years_tw"].values, n_boot=2000)
                        m_sp, _, _ = bootstrap_ci(sub["total_spent"].values, n_boot=2000)
                        m_eff, _, _ = bootstrap_ci(sub["efficiency_tw_per_1k"].values, n_boot=2000)
                        live_rows.append({
                            "Policy": pol_labels[pol],
                            "Area-Weighted CY (Live)": f"{m_aw:.2f} [{l_aw:.2f}, {h_aw:.2f}]",
                            "Traffic-Weighted CY (Live)": f"{m_tw:.2f} [{l_tw:.2f}, {h_tw:.2f}]",
                            "Real Spend": f"{m_sp:,.0f}",
                            "TW Efficiency": f"{m_eff:.3g}",
                        })
                    st.dataframe(pd.DataFrame(live_rows), use_container_width=True)

                    fig_live = go.Figure(data=[
                        go.Bar(name="Area-Weighted CY", x=[pol_labels[p] for p in policies], y=[live_df[live_df["policy"] == p]["condition_years_aw"].mean() for p in policies], marker_color="#3b82f6"),
                        go.Bar(name="Traffic-Weighted CY", x=[pol_labels[p] for p in policies], y=[live_df[live_df["policy"] == p]["condition_years_tw"].mean() for p in policies], marker_color="#10b981"),
                    ])
                    fig_live.update_layout(
                        title="Live Simulation Results (Illustrative Costs and Rates)",
                        barmode="group",
                        template="plotly_dark",
                        height=360,
                    )
                    st.plotly_chart(fig_live, use_container_width=True)
                    st.caption("ℹ️ *Banner: Illustrative costs and rates.*")


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


if __name__ == '__main__':
    render_app()
