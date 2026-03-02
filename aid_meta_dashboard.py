

LOGO_PATH = "AID.png"

import streamlit as st
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

password = st.sidebar.text_input("Enter password", type="password")
if st.sidebar.button("Login"):
    if password == "AIDIntel":  # Change this!
        st.session_state.authenticated = True
    else:
        st.error("Wrong password")

if not st.session_state.authenticated:
    st.stop()
    

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import os
from pathlib import Path

print("Dashboard script started successfully")


# ──────────────────────────────────────────────────────────────
# Page config + dark theme
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AID • Agent Identity Desk",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# Dark mode + branding styling
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #e6e6e6;
    }
    .stTabs [data-baseweb="tab-list"] button {
        color: #e6e6e6 !important;
        background-color: #1a1f2e !important;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #2a3b5f !important;
        color: white !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #0e1117 !important;
    }
    .stMetricLabel {
        color: #a0a0ff !important;
    }
    .stMetricValue {
        color: white !important;
    }
    h1, h2, h3 {
        color: #d4d4ff !important;
    }
    </style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Load logo (your file path)
# ──────────────────────────────────────────────────────────────


if os.path.exists(LOGO_PATH):
    logo_img = open(LOGO_PATH, "rb").read()
    st.sidebar.image(logo_img, width=220)
else:
    st.sidebar.warning("Logo file not found at specified path")

st.sidebar.title("Agent Identity Desk")
st.sidebar.caption("Intelligent Governance for AI Agents & Non-Human Identities")
st.sidebar.markdown("---")


# ====================== LOAD ALL META LAYER DATA ======================
@st.cache_data
def load_data():
    agents = pd.read_csv('agents_behavior_moat.csv')
    insights = pd.read_csv('agent_meta_insights.csv')
    signatures = pd.read_pickle('agent_signatures_256dim.pkl')
    
# Merge everything
    df = agents.merge(insights, on='agent_id', how='left')
    
    # 2D projection for visualization (cached)
    if 'tsne_x' not in df.columns:
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
        tsne_2d = tsne.fit_transform(signatures.values)
        df['tsne_x'] = tsne_2d[:, 0]
        df['tsne_y'] = tsne_2d[:, 1]
    
    return df, signatures

df, signatures = load_data()

# ──────────────────────────────────────────────────────────────
# Main title & branding
# ──────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 5])
with col_title:
    st.title("Agent Identity Desk")
    st.caption("Intelligent Visibility & Governance for AI Agents • February 2026")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "🔍 Agent Explorer",
    "🧬 Behavioral Map",
    "🚨 Audit Candidates",
    "🔬 Similarity Search",
    "🛡️ Audit Center"
])


print("Tabs are about to be defined")

# ──────────────────────────────────────────────────────────────
# TAB 1: Overview
# ──────────────────────────────────────────────────────────────
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Agents Monitored", len(df))
    with col2:
        high_risk = len(df[df['over_privilege_risk'] > 0.6])
        st.metric("High-Risk Agents", high_risk)
        if high_risk == 0:
            st.caption("All agents currently low-risk")
    with col3:
        st.metric("Agents Needing Attention", len(df[df['anomaly_score'] == -1]))
    with col4:
        st.metric("Average Impact Score", f"{df['blast_radius_score'].mean():.1f}")

    st.subheader("Risk Distribution by Agent Type")
    fig = px.treemap(
        df, path=['behavior_bias', 'cluster_kmeans'],
        values='blast_radius_score',
        color='over_privilege_risk',
        color_continuous_scale='RdYlGn_r',
        title="Where the highest risks are concentrated"
    )
    st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────
# TAB 2: Agent Explorer (same as before, minor wording tweak)
# ──────────────────────────────────────────────────────────────
with tab2:
    search = st.text_input("Find agent by ID or name", placeholder="AID-00123 or Financial...")
    
    filtered = df.copy()
    if search:
        filtered = filtered[filtered['agent_id'].str.contains(search, case=False) | 
                           filtered['agent_name'].str.contains(search, case=False)]
    
    st.dataframe(
        filtered[['agent_id', 'agent_name', 'behavior_bias', 'over_privilege_risk',
                  'blast_radius_score', 'anomaly_score']],
        column_config={
            "over_privilege_risk": st.column_config.ProgressColumn(
                "Over Privilege Risk",
                min_value=0.0,
                max_value=1.0,
            ),
            "behavior_bias": st.column_config.ProgressColumn(
                "Behavior Bias",
                min_value=0.0,
                max_value=1.0,
            ),
            "blast_radius_score": st.column_config.ProgressColumn(
                "Blast Radius",
                min_value=0.0,
                max_value=1.0,
            ),
            "anomaly_score": st.column_config.ProgressColumn(
                "Anomaly Score",
                min_value=0.0,
                max_value=1.0,
            ),
        },
        height=600,
    )

    if not filtered.empty:
        selected = st.selectbox("Deep dive into agent", filtered['agent_id'])
        row = df[df['agent_id'] == selected].iloc[0]
        
        colA, colB = st.columns(2)
        with colA:
            st.metric("Risk Level", f"{row['over_privilege_risk']:.3f}")
            st.metric("Business Impact", f"{row['blast_radius_score']:.1f}")
        with colB:
            st.metric("Behavioral Stability", f"{row['behavior_deviation']:.3f}")
            st.metric("Group", row['cluster_kmeans'])

# ──────────────────────────────────────────────────────────────
# TAB 3: Behavioral Map (was Clusters – renamed & reworded)
# ──────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Agent Behavioral Similarity Map")
    st.caption("Agents with similar operating patterns appear close together")

    fig = px.scatter(
        df, x='tsne_x', y='tsne_y',
        color='behavior_bias',
        hover_name='agent_id',
        hover_data=['agent_name', 'over_privilege_risk', 'blast_radius_score'],
        size='blast_radius_score',
        size_max=18,
        opacity=0.85,
        title="Visual overview of behavioral patterns across your agent fleet"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Pattern Group Summary")
    summary = df.groupby('cluster_kmeans').agg(
        count=('agent_id', 'count'),
        avg_risk=('over_privilege_risk', 'mean'),
        avg_impact=('blast_radius_score', 'mean')
    ).round(2)
    st.dataframe(summary)

# ──────────────────────────────────────────────────────────────
# TAB 4: Audit Candidates (same, minor reword)
# ──────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Priority Agents for Review")
    audit_df = df[df['anomaly_score'] == -1].sort_values('blast_radius_score', ascending=False)
    st.dataframe(
        audit_df[['agent_id', 'agent_name', 'behavior_bias', 'blast_radius_score', 
                  'behavior_deviation', 'over_privilege_risk', 'predicted_rollback_minutes']].head(20),
        use_container_width=True,
        height=700
    )

# ──────────────────────────────────────────────────────────────
# TAB 5: Similarity Search (rewritten business language)
# ──────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Find Agents That Behave Similarly")
    st.caption("Discover agents with comparable patterns to help with policy consistency and risk benchmarking")

    query_agent = st.selectbox("Select reference agent", df['agent_id'].unique())
    
    if query_agent:
        idx = signatures.index.get_loc(query_agent)
        query_sig = signatures.iloc[idx].values.reshape(1, -1)
        
        sims = cosine_similarity(query_sig, signatures.values)[0]
        top_idx = np.argsort(sims)[-6:-1][::-1]
        
        similar = pd.DataFrame({
            'agent_id': signatures.index[top_idx],
            'pattern_similarity': sims[top_idx].round(3),
            'type': df.iloc[top_idx]['behavior_bias'].values,
            'risk': df.iloc[top_idx]['over_privilege_risk'].values.round(3)
        })
        
        st.dataframe(
            similar.style.background_gradient(subset=['pattern_similarity'], cmap='Blues'),
            use_container_width=True
        )

# ──────────────────────────────────────────────────────────────
# TAB 6: AUDIT CENTER – FULL INTERACTIVE VERSION
# ──────────────────────────────────────────────────────────────
with tab6:
    st.header("🛡️ Audit Center")
    st.markdown("Run targeted reviews using AID's intelligent insights. Choose a scenario below.")

    if df.empty:
        st.error("No data loaded. Check file paths and refresh.")
    else:
        audit_options = [
            "Behavioral Drift Detection",
            "High Business Impact Audit",
            "Over-Privilege Risk Review",
            "Compliance Exposure Overview",
            "Potential Forgotten Agents"
        ]

        selected = st.selectbox("Select Audit Scenario", audit_options)

        # ──────────────────────────────
        # 1. Behavioral Drift
        # ──────────────────────────────
        if selected == "Behavioral Drift Detection":
            st.subheader("Agents Showing Unusual Behavioral Change")
            thresh = st.slider("Change Sensitivity", 0.10, 0.60, 0.25, 0.01)
            res = df[df['behavior_deviation'] > thresh].sort_values('behavior_deviation', ascending=False)
            
            if res.empty:
                st.info("No agents show significant change at this sensitivity level. Try lowering the threshold.")
            else:
                st.metric("Agents with Change", len(res))
                st.dataframe(
                    res[['agent_id', 'agent_name', 'behavior_bias', 'behavior_deviation', 
                         'blast_radius_score', 'over_privilege_risk']].head(30).style.background_gradient(
                        subset=['behavior_deviation'], cmap='OrRd'
                    ),
                    use_container_width=True,
                    height=500
                )
                csv = res.to_csv(index=False).encode('utf-8')
                st.download_button("Export Results", csv, "drift_audit.csv", "text/csv")

        # ──────────────────────────────
        # 2. High Business Impact
        # ──────────────────────────────
        elif selected == "High Business Impact Audit":
            st.subheader("Agents with Largest Potential Business Impact")
            thresh = st.slider("Impact Threshold", 10.0, 300.0, 30.0, 5.0)
            res = df[df['blast_radius_score'] > thresh].sort_values('blast_radius_score', ascending=False)
            
            st.metric("High-Impact Agents", len(res))
            st.dataframe(
                res[['agent_id', 'agent_name', 'blast_radius_score',  
                     'rollback_avg', 'predicted_rollback_minutes']].head(30).style.background_gradient(
                    subset=['blast_radius_score'], cmap='YlOrRd'
                ),
                use_container_width=True,
                height=500
            )
            csv = res.to_csv(index=False).encode('utf-8')
            st.download_button("Export Results", csv, "impact_audit.csv", "text/csv")

        # ──────────────────────────────
        # 3. Over-Privilege Risk
        # ──────────────────────────────
        elif selected == "Over-Privilege Risk Review":
            st.subheader("Agents with Elevated Privilege Risk")
            thresh = st.slider("Risk Level", 0.30, 0.90, 0.45, 0.05)
            res = df[df['over_privilege_risk'] > thresh].sort_values('over_privilege_risk', ascending=False)
            
            st.metric("Elevated Risk Agents", len(res))
            st.dataframe(
                res[['agent_id', 'agent_name', 'over_privilege_risk', 'high_privilege_ratio', 
                     'anomaly_rate']].head(30).style.background_gradient(
                    subset=['over_privilege_risk'], cmap='Reds'
                ),
                use_container_width=True,
                height=500
            )
            csv = res.to_csv(index=False).encode('utf-8')
            st.download_button("Export Results", csv, "privilege_audit.csv", "text/csv")

        # ──────────────────────────────
        # 4. Compliance Exposure
        # ──────────────────────────────
        elif selected == "Compliance Exposure Overview":
            st.subheader("Compliance Risk by Agent Type")
            pivot = df.pivot_table(
                index='behavior_bias', columns='cluster_kmeans', 
                values='compliance_exposure', aggfunc='mean'
            ).round(2).fillna(0)
            
            fig = px.imshow(pivot, color_continuous_scale='YlOrRd', 
                            title="Compliance Exposure Heatmap")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(pivot)
            csv = pivot.to_csv().encode('utf-8')
            st.download_button("Export Heatmap", csv, "compliance_overview.csv", "text/csv")

        # ──────────────────────────────
        # 5. Forgotten Agents
        # ──────────────────────────────
        elif selected == "Potential Forgotten Agents":
            st.subheader("Agents with Lingering Risk (Potential Forgotten)")
            thresh = st.slider("Minimum Residual Impact", 10.0, 150.0, 25.0, 5.0)
            res = df[(df['blast_radius_score'] > thresh) & 
                     (df['anomaly_score'] == -1)].sort_values('blast_radius_score', ascending=False)
            
            st.metric("Potential Forgotten Agents", len(res))
            st.dataframe(
                res[['agent_id', 'agent_name', 'blast_radius_score', 
                     'predicted_rollback_minutes', 'behavior_deviation']].head(20).style.background_gradient(
                    subset=['blast_radius_score'], cmap='Oranges'
                ),
                use_container_width=True,
                height=500
            )
            csv = res.to_csv(index=False).encode('utf-8')
            st.download_button("Export Results", csv, "forgotten_audit.csv", "text/csv")

# ──────────────────────────────────────────────────────────────
# Sidebar final controls
# ──────────────────────────────────────────────────────────────
st.sidebar.success("Connected • Live Intelligence Layer")
st.sidebar.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

if st.sidebar.button("Refresh Dashboard"):
    st.cache_data.clear()
    st.experimental_rerun()  # Use experimental_rerun if not upgraded yet
