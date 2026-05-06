import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from drought_utils import (
    load_complete_dataset,
    generate_demo_data,
    get_class_distribution,
    get_correlation_matrix,
    DROUGHT_LABELS,
    DROUGHT_COLORS
)

# Preferred: keep this helper in drought_utils.py.
# The fallback keeps the dashboard runnable until drought_utils.py is updated.
try:
    from drought_utils import add_relative_scaled_score
except ImportError:
    def add_relative_scaled_score(
        df,
        score_col="avg_score",
        scaled_col="scaled_score",
        min_value=0.0,
        max_value=3.0,
    ):
        """Min-max scale a score column to a relative range."""
        result = df.copy()
        score_min = result[score_col].min()
        score_max = result[score_col].max()

        if pd.isna(score_min) or pd.isna(score_max):
            result[scaled_col] = np.nan
        elif score_max == score_min:
            result[scaled_col] = (min_value + max_value) / 2
        else:
            result[scaled_col] = (
                (result[score_col] - score_min)
                / (score_max - score_min)
                * (max_value - min_value)
                + min_value
            )

        return result

import json
from urllib.request import urlopen

@st.cache_data
def load_state_geojson():
    url = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
    with urlopen(url) as response:
        states = json.load(response)
    return states

states_geojson = load_state_geojson()

fips_to_name = {
    "01":"Alabama","02":"Alaska","04":"Arizona","05":"Arkansas","06":"California",
    "08":"Colorado","09":"Connecticut","10":"Delaware","12":"Florida","13":"Georgia",
    "16":"Idaho","17":"Illinois","18":"Indiana","19":"Iowa","20":"Kansas",
    "21":"Kentucky","22":"Louisiana","23":"Maine","24":"Maryland","25":"Massachusetts",
    "26":"Michigan","27":"Minnesota","28":"Mississippi","29":"Missouri","30":"Montana",
    "31":"Nebraska","32":"Nevada","33":"New Hampshire","34":"New Jersey","35":"New Mexico",
    "36":"New York","37":"North Carolina","38":"North Dakota","39":"Ohio","40":"Oklahoma",
    "41":"Oregon","42":"Pennsylvania","44":"Rhode Island","45":"South Carolina","46":"South Dakota",
    "47":"Tennessee","48":"Texas","49":"Utah","50":"Vermont","51":"Virginia",
    "53":"Washington","54":"West Virginia","55":"Wisconsin","56":"Wyoming"
}

# Import FIPS coordinate mapping
try:
    from fips_coordinates import add_coordinates_to_dataframe
    HAS_FIPS_COORDS = True
except ImportError:
    HAS_FIPS_COORDS = False

# â”€â”€ Page config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.set_page_config(
    page_title="U.S. Drought Monitor",
    page_icon="ðŸŒµ",
    layout="wide",
    initial_sidebar_state="expanded",
)

# â”€â”€ Custom CSS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');
 
  html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
 
  /* Dark earthy theme */
  .stApp { background-color: #0f0e0b; color: #e8dcc8; }
 
  .block-container { padding: 2rem 2.5rem; }
 
  h1, h2, h3 { font-family: 'Syne', sans-serif; font-weight: 800; }
 
  /* Hero header */
  .hero {
    background: linear-gradient(135deg, #1a1608 0%, #2d2010 50%, #1a1608 100%);
    border: 1px solid #4a3820;
    border-radius: 4px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: '';
    position: absolute; inset: 0;
    background: repeating-linear-gradient(
      -45deg, transparent, transparent 20px,
      rgba(180,120,40,0.03) 20px, rgba(180,120,40,0.03) 21px
    );
  }
  .hero-title {
    font-size: 3rem; font-weight: 800; letter-spacing: -1px;
    color: #e8c87a; line-height: 1.1; margin: 0;
  }
  .hero-sub { color: #8a7a60; font-size: 1rem; margin-top: .5rem; font-family: 'Space Mono', monospace; }
 
  /* Metric cards */
  .metric-card {
    background: #1a1810; border: 1px solid #3a3020;
    border-radius: 4px; padding: 1.2rem 1.5rem;
    border-left: 3px solid #c8873a;
  }
  .metric-label { color: #8a7a60; font-size: .75rem; letter-spacing: 2px; text-transform: uppercase; font-family: 'Space Mono', monospace; }
  .metric-value { font-size: 2rem; font-weight: 800; color: #e8c87a; line-height: 1.2; }
 
  /* Section headers */
  .section-header {
    border-top: 1px solid #3a3020; padding-top: 1rem;
    margin-top: .5rem; margin-bottom: 1.5rem;
    color: #c8873a; font-family: 'Space Mono', monospace;
    font-size: .85rem; letter-spacing: 3px; text-transform: uppercase;
  }
 
  /* Sidebar */
  [data-testid="stSidebar"] {
    background-color: #130f08 !important;
    border-right: 1px solid #3a3020;
  }
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stMultiSelect label,
  [data-testid="stSidebar"] .stSlider label { color: #8a7a60 !important; font-size: .75rem; letter-spacing: 1px; }
 
  /* Plotly containers */
  .js-plotly-plot { border: 1px solid #2a2216; border-radius: 4px; }
 
  div[data-testid="stHorizontalBlock"] > div { gap: 1rem; }
</style>
""", unsafe_allow_html=True)


# â”€â”€ Data loading â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@st.cache_data
def load_data():
    """Load TEST data only."""
    try:
        data = load_complete_dataset(
            data_path="test.csv",
            soil_path="soil_data.csv",
            apply_outlier_treatment_flag=True,
            sample_size=None
        )
        return data

    except Exception as e:
        # Fall back to demo data
        demo = generate_demo_data(n_samples=5000)
        return demo


# Load test data
data = load_data()
test_df = data["df"]
soil_df = data.get("soil_df")
is_demo = data.get("is_demo", False)

# Add proper FIPS coordinates if not already present
if 'lat' not in test_df.columns or 'lon' not in test_df.columns:
    if HAS_FIPS_COORDS and 'fips' in test_df.columns:
        test_df = add_coordinates_to_dataframe(test_df)

PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='#130f08',
    font=dict(family='Syne', color='#e8dcc8'),
    xaxis=dict(gridcolor='#2a2216', linecolor='#3a3020'),
    yaxis=dict(gridcolor='#2a2216', linecolor='#3a3020'),
    margin=dict(l=40, r=20, t=50, b=40),
)

# â”€â”€ Sidebar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
with st.sidebar:
    st.markdown("### ðŸŒµ Controls")
    
    df_active = test_df

    st.markdown("---")

    all_scores = sorted(df_active['score'].unique())
    score_filter = st.multiselect(
        "Drought Levels",
        options=all_scores,
        default=all_scores,
        format_func=lambda x: DROUGHT_LABELS.get(x, str(x))
    )

    st.markdown("---")
    if 'year' in df_active.columns:
        years = sorted(df_active['year'].unique())
        year_range = st.select_slider("Year Range", options=years, value=(years[0], years[-1]))
    else:
        year_range = None
 
    st.markdown("---")
    if 'fips' in df_active.columns:
        top_fips = df_active['fips'].value_counts().head(30).index.tolist()
        region_compare = st.multiselect("Compare Regions (FIPS)", options=top_fips, default=top_fips[:3] if len(top_fips) >= 3 else top_fips)
    else:
        region_compare = []


# â”€â”€ Filter data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Use the sidebar selections as global filters. The state table and relative map
# are both based on this filtered dataframe.
df = df_active.copy()

if year_range and 'year' in df.columns:
    df = df[
        (df['year'] >= year_range[0]) &
        (df['year'] <= year_range[1])
    ]

if score_filter and 'score' in df.columns:
    df = df[df['score'].isin(score_filter)]

# â”€â”€ State-level aggregation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
state_scores = pd.DataFrame()

if 'fips' in df.columns:
    # Convert county FIPS â†’ state FIPS using the first 2 digits.
    df['state_fips'] = df['fips'].astype(str).str.zfill(5).str[:2]

    state_scores = df.groupby('state_fips').agg(
        avg_score=('score', 'mean'),
        count=('score', 'count')
    ).reset_index()

    state_scores['state_name'] = state_scores['state_fips'].map(fips_to_name)
    state_scores = state_scores.dropna(subset=['state_name'])

    # Relative scaling: lowest state average becomes 0, highest becomes 3.
    state_scores = add_relative_scaled_score(
        state_scores,
        score_col='avg_score',
        scaled_col='scaled_score',
        min_value=0.0,
        max_value=3.0,
    )

    state_scores = state_scores[[
        'state_fips', 'state_name', 'avg_score', 'scaled_score', 'count'
    ]].sort_values('avg_score', ascending=False)

# â”€â”€ Hero â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.markdown("""
<div class="hero">
  <p class="hero-title">U.S. DROUGHT MONITOR</p>
  <p class="hero-sub">// METEOROLOGICAL & SOIL ANALYSIS DASHBOARD</p>
</div>
""", unsafe_allow_html=True)
 
if is_demo:
    st.info("âš ï¸ **Demo Mode** â€” Place `test.csv` and `soil_data.csv` in the same directory to use real data.")


# â”€â”€ KPI row â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
c1, c2, c3, c4 = st.columns(4)
metrics = [
    ("RECORDS", f"{len(df):,}"),
    ("AVG TEMP (Â°C)", f"{df['t2m'].mean():.1f}" if 't2m' in df else "â€”"),
    ("AVG PRECIP (mm)", f"{df['prectot'].mean():.2f}" if 'prectot' in df else "â€”"),
    ("DROUGHT â‰¥ D2 (%)", f"{(df['score'] >= 2).mean()*100:.1f}%"),
]
for col, (lbl, val) in zip([c1, c2, c3, c4], metrics):
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{lbl}</div>
      <div class="metric-value">{val}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# â”€â”€ Tab layout â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
tab1, tab2, tab3, tab4, tab5 = st.tabs(["ðŸ—ºï¸ Map", "ðŸ“ˆ Time Series", "ðŸ”¬ Feature Analysis", "ðŸ“Š Distributions", "ðŸ“‹ State Scores"])

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# TAB 1 â€“ MAP
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
with tab1:
    st.markdown('<p class="section-header">Geographic Drought Distribution</p>', unsafe_allow_html=True)

    map_col, info_col = st.columns([3, 1])

    with map_col:
        if not state_scores.empty:
            agg = state_scores.copy()
            agg['drought_label'] = (
                agg['avg_score']
                .round()
                .clip(lower=0, upper=5)
                .astype(int)
                .map(DROUGHT_LABELS)
            )

            fig_map = px.choropleth_mapbox(
                agg,
                geojson=states_geojson,
                locations='state_name',
                color='scaled_score',
                color_continuous_scale=list(DROUGHT_COLORS.values()),
                range_color=[0, 3],
                mapbox_style="carto-darkmatter",
                zoom=3.2,
                center={"lat": 37.5, "lon": -96},
                opacity=0.8,
                hover_name='state_name',
                hover_data={
                    'avg_score': ':.3f',
                    'scaled_score': ':.3f',
                    'count': True,
                    'state_fips': False,
                },
                labels={
                    'avg_score': 'Actual Avg Score',
                    'scaled_score': 'Relative Score (0-3)',
                    'count': 'Records',
                },
                featureidkey="properties.name"
            )

            fig_map.update_layout(
                **PLOTLY_LAYOUT,
                height=500,
                coloraxis_colorbar=dict(
                    title="Relative<br>Score",
                    tickvals=[0, 1, 2, 3],
                    ticktext=["Lowest", "1", "2", "Highest"]
                ),
            )

            st.plotly_chart(fig_map, use_container_width=True)
            
        else:
            st.warning("FIPS column required for choropleth map.")
 
    # Legend/Map Key
    with info_col:
        st.markdown("**Relative Map Scale**")
        if not state_scores.empty:
            min_row = state_scores.loc[state_scores['avg_score'].idxmin()]
            max_row = state_scores.loc[state_scores['avg_score'].idxmax()]
            st.markdown(
                f"""
                <div style="color:#8a7a60;font-size:.85rem;line-height:1.45;">
                  <b style="color:#e8dcc8;">0</b> = lowest state average in the current data<br>
                  <span>{min_row['state_name']}: {min_row['avg_score']:.3f}</span><br><br>
                  <b style="color:#e8dcc8;">3</b> = highest state average in the current data<br>
                  <span>{max_row['state_name']}: {max_row['avg_score']:.3f}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("**Record Drought Levels**")
        class_dist = get_class_distribution(df['score'])
        for lvl, label in DROUGHT_LABELS.items():
            if lvl in class_dist['counts']:
                color = DROUGHT_COLORS[lvl]
                count = class_dist['counts'][lvl]
                pct = class_dist['percentages'][lvl]
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:8px;margin:6px 0;font-size:.85rem;">
                  <div style="width:12px;height:12px;border-radius:2px;background:{color};flex-shrink:0;"></div>
                  <div style="color:#e8dcc8;">{label}<br>
                    <span style="color:#8a7a60;font-size:.75rem;">{count:,} ({pct:.1f}%)</span>
                  </div>
                </div>""", unsafe_allow_html=True)
 
    # Region comparison graph
    if region_compare and 'fips' in df.columns:
        st.markdown('<p class="section-header">Region Comparison</p>', unsafe_allow_html=True)
        reg_df = df[df['fips'].isin(region_compare)]
        if not reg_df.empty and 'date' in reg_df.columns:
            fig_reg = px.line(
                reg_df.sort_values('date'), x='date', y='score', color='fips',
                title="Drought Score Over Time â€” Selected Regions",
                labels={'score': 'Drought Score', 'date': 'Date', 'fips': 'County FIPS'}
            )
            fig_reg.update_layout(**PLOTLY_LAYOUT, height=300)
            st.plotly_chart(fig_reg, use_container_width=True)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# TAB 2 â€“ TIME SERIES
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
with tab2:
    st.markdown('<p class="section-header">Temporal Patterns</p>', unsafe_allow_html=True)
 
    if 'date' in df.columns:
        ts = df.groupby('date').agg(
            score=('score', 'mean'),
            t2m=('t2m', 'mean') if 't2m' in df else ('score', 'count'),
            prectot=('prectot', 'mean') if 'prectot' in df else ('score', 'count')
        ).reset_index()
 
        fig_ts = make_subplots(rows=3, cols=1, shared_xaxes=True,
                               subplot_titles=("Avg Drought Score", "Avg Temperature (Â°C)", "Avg Precipitation (mm)"),
                               vertical_spacing=0.08)
 
        fig_ts.add_trace(go.Scatter(x=ts['date'], y=ts['score'], mode='lines',
                                    line=dict(color='#c8873a', width=2), name='Drought Score',
                                    fill='tozeroy', fillcolor='rgba(200,135,58,0.15)'), row=1, col=1)
        
        if 't2m' in df.columns:
            fig_ts.add_trace(go.Scatter(x=ts['date'], y=ts['t2m'], mode='lines',
                                        line=dict(color='#e87030', width=1.5), name='Temp'), row=2, col=1)
        
        if 'prectot' in df.columns:
            fig_ts.add_trace(go.Bar(x=ts['date'], y=ts['prectot'],
                                    marker_color='#4a9e6b', opacity=0.7, name='Precip'), row=3, col=1)
 
        fig_ts.update_layout(**PLOTLY_LAYOUT, height=520, showlegend=False)
        for i in range(1, 4):
            fig_ts.update_xaxes(gridcolor='#2a2216', linecolor='#3a3020', row=i, col=1)
            fig_ts.update_yaxes(gridcolor='#2a2216', linecolor='#3a3020', row=i, col=1)
        st.plotly_chart(fig_ts, use_container_width=True)
 
        # Monthly heatmap
        st.markdown('<p class="section-header">Monthly Drought Heatmap</p>', unsafe_allow_html=True)
        if 'month' in df.columns and 'year' in df.columns:
            pivot = df.groupby(['year', 'month'])['score'].mean().unstack()
            fig_heat = go.Figure(go.Heatmap(
                z=pivot.values,
                x=[pd.Timestamp(2000, m, 1).strftime('%b') for m in pivot.columns],
                y=pivot.index.astype(str),
                colorscale=list(DROUGHT_COLORS.values()),
                zmin=0, zmax=5,
                hoverongaps=False,
                colorbar=dict(title='Score')
            ))
            fig_heat.update_layout(**PLOTLY_LAYOUT, height=300, title="Mean Drought Score by Year Ã— Month")
            st.plotly_chart(fig_heat, use_container_width=True)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# TAB 3 â€“ FEATURE ANALYSIS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
with tab3:
    st.markdown('<p class="section-header">Feature Relationships</p>', unsafe_allow_html=True)
 
    col_a, col_b = st.columns(2)
 
    with col_a:
        # Scatter: T2M vs PRECTOT
        if 't2m' in df.columns and 'prectot' in df.columns:
            sample = df.sample(min(2000, len(df)), random_state=42)
            fig_sc = px.scatter(
                sample, x='t2m', y='prectot', color='score',
                color_continuous_scale=list(DROUGHT_COLORS.values()),
                range_color=[0,5], opacity=0.6,
                title="Temperature vs Precipitation",
                labels={'t2m':'Temperature (Â°C)', 'prectot':'Precipitation (mm)'}
            )
            fig_sc.update_layout(**PLOTLY_LAYOUT, height=370)
            st.plotly_chart(fig_sc, use_container_width=True)
 
    with col_b:
        # Correlation heatmap
        corr = get_correlation_matrix(df)
        if not corr.empty:
            # Limit to top features
            top_features = corr['score'].abs().sort_values(ascending=False).head(10).index.tolist()
            corr_subset = corr.loc[top_features, top_features]
            
            fig_corr = go.Figure(go.Heatmap(
                z=corr_subset.values,
                x=corr_subset.columns, y=corr_subset.columns,
                colorscale='RdYlGn', zmid=0,
                text=corr_subset.round(2).values,
                texttemplate='%{text}',
                textfont=dict(size=9),
                hoverongaps=False,
            ))
            fig_corr.update_layout(**PLOTLY_LAYOUT, height=370, title="Feature Correlation Matrix")
            st.plotly_chart(fig_corr, use_container_width=True)
 
    # Box plots by drought level
    st.markdown('<p class="section-header">Variable Distribution by Drought Level</p>', unsafe_allow_html=True)
    weather_vars = [c for c in ['t2m', 'prectot', 'ws10m', 'qv2m'] if c in df.columns]
    if weather_vars:
        sel_var = st.selectbox("Variable", weather_vars)
        fig_box = px.box(
            df, x='score', y=sel_var,
            color='score',
            color_discrete_map=DROUGHT_COLORS,
            labels={'score': 'Drought Score'},
            title=f"{sel_var.upper()} distribution across drought levels",
            category_orders={'score': sorted(df['score'].unique())}
        )
        fig_box.update_layout(**PLOTLY_LAYOUT, height=360, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# TAB 4 â€“ DISTRIBUTIONS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
with tab4:
    st.markdown('<p class="section-header">Score Distributions</p>', unsafe_allow_html=True)
 
    d_col1, d_col2 = st.columns(2)
 
    with d_col1:
        counts = df['score'].value_counts().sort_index()
        fig_bar = go.Figure(go.Bar(
            x=[DROUGHT_LABELS.get(i, str(i)) for i in counts.index],
            y=counts.values,
            marker_color=[DROUGHT_COLORS.get(i, '#888') for i in counts.index],
            text=counts.values, textposition='outside',
        ))
        fig_bar.update_layout(**PLOTLY_LAYOUT, height=380, title="Record Count per Drought Level",
                              xaxis_tickangle=-30)
        st.plotly_chart(fig_bar, use_container_width=True)
 
    with d_col2:
        fig_pie = go.Figure(go.Pie(
            labels=[DROUGHT_LABELS.get(i, str(i)) for i in counts.index],
            values=counts.values,
            marker_colors=[DROUGHT_COLORS.get(i, '#888') for i in counts.index],
            hole=0.45,
            textinfo='percent+label',
            textfont=dict(size=11),
        ))
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Syne', color='#e8dcc8'),
            height=380, title="Drought Level Share",
            showlegend=False, margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_pie, use_container_width=True)
 

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# TAB 5 â€“ STATE SCORES
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
with tab5:
    st.markdown('<p class="section-header">Actual Average Drought Scores by State</p>', unsafe_allow_html=True)
    st.caption(
        "The table keeps the original unscaled average score. "
        "The scaled score is included only to show how the map color was calculated."
    )

    if not state_scores.empty:
        table_df = state_scores[[
            'state_name', 'avg_score', 'scaled_score', 'count'
        ]].rename(columns={
            'state_name': 'State',
            'avg_score': 'Actual Avg Score',
            'scaled_score': 'Relative Score (0-3)',
            'count': 'Records',
        })

        st.dataframe(
            table_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Actual Avg Score': st.column_config.NumberColumn(format='%.4f'),
                'Relative Score (0-3)': st.column_config.NumberColumn(format='%.4f'),
                'Records': st.column_config.NumberColumn(format='%d'),
            },
        )

        fig_state_scores = px.bar(
            state_scores.sort_values('avg_score', ascending=True),
            x='avg_score',
            y='state_name',
            orientation='h',
            color='avg_score',
            color_continuous_scale=list(DROUGHT_COLORS.values()),
            range_color=[0, 5],
            labels={
                'avg_score': 'Actual Avg Score',
                'state_name': 'State',
            },
            title='Unscaled Average Drought Score by State',
        )
        fig_state_scores.update_layout(**PLOTLY_LAYOUT, height=900, showlegend=False)
        st.plotly_chart(fig_state_scores, use_container_width=True)
    else:
        st.warning("FIPS column required to calculate state-level averages.")


# â”€â”€ Footer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.markdown("---")
st.markdown(
    f"<p style='color:#4a3820;font-family:Space Mono,monospace;font-size:.75rem;text-align:center;'>"
    f"U.S. DROUGHT MONITOR DASHBOARD Â· Test Set: {len(test_df):,} records</p>",
    unsafe_allow_html=True
)