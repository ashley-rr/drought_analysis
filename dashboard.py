import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from drought_utils import (
    load_complete_dataset,
    get_class_distribution,
    get_correlation_matrix,
    SCALED_DROUGHT_LABELS,
    DROUGHT_COLORS,
    FIPS_TO_NAME
)

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

# Import FIPS coordinate mapping
try:
    from fips_coordinates import add_coordinates_to_dataframe
    HAS_FIPS_COORDS = True
except ImportError:
    HAS_FIPS_COORDS = False

# --------------------- Page config ---------------------
st.set_page_config(
    page_title="U.S. Drought Monitor",
    page_icon="🌞",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------- Custom CSS ---------------------
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


# --------------------- Data loading ---------------------
@st.cache_data
def load_data():
    data = load_complete_dataset(
        data_path="train.csv",
        soil_path="soil_data.csv",
    )

    return data

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
)

st.markdown("""
    <style>
    .st-emotion-cache-1ix68xf h1, 
    .st-emotion-cache-1ix68xf h2, 
    .st-emotion-cache-1ix68xf h3, 
    .st-emotion-cache-1ix68xf h4, 
    .st-emotion-cache-1ix68xf h5, 
    .st-emotion-cache-1ix68xf h6 {
        color: #acaeb0;
        font-family: "Source Sans", sans-serif;
        text-align: center;
        font-size: 20px;
        !opacity: 1
    }
 
    tspan {
        fill: #acaeb0;
        font-size: 18px;
    }
    </style>
    """, unsafe_allow_html=True)

# --------------------- Sidebar ---------------------
with st.sidebar:
    st.markdown("## Controls")

# st.title("Styled Streamlit App")

    df_active = test_df

    st.markdown("---")

    all_scores = sorted(df_active['score'].unique())[:4]
    
    score_filter = st.multiselect(
        "Drought Levels",
        options=all_scores,
        default=all_scores,
        format_func=lambda x: SCALED_DROUGHT_LABELS.get(x, str(x))
    )

    st.markdown("---")
    if 'year' in df_active.columns:
        years = sorted(df_active['year'].unique())
        year_range = st.select_slider("Year Range", options=years, value=(years[0], years[-1]))
    else:
        year_range = None
 
    st.markdown("---")
    if 'fips' in df_active.columns:

        # Create state FIPS from county FIPS
        df_active['state_fips'] = (
            df_active['fips']
            .astype(str)
            .str.zfill(5)
            .str[:2]
        )

        # Convert to readable state names
        df_active['state_name'] = (
            df_active['state_fips']
            .map(FIPS_TO_NAME)
        )

        # Sorted list of available states
        available_states = sorted(
            df_active['state_name']
            .dropna()
            .unique()
            .tolist()
        )

        # User selects up to 3 states
        region_compare = st.multiselect(
            "Compare States",
            options=available_states,
            default=available_states[:3],
            max_selections=3
        )

    else:
        region_compare = []


# --------------------- Sidebar Filter ---------------------

# Make copy of dataset for filter
df = df_active.copy()

if year_range and 'year' in df.columns:
    df = df[
        (df['year'] >= year_range[0]) &
        (df['year'] <= year_range[1])
    ]

# --------------------- State-level aggregation ---------------------
state_scores = pd.DataFrame()

if 'fips' in df.columns:
    # Convert county FIPS to state FIPS using the first 2 digits.
    df['state_fips'] = df['fips'].astype(str).str.zfill(5).str[:2]

    state_scores = df.groupby('state_fips').agg(
        avg_score=('score', 'mean'),
        count=('score', 'count')
    ).reset_index()

    state_scores['state_name'] = state_scores['state_fips'].map(FIPS_TO_NAME)
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

# --------------------- Hero ---------------------
st.markdown("""
<div class="hero">
  <p class="hero-title">U.S. DROUGHT MONITOR</p>
  <p class="hero-sub">// METEOROLOGICAL & SOIL ANALYSIS DASHBOARD</p>
</div>
""", unsafe_allow_html=True)

# --------------------- KPI row ---------------------
c1, c2, c3, c4 = st.columns(4)
metrics = [
    ("RECORDS", f"{len(df):,}"),
    ("AVG TEMP (F°)", f"{(df['t2m'] * 1.8 + 30).mean():.1f}" if 't2m' in df else "---"),
    ("AVG PRECIP (mm)", f"{df['prectot'].mean():.2f}" if 'prectot' in df else "---"),
    ("DROUGHT ≥ D2 (%)", f"{(df['score'] >= 2).mean()*100:.1f}%"),
]
for col, (lbl, val) in zip([c1, c2, c3, c4], metrics):
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{lbl}</div>
      <div class="metric-value">{val}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --------------------- Tab layout ---------------------
tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Map", "⌛ Time Series", "💡 Feature Analysis", "📈 State Scores"])

# --------------------- Map Tab ---------------------
with tab1:
    st.markdown('<p class="section-header">Geographic Drought Distribution</p>', unsafe_allow_html=True)

    map_col, info_col = st.columns([3, 1])
    
    with map_col:
        if not state_scores.empty:
            agg = state_scores.copy()

            agg['drought_label'] = (
                agg['scaled_score']
                .round()
                .clip(lower=0, upper=3)
                .astype(int)
                .map(SCALED_DROUGHT_LABELS)
            )

            filtered_agg = agg

            # Filter for drought labels selected by user (unless filter is empty) 
            if (len(score_filter) > 0):
                selected_labels = [SCALED_DROUGHT_LABELS[s] for s in score_filter]
                filtered_agg = agg[agg['drought_label'].isin(selected_labels)]

            fig_map = px.choropleth_mapbox(
                filtered_agg,
                geojson=states_geojson,
                locations='state_name',
                color='scaled_score',
                color_continuous_scale=list(DROUGHT_COLORS.values()),
                range_color=[0, 3],
                mapbox_style="carto-darkmatter",
                zoom=2.6,
                center={"lat": 39.8282, "lon": -95.7129},
                hover_name='state_name',
                hover_data={
                    'avg_score': ':.3f',
                    'scaled_score': ':.3f',
                    'count': True,
                    'state_fips': False,
                },
                labels={
                    'state_name': 'State',
                    'avg_score': 'Actual Average Score',
                    'scaled_score': 'Relative Score',
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
 
    # Highest and lowest state scores
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

        # Map legend
        st.markdown("---")
        st.markdown("**Record Drought Levels**")
        class_dist = get_class_distribution(df['score'])

        # Count unique states
        state_counts = agg['drought_label'].value_counts().sort_index()

        # Convert to percentages
        state_percentages = (state_counts / state_counts.sum()) * 100

        for lvl, label in SCALED_DROUGHT_LABELS.items():
            if lvl in class_dist['counts']:
                color = DROUGHT_COLORS[lvl]
                count = state_counts[label]
                pct = state_percentages[label]
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
        reg_df = df[df['state_name'].isin(region_compare)]
        
        if not reg_df.empty and 'date' in reg_df.columns:
            # Average drought score per state per year
            state_compare = (
                reg_df
                .groupby(['year', 'state_name'])['score']
                .mean()
                .reset_index()
            )

            fig_reg = px.line(
                state_compare.sort_values('year'),
                x='year',
                y='score',
                color='state_name',
                markers=True,
                title="Average Drought Score by State Over Time",
                labels={
                    'score': 'Average Drought Score',
                    'year': 'Year',
                    'state_name': 'State'
                }
            )

            fig_reg.update_layout(
                **PLOTLY_LAYOUT,
                height=400
            )

            fig_reg.update_yaxes(
                tickvals=[0,1,2,3],
                ticktext=["None", "Abnormal", "Moderate", "Severe"]
            )
                    
            st.plotly_chart(fig_reg, use_container_width=True)

# --------------------- Time Series Tab ---------------------
with tab2:
    st.markdown('<p class="section-header">Temporal Patterns</p>', unsafe_allow_html=True)
 
    if 'date' in df.columns:
        
        # Get year from date
        df['year'] = df['date'].dt.year
        df['t2m'] = df['t2m'] * 1.8 + 32

        # State-year average
        state_year = df.groupby(['year', 'state_fips']).agg(
            state_score=('score', 'mean'),
            t2m=('t2m', 'mean'),
            prectot=('prectot', 'mean')
        ).reset_index()

        # Average across states per year
        ts = df.groupby('year')[['score', 't2m', 'prectot']].mean().reset_index()

        fig_ts = make_subplots(rows=3, cols=1, x_title="Year",
            subplot_titles=("Average Drought Score", "Average Temperature (F°)", "Average Precipitation (mm)"),
            vertical_spacing=0.15,
            )
 
        fig_ts.add_trace(go.Scatter(
            x=ts['year'],
            y=ts['score'],
            mode='lines+markers',
            line=dict(color='#c8873a', width=2),
            name='Drought Score',
            fill='tozeroy',
            fillcolor='rgba(200,135,58,0.15)'
        ), row=1, col=1)

        fig_ts.update_annotations(font_size=25)

        fig_ts.update_yaxes(
            tickvals=[0, 1, 2, 3],
            ticktext=["None","Abnormal","Moderate","Severe"],
            row=1, col=1
        )

        fig_ts.add_trace(go.Scatter(x=ts['year'], y=ts['t2m'], mode='lines',
            line=dict(color='#e87030', width=1.5), name='Temp'), row=2, col=1)
    
        fig_ts.add_trace(go.Bar(x=ts['year'], y=ts['prectot'],
            marker_color='#4a9e6b', name='Precip'), row=3, col=1)
 
        fig_ts.update_layout(**PLOTLY_LAYOUT, height=1500, showlegend=False, margin=dict(l=220, r=120))
        
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
                zmin=-0.25, zmax=2.5,
                hoverongaps=False,
                colorbar=dict(title='Score')
            ))
            fig_heat.update_layout(**PLOTLY_LAYOUT, height=600, margin=dict(l=150, r=150), title="Mean Drought Score by Year & Month")
            fig_ts.update_annotations(font_size=60)
            st.plotly_chart(fig_heat, use_container_width=True)


# --------------------- Feature Analysis Tab ---------------------
with tab3:
    st.markdown('<p class="section-header">Feature Relationships</p>', unsafe_allow_html=True)
 
    col_a, col_b = st.columns(2)
 
    # Scatter: T2M vs PRECTOT
    if 't2m' in df.columns and 'prectot' in df.columns:
        # Convert to Celsius
        df["t2m"] = df["t2m"] / 1.8 - 32                           
        sample = df.sample(min(300, len(df)), random_state=42)
        fig_sc = px.scatter(
            sample, x='t2m', y='prectot', color='score',
            color_continuous_scale=list(DROUGHT_COLORS.values()),
            range_color=[0,3],
            title="Temperature vs Precipitation (Random Sample of 300 Points)",
            labels={'t2m':'Temperature (C°)', 'prectot':'Precipitation (mm)'}
        )
        fig_sc.update_layout(**PLOTLY_LAYOUT, height=500, margin=dict(l=200, r=200))
        st.plotly_chart(fig_sc, use_container_width=True)

    # Correlation heatmap
    corr = get_correlation_matrix(df)
    if not corr.empty:

        top_features = corr['score'].abs().sort_values(ascending=False).head(10).index.tolist()
        corr_subset = corr.loc[top_features, top_features]

        fig_corr = go.Figure(go.Heatmap(
            z=corr_subset.values, zmin=-0.5, zmax=1,
            x=corr_subset.columns, y=corr_subset.columns,
            colorscale='RdYlGn',
            text=corr_subset.round(2).values,
            texttemplate='%{text}',
            textfont=dict(size=9),
            hoverongaps=False,
        ))
        fig_corr.update_layout(**PLOTLY_LAYOUT, height=500, margin=dict(l=200, r=200), title="Top 10 Most Correlated Features")
        st.plotly_chart(fig_corr, use_container_width=True)

# --------------------- State scores tab ---------------------
with tab4:
    st.markdown('<p class="section-header">Average Scores, Scaled Scores, and Records by State</p>', unsafe_allow_html=True)
    st.caption(
        "The table also shows the original unscaled average score. "
    )

    if not state_scores.empty:
        table_df = state_scores[[
            'state_name', 'avg_score', 'scaled_score', 'count'
        ]].rename(columns={
            'state_name': 'State',
            'avg_score': 'Actual Average Score',
            'scaled_score': 'Relative Score (0-3)',
            'count': 'Number of Records',
        })

        st.dataframe(
            table_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Actual Average Score': st.column_config.NumberColumn(format='%.4f'),
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
            range_color=[0, 3],
            labels={
                'state_name': 'States',
                'avg_score': 'Score'
            },
            title='Unscaled Average Drought Scores per State',
        )
        fig_state_scores.update_layout(**PLOTLY_LAYOUT, height=900, showlegend=False)
        st.plotly_chart(fig_state_scores, use_container_width=True)
    else:
        st.warning("FIPS column required to calculate state-level averages.")


# --------------------- Footer ---------------------
st.markdown("---")
st.markdown(
    f"<p style='color:#4a3820;font-family:Space Mono,monospace;font-size:.75rem;text-align:center;'>"
    f"U.S. DROUGHT MONITOR DASHBOARD: {len(test_df):,} records</p>",
    unsafe_allow_html=True
)
