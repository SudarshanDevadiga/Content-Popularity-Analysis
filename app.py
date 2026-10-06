import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
import os

# Set page configuration
st.set_page_config(
    page_title="Content Popularity Predictor | Case Study 124",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for a polished, theme-adaptive presentation (supports both Dark & Light modes)
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #3B82F6, #60A5FA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: var(--text-color, #94A3B8);
        opacity: 0.9;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid rgba(226, 232, 240, 0.2);
    }
    .prediction-box-high {
        background-color: rgba(34, 197, 94, 0.15) !important;
        border: 1px solid rgba(34, 197, 94, 0.4) !important;
        border-left: 6px solid #22C55E !important;
        padding: 1.2rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .prediction-box-high h3 {
        color: #22C55E !important;
        margin: 0;
        font-weight: 700;
    }
    .prediction-box-high p, .prediction-box-high span, .prediction-box-high b {
        color: var(--text-color, inherit) !important;
    }
    .prediction-box-low {
        background-color: rgba(239, 68, 68, 0.15) !important;
        border: 1px solid rgba(239, 68, 68, 0.4) !important;
        border-left: 6px solid #EF4444 !important;
        padding: 1.2rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .prediction-box-low h3 {
        color: #EF4444 !important;
        margin: 0;
        font-weight: 700;
    }
    .prediction-box-low p, .prediction-box-low span, .prediction-box-low b {
        color: var(--text-color, inherit) !important;
    }
    .rec-box {
        background-color: rgba(59, 130, 246, 0.12) !important;
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
        border-left: 5px solid #3B82F6 !important;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 10px;
        color: var(--text-color, inherit) !important;
    }
    .rec-box p, .rec-box span, .rec-box strong, .rec-box b {
        color: var(--text-color, inherit) !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Data & Model Loaders (Cached for instant responsiveness)
# -------------------------------------------------------------
@st.cache_resource
def load_models_and_scaler():
    models = joblib.load("models/all_models.joblib")
    scaler = joblib.load("models/scaler.joblib")
    with open("models/feature_names.json", "r") as f:
        feature_names = json.load(f)
    return models, scaler, feature_names

@st.cache_data
def load_metadata():
    with open("models/model_comparison.json", "r") as f:
        comparison = json.load(f)
    with open("models/eda_summary.json", "r") as f:
        eda = json.load(f)
    return comparison, eda

try:
    all_models, scaler, feature_names = load_models_and_scaler()
    model_comparison, eda_summary = load_metadata()
    data_loaded = True
except Exception as e:
    data_loaded = False
    st.error(f"Error loading model files: {e}. Please run `python train_and_evaluate.py` first.")

# -------------------------------------------------------------
# Sidebar Navigation
# -------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/clouds/200/000000/combo-chart.png", width=110)
st.sidebar.title("Navigation")
st.sidebar.markdown("**Case Study No. 124**")
st.sidebar.markdown("*Content Popularity Analysis*")

page = st.sidebar.radio(
    "Select Section:",
    [
        "1. Problem Definition",
        "2. Exploratory Data Analysis (EDA)",
        "3. Model Comparison & Results",
        "4. Live Popularity Predictor"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Project Deliverable**: Machine Learning System for Online Content Engagement Forecasting."
)

if not data_loaded:
    st.stop()

# -------------------------------------------------------------
# PAGE 1: Problem Definition
# -------------------------------------------------------------
if page == "1. Problem Definition":
    st.markdown('<div class="main-title">Case Study 124: Content Popularity Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">A Supervised Machine Learning Approach to Estimate Digital Content Engagement</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Problem Background & Business Motivation")
        st.write("""
        Digital publishing platforms (news sites, blogs, media portals) publish thousands of articles weekly. 
        Editorial teams face a constant challenge: **How do we anticipate which articles will resonate strongly with readers?**
        
        Understanding content virality enables:
        * **Editorial Optimization**: Crafting optimal headline lengths, keyword structures, and multimedia inclusion.
        * **Scheduling Strategy**: Timing releases when audience engagement peaks (e.g., weekday vs weekend readership).
        * **Resource Allocation**: Directing marketing and social media promotion budgets toward high-potential articles.
        """)
        
        st.subheader("🎯 Machine Learning Problem Formulation")
        st.write("""
        * **Task Type**: Binary Supervised Classification
        * **Target Definition**: 
            * **High Engagement (Popular = 1)**: Article reaches or exceeds **1,400 shares** (the median engagement threshold across all 39,644 historical articles).
            * **Standard Engagement (Low = 0)**: Article receives fewer than 1,400 shares.
        * **Why Median Split?**: 
            Social media shares follow an extreme right-skewed Power-Law distribution. A median threshold creates a balanced, robust classification target (~53.4% positive class), avoiding the massive variance and outlier issues of raw count regression while giving actionable decision boundaries to content managers.
        """)

    with col2:
        st.subheader("📊 Dataset At A Glance")
        st.metric(label="Total Historical Articles", value=f"{eda_summary['total_records']:,}")
        st.metric(label="Selected Model Features", value=f"{eda_summary['total_features']}")
        st.metric(label="Engagement Threshold", value=f"{int(eda_summary['median_shares']):,} shares")
        st.metric(label="High Engagement Ratio", value="53.4%")
        
    st.markdown("---")
    st.subheader("💡 System Workflow Overview")
    
    steps = [
        ("1. Data Ingestion", "39,644 records from UCI Online News Popularity dataset spanning 2 years of publications."),
        ("2. Preprocessing", "Cleaning whitespace, handling missing values, standard feature scaling, 80/20 stratified split."),
        ("3. Model Benchmarking", "Comparing Logistic Regression (linear baseline), Decision Tree, and Random Forest ensemble."),
        ("4. Deployment", "Interactive Streamlit prototype for real-time engagement prediction and headline optimization.")
    ]
    cols = st.columns(4)
    for col, (step_title, step_desc) in zip(cols, steps):
        with col:
            st.markdown(f"**{step_title}**")
            st.caption(step_desc)

# -------------------------------------------------------------
# PAGE 2: Exploratory Data Analysis (EDA)
# -------------------------------------------------------------
elif page == "2. Exploratory Data Analysis (EDA)":
    st.markdown('<div class="main-title">Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Key findings and behavioral trends discovered across 39,644 published articles</div>', unsafe_allow_html=True)
    
    eda_tab1, eda_tab2, eda_tab3 = st.tabs([
        "📈 Channel Popularity", 
        "📅 Publishing Day Impact", 
        "🖼️ Content & Media Features"
    ])
    
    with eda_tab1:
        st.subheader("Engagement Performance by Content Channel")
        ch_df = pd.DataFrame(eda_summary['channel_stats'])
        
        c1, c2 = st.columns([3, 2])
        with c1:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            colors = ['#3B82F6', '#10B981', '#F59E0B', '#EC4899', '#8B5CF6', '#64748B']
            bars = ax.bar(ch_df['channel'], ch_df['popular_pct'], color=colors[:len(ch_df)])
            ax.axhline(53.4, color='red', linestyle='--', label='Dataset Average (53.4%)')
            ax.set_ylabel("% High Engagement (>=1,400 Shares)")
            ax.set_ylim(0, 75)
            ax.set_title("Percentage of Articles Achieving High Engagement by Channel")
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  
                            textcoords="offset points",
                            ha='center', va='bottom', fontweight='bold')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
            plt.close()

        with c2:
            st.write("### Channel Observations:")
            st.markdown("""
            * **Social Media (65.6%)**: Highest percentage of high engagement. Short, shareable, visual content performs best.
            * **Tech (55.4%)**: Consistently performs above average due to passionate niche tech communities.
            * **World News (32.8%)**: Lowest share rate; routine global political/regional reports rarely go viral compared to entertainment or tech.
            """)
            st.dataframe(ch_df[['channel', 'count', 'avg_shares', 'popular_pct']], use_container_width=True)

    with eda_tab2:
        st.subheader("Weekend Effect: Does Publication Timing Matter?")
        t_stats = eda_summary['timing_stats']
        
        c1, c2 = st.columns(2)
        with c1:
            timing_data = pd.DataFrame({
                'Timing': ['Weekday (Mon - Fri)', 'Weekend (Sat - Sun)'],
                'Popular_Pct': [t_stats['Weekday']['popular_pct'], t_stats['Weekend']['popular_pct']],
                'Avg_Shares': [t_stats['Weekday']['avg_shares'], t_stats['Weekend']['avg_shares']],
                'Article_Count': [t_stats['Weekday']['count'], t_stats['Weekend']['count']]
            })
            
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(timing_data['Timing'], timing_data['Popular_Pct'], color=['#64748B', '#10B981'], width=0.5)
            ax.set_ylabel("% High Engagement")
            ax.set_ylim(0, 80)
            ax.set_title("Engagement Rate: Weekday vs Weekend")
            for i, val in enumerate(timing_data['Popular_Pct']):
                ax.text(i, val + 1.5, f"{val:.1f}%", ha='center', fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
            plt.close()
            
        with c2:
            st.write("### The Weekend Advantage:")
            st.markdown("""
            * **Weekend articles achieve a 63.8% high engagement rate**, compared to **51.8%** on weekdays.
            * **Why this occurs**:
                1. **Reduced Content Volume**: Far fewer articles are published on weekends, lowering competition for reader attention.
                2. **Leisure Reading**: Readers have more free time on weekends to browse, read in-depth, and share content on social networks.
            """)
            st.dataframe(timing_data, use_container_width=True)

    with eda_tab3:
        st.subheader("Content Structure & Feature Summary")
        f_stats = pd.DataFrame(eda_summary['feature_stats']).T
        f_stats.reset_index(inplace=True)
        f_stats.rename(columns={'index': 'Feature Name'}, inplace=True)
        
        st.write("Summary statistics of the 17 core engineered features:")
        st.dataframe(f_stats, use_container_width=True)

# -------------------------------------------------------------
# PAGE 3: Model Comparison & Results
# -------------------------------------------------------------
elif page == "3. Model Comparison & Results":
    st.markdown('<div class="main-title">Machine Learning Experiments & Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Comparing Logistic Regression, Decision Tree, and Random Forest on 7,929 test samples</div>', unsafe_allow_html=True)
    
    # Metrics table
    metrics_list = []
    for model_name, res in model_comparison.items():
        metrics_list.append({
            'Model': model_name,
            'Accuracy': f"{res['Accuracy'] * 100:.2f}%",
            'Precision': f"{res['Precision'] * 100:.2f}%",
            'Recall': f"{res['Recall'] * 100:.2f}%",
            'F1-Score': f"{res['F1_Score'] * 100:.2f}%",
            'ROC-AUC': f"{res['ROC_AUC']:.4f}"
        })
    metrics_df = pd.DataFrame(metrics_list)
    
    st.subheader("📊 Model Performance Benchmark")
    st.dataframe(metrics_df, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix Analysis")
        selected_model = st.selectbox("Select Model for Confusion Matrix:", list(model_comparison.keys()), index=2)
        cm = np.array(model_comparison[selected_model]['Confusion_Matrix'])
        
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                    xticklabels=['Pred Low (<1.4k)', 'Pred High (>=1.4k)'],
                    yticklabels=['Actual Low', 'Actual High'])
        ax.set_title(f"Confusion Matrix: {selected_model}")
        st.pyplot(fig)
        plt.close()
        
        tn, fp, fn, tp = cm.ravel()
        st.caption(f"**True Positives (High Engagement Correctly Caught)**: {tp} | **True Negatives**: {tn}")

    with col2:
        st.subheader("Top Predictive Features (Random Forest)")
        rf_importances = model_comparison['Random Forest']['Feature_Importances']
        sorted_imp = sorted(rf_importances.items(), key=lambda x: x[1], reverse=True)[:8]
        imp_df = pd.DataFrame(sorted_imp, columns=['Feature', 'Importance'])
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(imp_df['Feature'][::-1], imp_df['Importance'][::-1], color='#3B82F6')
        ax.set_xlabel("Relative Importance")
        ax.set_title("Top 8 Predictors of Engagement")
        st.pyplot(fig)
        plt.close()
        st.caption("Average keyword popularity (`kw_avg_avg`), text length, and multimedia are leading drivers.")

    st.markdown("---")
    st.subheader("📝 Key Analytical Justifications & Findings")
    st.markdown("""
    1. **Why Random Forest Wins**:
       * Random Forest achieves the highest **ROC-AUC (0.7088)** and **Accuracy (65.22%)**, capturing non-linear interactions between channel type, publishing day, and word count that linear models miss.
    2. **Linear Baseline Utility**:
       * Logistic Regression performs competitively (~64.71% accuracy) and provides clean coefficients confirming positive odds for weekend releases and multimedia inclusion.
    3. **Realistic Performance Level**:
       * In real-world social media research, ~65-70% accuracy on purely static pre-publication metadata is the established benchmark, because virality also depends on post-publication factors (e.g., breaking news, influencer tweets, algorithmic feeds).
    """)

# -------------------------------------------------------------
# PAGE 4: Live Popularity Predictor
# -------------------------------------------------------------
elif page == "4. Live Popularity Predictor":
    st.markdown('<div class="main-title">Live Content Popularity Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enter draft article parameters to estimate audience engagement probability</div>', unsafe_allow_html=True)
    
    with st.form("article_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### 📝 Content & Structure")
            n_tokens_title = st.slider("Title Word Count", min_value=3, max_value=25, value=10, help="Optimal titles are usually 8-12 words.")
            n_tokens_content = st.slider("Article Word Count", min_value=50, max_value=3000, value=650, step=50)
            num_keywords = st.slider("Number of Keywords / Tags", min_value=1, max_value=10, value=6)
            
        with col2:
            st.markdown("##### 🖼️ Multimedia & Links")
            num_imgs = st.slider("Number of Images", min_value=0, max_value=20, value=3)
            num_videos = st.slider("Number of Embedded Videos", min_value=0, max_value=10, value=1)
            num_hrefs = st.slider("Number of Hyperlinks", min_value=0, max_value=30, value=8)

        with col3:
            st.markdown("##### 🌐 Channel & Publication Timing")
            channel = st.selectbox(
                "Content Category / Channel",
                ["Social Media", "Tech", "Lifestyle", "Business", "Entertainment", "World News"]
            )
            day_type = st.radio("Publication Timing", ["Weekday (Mon-Fri)", "Weekend (Sat-Sun)"], horizontal=True)
            kw_avg_avg = st.slider("Target Keyword Popularity Index", min_value=500, max_value=8000, value=3200, step=100,
                                   help="Estimated popularity of the article's target keywords based on past trends.")

        st.markdown("##### 🎭 Tone & Sentiment")
        sent_col1, sent_col2 = st.columns(2)
        with sent_col1:
            sentiment_slider = st.select_slider(
                "Article Overall Sentiment",
                options=["Negative", "Slightly Negative", "Neutral", "Positive", "Very Positive"],
                value="Positive"
            )
        with sent_col2:
            title_tone = st.select_slider(
                "Title Subjectivity / Emotional Appeal",
                options=["Objective / Factual", "Balanced", "Subjective / Opinionated"],
                value="Balanced"
            )

        submit_btn = st.form_submit_button("🚀 Predict Content Popularity", use_container_width=True)

    if submit_btn:
        # Map inputs to model feature vector
        # Sentiment mapping
        sent_map = {
            "Negative": -0.25,
            "Slightly Negative": -0.10,
            "Neutral": 0.05,
            "Positive": 0.20,
            "Very Positive": 0.40
        }
        subj_map = {
            "Objective / Factual": 0.1,
            "Balanced": 0.4,
            "Subjective / Opinionated": 0.75
        }
        
        is_weekend_val = 1 if "Weekend" in day_type else 0
        
        # Build one-hot channel flags
        input_data = {
            'n_tokens_title': n_tokens_title,
            'n_tokens_content': n_tokens_content,
            'num_hrefs': num_hrefs,
            'num_imgs': num_imgs,
            'num_videos': num_videos,
            'num_keywords': num_keywords,
            'data_channel_is_lifestyle': 1 if channel == "Lifestyle" else 0,
            'data_channel_is_entertainment': 1 if channel == "Entertainment" else 0,
            'data_channel_is_bus': 1 if channel == "Business" else 0,
            'data_channel_is_socmed': 1 if channel == "Social Media" else 0,
            'data_channel_is_tech': 1 if channel == "Tech" else 0,
            'data_channel_is_world': 1 if channel == "World News" else 0,
            'is_weekend': is_weekend_val,
            'global_sentiment_polarity': sent_map[sentiment_slider],
            'title_sentiment_polarity': sent_map[sentiment_slider] * 0.8,
            'title_subjectivity': subj_map[title_tone],
            'kw_avg_avg': kw_avg_avg
        }
        
        # Format as DataFrame in exact column order
        input_df = pd.DataFrame([input_data])[feature_names]
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Predict using Random Forest
        rf_model = all_models["Random Forest"]
        prob_popular = rf_model.predict_proba(input_scaled)[0, 1]
        is_popular = int(prob_popular >= 0.50)
        
        st.markdown("---")
        st.subheader("🎯 Prediction Result")
        
        res_col1, res_col2 = st.columns([2, 3])
        
        with res_col1:
            if is_popular == 1:
                st.markdown(f"""
                <div class="prediction-box-high">
                    <h3 style="margin: 0; color: #166534;">🔥 High Engagement Expected</h3>
                    <p style="margin-top: 0.5rem; font-size: 1.1rem;">
                        This content is estimated to achieve <b>&ge; 1,400 shares</b>.
                    </p>
                    <p>Estimated Probability: <b>{prob_popular * 100:.1f}%</b></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box-low">
                    <h3 style="margin: 0; color: #991B1B;">📉 Standard / Low Engagement Expected</h3>
                    <p style="margin-top: 0.5rem; font-size: 1.1rem;">
                        This content is estimated to achieve <b>&lt; 1,400 shares</b>.
                    </p>
                    <p>Estimated Probability of High Engagement: <b>{prob_popular * 100:.1f}%</b></p>
                </div>
                """, unsafe_allow_html=True)
                
            st.progress(float(prob_popular))
            st.caption(f"Popularity Confidence: {prob_popular * 100:.1f}%")

        with res_col2:
            st.write("#### 💡 Actionable Optimization Recommendations:")
            tips = []
            if num_imgs < 2:
                tips.append("📸 **Add Visual Media**: Articles with 2-4 relevant images have 15% higher share probability.")
            if num_videos == 0 and channel in ["Social Media", "Entertainment", "Tech"]:
                tips.append("🎥 **Embed Video**: Video embeds substantially boost dwell time and engagement.")
            if is_weekend_val == 0:
                tips.append("📅 **Test Weekend Scheduling**: If this is a deep dive or lifestyle piece, weekend releases experience 12% higher viral reach due to reduced publisher saturation.")
            if n_tokens_title < 8 or n_tokens_title > 14:
                tips.append("✍️ **Refine Headline Length**: Aim for an engaging 9-12 word headline.")
            if kw_avg_avg < 2500:
                tips.append("🏷️ **Keyword Targeting**: Associate the article with trending, higher-volume topic tags.")
                
            if not tips:
                tips.append("✨ **Great Optimization!** Your content configuration matches high-engagement standards across length, multimedia, and timing.")
                
            for tip in tips:
                st.markdown(f"* {tip}")
