"""
Academic Performance Analytics Platform
A modern analytics tool for student success prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path

# Page config - Dark modern theme
st.set_page_config(
    page_title="EduPredict Analytics",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern dark CSS theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    .card {
        background: linear-gradient(145deg, #1e1e2e, #2d2d44);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 1rem;
    }
    
    .stat-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stat-label {
        color: #888;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .prediction-result {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        border: 2px solid;
    }
    
    .grade-display {
        font-size: 4rem;
        font-weight: 700;
    }
    
    .tab-content {
        padding: 1rem 0;
    }
    
    .insight-box {
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 10px 10px 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(102, 126, 234, 0.1);
        border-radius: 10px;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_ml_components():
    """Load ML model and scaler with caching."""
    model_path = Path("models/best_model.joblib")
    scaler_path = Path("models/scaler.joblib")
    
    if model_path.exists() and scaler_path.exists():
        model_data = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        model = model_data['model'] if isinstance(model_data, dict) else model_data
        return model, scaler
    return None, None


@st.cache_data
def load_dataset():
    """Load student dataset with caching."""
    paths = [Path("data/student-mat.csv"), Path("data/raw/student-mat.csv")]
    for p in paths:
        if p.exists():
            return pd.read_csv(p, sep=';')
    return None


def classify_performance(grade):
    """Classify student performance tier."""
    if grade >= 16: return ("Outstanding", "🌟", "#10b981")
    if grade >= 14: return ("Proficient", "✨", "#3b82f6")
    if grade >= 12: return ("Developing", "📈", "#f59e0b")
    if grade >= 10: return ("Approaching", "📊", "#f97316")
    return ("Needs Support", "🎯", "#ef4444")


def engineer_prediction_features(raw_input):
    """Transform raw input into model-ready features."""
    df = pd.DataFrame([raw_input])
    
    # Binary encodings
    binary_maps = {
        'school': {'GP': 0, 'MS': 1}, 'sex': {'F': 0, 'M': 1},
        'address': {'U': 0, 'R': 1}, 'famsize': {'LE3': 0, 'GT3': 1},
        'Pstatus': {'T': 0, 'A': 1}
    }
    yes_no = ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
    
    for col, mapping in binary_maps.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    
    for col in yes_no:
        if col in df.columns:
            df[col] = df[col].map({'no': 0, 'yes': 1})
    
    # One-hot encoding
    mjob, fjob = raw_input.get('Mjob', 'other'), raw_input.get('Fjob', 'other')
    reason, guardian = raw_input.get('reason', 'other'), raw_input.get('guardian', 'mother')
    
    for job_type in ['health', 'other', 'services', 'teacher']:
        df[f'Mjob_{job_type}'] = 1 if mjob == job_type else 0
        df[f'Fjob_{job_type}'] = 1 if fjob == job_type else 0
    
    for r in ['home', 'other', 'reputation']:
        df[f'reason_{r}'] = 1 if reason == r else 0
    
    df['guardian_mother'] = 1 if guardian == 'mother' else 0
    df['guardian_other'] = 1 if guardian == 'other' else 0
    
    df = df.drop(columns=['Mjob', 'Fjob', 'reason', 'guardian'], errors='ignore')
    
    # Derived features
    medu, fedu = raw_input.get('Medu', 2), raw_input.get('Fedu', 2)
    studytime = raw_input.get('studytime', 2)
    freetime, goout = raw_input.get('freetime', 3), raw_input.get('goout', 3)
    dalc, walc = raw_input.get('Dalc', 1), raw_input.get('Walc', 1)
    g1, g2 = raw_input.get('G1', 10), raw_input.get('G2', 10)
    absences, failures = raw_input.get('absences', 5), raw_input.get('failures', 0)
    
    df['parent_edu_avg'] = (medu + fedu) / 2
    df['parent_edu_max'] = max(medu, fedu)
    df['parent_edu_diff'] = abs(medu - fedu)
    df['study_leisure_ratio'] = studytime / (freetime + goout + 0.1)
    df['study_leisure_diff'] = studytime - (freetime + goout) / 2
    df['social_engagement'] = (freetime + goout) / 2
    df['total_alcohol'] = dalc + walc
    df['alcohol_weekly_avg'] = (dalc * 5 + walc * 2) / 7
    
    schoolsup_val = 1 if raw_input.get('schoolsup') == 'yes' else 0
    famsup_val = 1 if raw_input.get('famsup') == 'yes' else 0
    paid_val = 1 if raw_input.get('paid') == 'yes' else 0
    df['support_score'] = (schoolsup_val + famsup_val + paid_val) / 3
    
    df['grade_progression'] = g2 - g1
    df['grade_avg_g1g2'] = (g1 + g2) / 2
    df['grade_trend'] = g2 - g1
    df['age_above_avg'] = 1 if raw_input.get('age', 17) > 17 else 0
    df['absence_category'] = 0 if absences < 5 else (1 if absences < 15 else 2)
    df['log_absences'] = np.log1p(absences)
    
    # Interactions
    df['studytime_x_parent_edu'] = studytime * df['parent_edu_avg'].values[0]
    df['failures_x_absences'] = failures * absences
    internet_val = 1 if raw_input.get('internet') == 'yes' else 0
    higher_val = 1 if raw_input.get('higher') == 'yes' else 0
    df['internet_x_higher'] = internet_val * higher_val
    df['famsup_x_studytime'] = famsup_val * studytime
    df['alcohol_x_studytime'] = df['alcohol_weekly_avg'].values[0] * studytime
    
    # Risk flags
    df['high_absence_risk'] = 1 if absences > 10 else 0
    df['has_failures'] = 1 if failures > 0 else 0
    df['multiple_failures'] = 1 if failures > 1 else 0
    df['low_study_risk'] = 1 if studytime <= 1 else 0
    df['high_alcohol_risk'] = 1 if (dalc >= 3 or walc >= 4) else 0
    df['combined_risk_score'] = sum([
        df['high_absence_risk'].values[0], df['has_failures'].values[0],
        df['low_study_risk'].values[0], df['high_alcohol_risk'].values[0]
    ])
    
    # Reorder to match training
    feature_order = [
        'school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
        'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid',
        'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel',
        'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2',
        'Mjob_health', 'Mjob_other', 'Mjob_services', 'Mjob_teacher',
        'Fjob_health', 'Fjob_other', 'Fjob_services', 'Fjob_teacher',
        'reason_home', 'reason_other', 'reason_reputation',
        'guardian_mother', 'guardian_other', 'parent_edu_avg', 'parent_edu_max',
        'parent_edu_diff', 'study_leisure_ratio', 'study_leisure_diff',
        'social_engagement', 'total_alcohol', 'alcohol_weekly_avg', 'support_score',
        'grade_progression', 'grade_avg_g1g2', 'grade_trend', 'age_above_avg',
        'absence_category', 'log_absences', 'studytime_x_parent_edu',
        'failures_x_absences', 'internet_x_higher', 'famsup_x_studytime',
        'alcohol_x_studytime', 'high_absence_risk', 'has_failures',
        'multiple_failures', 'low_study_risk', 'high_alcohol_risk', 'combined_risk_score'
    ]
    
    return df.reindex(columns=feature_order)


def run_prediction(model, scaler, features_df):
    """Execute prediction pipeline."""
    try:
        scaled = scaler.transform(features_df)
        pred = model.predict(scaled)[0]
        return max(0, min(20, pred))
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None


def render_hero():
    """Render hero section."""
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">📚 EduPredict Analytics</div>
        <div class="hero-subtitle">AI-Powered Academic Performance Prediction Platform</div>
    </div>
    """, unsafe_allow_html=True)


def render_overview(df):
    """Render dashboard overview."""
    if df is None:
        st.warning("Dataset not available")
        return
    
    st.markdown("### 📊 Dataset Insights")
    
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown(f"""
        <div class="card">
            <div class="stat-label">Total Students</div>
            <div class="stat-value">{len(df)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with c2:
        avg = df['G3'].mean()
        st.markdown(f"""
        <div class="card">
            <div class="stat-label">Average Score</div>
            <div class="stat-value">{avg:.1f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with c3:
        pass_rate = (df['G3'] >= 10).mean() * 100
        st.markdown(f"""
        <div class="card">
            <div class="stat-label">Success Rate</div>
            <div class="stat-value">{pass_rate:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with c4:
        top_students = (df['G3'] >= 16).sum()
        st.markdown(f"""
        <div class="card">
            <div class="stat-label">Top Performers</div>
            <div class="stat-value">{top_students}</div>
        </div>
        """, unsafe_allow_html=True)


def render_analytics(df):
    """Render analytics section."""
    if df is None:
        return
    
    st.markdown("### 🔬 Performance Analytics")
    
    tab1, tab2, tab3 = st.tabs(["📈 Distributions", "🔗 Relationships", "📋 Raw Data"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x='G3', nbins=21, 
                             color_discrete_sequence=['#667eea'],
                             title="Final Grade Distribution")
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#888',
                xaxis_title="Grade",
                yaxis_title="Count"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Performance tier breakdown
            tiers = pd.cut(df['G3'], bins=[-1, 9, 11, 13, 15, 20], 
                          labels=['Needs Support', 'Approaching', 'Developing', 'Proficient', 'Outstanding'])
            tier_counts = tiers.value_counts().sort_index()
            
            fig = px.pie(values=tier_counts.values, names=tier_counts.index,
                        color_discrete_sequence=['#ef4444', '#f97316', '#f59e0b', '#3b82f6', '#10b981'],
                        title="Performance Tier Breakdown")
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#888'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(df, x='G1', y='G3', trendline='ols',
                           color_discrete_sequence=['#764ba2'],
                           title="Early vs Final Performance")
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#888'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(df, x='studytime', y='G3',
                        color_discrete_sequence=['#667eea'],
                        title="Study Time Impact")
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#888'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation insights
        st.markdown("#### 🎯 Key Correlations")
        numeric = df.select_dtypes(include=[np.number])
        corrs = numeric.corr()['G3'].drop('G3').sort_values(key=abs, ascending=False).head(6)
        
        cols = st.columns(6)
        for i, (feat, val) in enumerate(corrs.items()):
            color = "#10b981" if val > 0 else "#ef4444"
            cols[i].markdown(f"""
            <div style="text-align:center; padding:0.5rem;">
                <div style="color:{color}; font-size:1.5rem; font-weight:600;">{val:.2f}</div>
                <div style="color:#888; font-size:0.7rem;">{feat}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        st.dataframe(df, use_container_width=True, height=400)


def render_predictor(model, scaler):
    """Render prediction interface."""
    st.markdown("### 🎯 Performance Predictor")
    
    if model is None or scaler is None:
        st.error("⚠️ Model not loaded. Please ensure model files exist in /models directory.")
        return
    
    st.markdown("""
    <div class="insight-box">
        💡 Enter student information below to predict their expected final grade.
        The model analyzes 67 features including academic history, family background, and lifestyle factors.
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        st.markdown("#### 👤 Student Profile")
        
        row1 = st.columns(4)
        with row1[0]:
            age = st.number_input("Age", 15, 22, 17)
        with row1[1]:
            sex = st.radio("Gender", ["M", "F"], horizontal=True)
        with row1[2]:
            school = st.radio("School", ["GP", "MS"], horizontal=True)
        with row1[3]:
            address = st.radio("Location", ["U", "R"], horizontal=True, help="Urban/Rural")
        
        st.markdown("#### 👨‍👩‍👧 Family Background")
        
        row2 = st.columns(4)
        with row2[0]:
            Medu = st.select_slider("Mother's Education", options=[0,1,2,3,4], value=2)
        with row2[1]:
            Fedu = st.select_slider("Father's Education", options=[0,1,2,3,4], value=2)
        with row2[2]:
            famsize = st.radio("Family Size", ["LE3", "GT3"], horizontal=True)
        with row2[3]:
            Pstatus = st.radio("Parents", ["T", "A"], horizontal=True, help="Together/Apart")
        
        row3 = st.columns(4)
        with row3[0]:
            Mjob = st.selectbox("Mother's Occupation", ["teacher", "health", "services", "at_home", "other"])
        with row3[1]:
            Fjob = st.selectbox("Father's Occupation", ["teacher", "health", "services", "at_home", "other"])
        with row3[2]:
            guardian = st.selectbox("Primary Guardian", ["mother", "father", "other"])
        with row3[3]:
            reason = st.selectbox("School Choice Reason", ["home", "reputation", "course", "other"])
        
        st.markdown("#### 📚 Academic Profile")
        
        row4 = st.columns(4)
        with row4[0]:
            G1 = st.slider("Period 1 Grade", 0, 20, 12)
        with row4[1]:
            G2 = st.slider("Period 2 Grade", 0, 20, 12)
        with row4[2]:
            studytime = st.select_slider("Weekly Study (hrs)", options=["<2", "2-5", "5-10", ">10"], value="2-5")
            studytime = {"<2": 1, "2-5": 2, "5-10": 3, ">10": 4}[studytime]
        with row4[3]:
            failures = st.number_input("Past Failures", 0, 4, 0)
        
        row5 = st.columns(4)
        with row5[0]:
            absences = st.number_input("Absences", 0, 100, 4)
        with row5[1]:
            traveltime = st.select_slider("Commute", options=["<15m", "15-30m", "30-60m", ">1h"], value="15-30m")
            traveltime = {"<15m": 1, "15-30m": 2, "30-60m": 3, ">1h": 4}[traveltime]
        with row5[2]:
            higher = st.radio("Wants Higher Ed?", ["yes", "no"], horizontal=True)
        with row5[3]:
            internet = st.radio("Has Internet?", ["yes", "no"], horizontal=True)
        
        st.markdown("#### 🌟 Support & Lifestyle")
        
        row6 = st.columns(6)
        with row6[0]:
            schoolsup = st.checkbox("School Support")
            schoolsup = "yes" if schoolsup else "no"
        with row6[1]:
            famsup = st.checkbox("Family Support")
            famsup = "yes" if famsup else "no"
        with row6[2]:
            paid = st.checkbox("Extra Classes")
            paid = "yes" if paid else "no"
        with row6[3]:
            activities = st.checkbox("Activities")
            activities = "yes" if activities else "no"
        with row6[4]:
            nursery = st.checkbox("Nursery")
            nursery = "yes" if nursery else "no"
        with row6[5]:
            romantic = st.checkbox("Dating")
            romantic = "yes" if romantic else "no"
        
        row7 = st.columns(5)
        with row7[0]:
            famrel = st.slider("Family Relations", 1, 5, 4)
        with row7[1]:
            freetime = st.slider("Free Time", 1, 5, 3)
        with row7[2]:
            goout = st.slider("Social Activity", 1, 5, 3)
        with row7[3]:
            Dalc = st.slider("Weekday Alcohol", 1, 5, 1)
        with row7[4]:
            Walc = st.slider("Weekend Alcohol", 1, 5, 1)
        
        health = st.slider("Health Status", 1, 5, 4)
        
        submitted = st.form_submit_button("🔮 Generate Prediction", use_container_width=True)
    
    if submitted:
        input_data = {
            'school': school, 'sex': sex, 'age': age, 'address': address,
            'famsize': famsize, 'Pstatus': Pstatus, 'Medu': Medu, 'Fedu': Fedu,
            'Mjob': Mjob, 'Fjob': Fjob, 'reason': reason, 'guardian': guardian,
            'traveltime': traveltime, 'studytime': studytime, 'failures': failures,
            'schoolsup': schoolsup, 'famsup': famsup, 'paid': paid,
            'activities': activities, 'nursery': nursery, 'higher': higher,
            'internet': internet, 'romantic': romantic, 'famrel': famrel,
            'freetime': freetime, 'goout': goout, 'Dalc': Dalc, 'Walc': Walc,
            'health': health, 'absences': absences, 'G1': G1, 'G2': G2
        }
        
        features = engineer_prediction_features(input_data)
        prediction = run_prediction(model, scaler, features)
        
        if prediction is not None:
            tier, icon, color = classify_performance(prediction)
            
            st.markdown("---")
            
            res_cols = st.columns([1, 2, 1])
            
            with res_cols[1]:
                st.markdown(f"""
                <div class="prediction-result" style="border-color: {color};">
                    <div style="font-size: 1.2rem; color: #888; margin-bottom: 0.5rem;">Predicted Final Grade</div>
                    <div class="grade-display" style="color: {color};">{prediction:.1f}</div>
                    <div style="font-size: 1.5rem; margin-top: 0.5rem;">{icon} {tier}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Insights based on prediction
            st.markdown("#### 📋 Analysis")
            
            insight_cols = st.columns(3)
            
            with insight_cols[0]:
                trend = "📈 Improving" if G2 > G1 else ("📉 Declining" if G2 < G1 else "➡️ Stable")
                st.info(f"**Grade Trend:** {trend}\n\nPeriod 1: {G1} → Period 2: {G2}")
            
            with insight_cols[1]:
                risk_factors = []
                if failures > 0: risk_factors.append("Past failures")
                if absences > 10: risk_factors.append("High absences")
                if studytime <= 1: risk_factors.append("Low study time")
                if Dalc >= 3 or Walc >= 4: risk_factors.append("Alcohol consumption")
                
                if risk_factors:
                    st.warning(f"**Risk Factors:**\n\n" + "\n".join([f"• {r}" for r in risk_factors]))
                else:
                    st.success("**Risk Assessment:**\n\nNo significant risk factors detected")
            
            with insight_cols[2]:
                strengths = []
                if higher == "yes": strengths.append("Higher education motivation")
                if famsup == "yes" or schoolsup == "yes": strengths.append("Support system")
                if studytime >= 3: strengths.append("Good study habits")
                if (G1 + G2) / 2 >= 12: strengths.append("Strong academic base")
                
                if strengths:
                    st.success(f"**Strengths:**\n\n" + "\n".join([f"• {s}" for s in strengths]))
                else:
                    st.info("**Strengths:**\n\nContinue building positive habits")


def main():
    """Main application."""
    model, scaler = load_ml_components()
    df = load_dataset()
    
    render_hero()
    
    # Horizontal navigation
    nav = st.radio(
        "Navigation",
        ["🏠 Overview", "📊 Analytics", "🎯 Predictor"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    if nav == "🏠 Overview":
        render_overview(df)
        if df is not None:
            st.markdown("### 📈 Quick Trends")
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(df, x='G3', nbins=21, color_discrete_sequence=['#667eea'])
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                                font_color='#888', title="Grade Distribution", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.scatter(df, x='G2', y='G3', trendline='ols', color_discrete_sequence=['#764ba2'])
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                                font_color='#888', title="Period 2 → Final Correlation")
                st.plotly_chart(fig, use_container_width=True)
    
    elif nav == "📊 Analytics":
        render_analytics(df)
    
    elif nav == "🎯 Predictor":
        render_predictor(model, scaler)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        EduPredict Analytics Platform • Built with Machine Learning • 2026
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
