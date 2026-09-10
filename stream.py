import joblib
import pandas as pd
import streamlit as st

# ==========================================
# 1. Page Configuration
# ==========================================
st.set_page_config(
    page_title='Startup Failure Predictor', page_icon='🚀', layout='wide'
)

st.title('🚀 Startup Failure Predictor')
st.markdown(
    'Select a machine learning model and enter founder and startup metrics to get instant predictions.'
)

# ==========================================
# 2. Load Selected Model
# ==========================================
model_files = {
    'Logistic Regression': 'logistic_regression.pkl',
    'Adaboost': 'adaboost.pkl',
    'XGBoost': 'xgboost.pkl',
    'MLP (Neural Network)': 'mlp_neural_network.pkl',
}

st.sidebar.header('⚙️ Model Selection')
model_choice = st.sidebar.selectbox('Choose a Model:', list(model_files.keys()))


@st.cache_resource
def load_model(file_name):
    return joblib.load(file_name)


try:
    selected_model = load_model(model_files[model_choice])
    st.sidebar.success(f'Activated: {model_choice}')
except Exception as e:
    st.sidebar.error(f'Error loading {model_files[model_choice]}: {e}')
    st.stop()

# ==========================================
# 3. User Input Features
# ==========================================
st.subheader('📝 Founder & Startup Metrics')

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('*Founder Profile*')
    founder_type = st.selectbox(
        'Founder Type',
        [
            'Solo Hustler',
            'Technical Builder',
            'Serial Entrepreneur',
            'Growth Obsessed Founder',
            'Visionary CEO',
            'Burned-Out Operator',
            'Chaotic Innovator',
            'Calm Operator',
        ],
    )
    founder_age = st.number_input('Founder Age', min_value=18, max_value=80, value=34)
    founder_exp = st.number_input('Experience (Years)', min_value=0, max_value=40, value=3)
    work_hours = st.slider('Weekly Work Hours', 20.0, 110.0, 63.0)
    sleep_hours = st.slider('Daily Sleep Hours', 2.0, 10.0, 5.7)
    exercise_days = st.slider('Exercise Days (Weekly)', 0.0, 7.0, 3.7)
    vacation_days = st.number_input(
        'Vacation Days Taken (Yearly)', min_value=0.0, max_value=40.0, value=9.0
    )

with col2:
    st.markdown('*Startup & Funding Details*')
    industry = st.selectbox(
        'Industry',
        [
            'AI',
            'SaaS',
            'E-commerce',
            'FinTech',
            'HealthTech',
            'Cybersecurity',
            'EdTech',
            'Gaming',
            'ClimateTech',
            'Biotech',
        ],
    )
    funding_stage = st.selectbox(
        'Funding Stage',
        ['Seed', 'Pre-Seed', 'Series A', 'Bootstrapped', 'Series B', 'Series C'],
    )
    work_mode = st.selectbox('Work Mode', ['Remote', 'Hybrid', 'Office'])
    team_size = st.number_input('Team Size', min_value=1, max_value=1000, value=15)
    startup_age = st.number_input('Startup Age (Months)', min_value=1, max_value=200, value=72)
    monthly_growth = st.number_input('Monthly Revenue Growth (%)', value=5.5)
    runway = st.number_input('Runway Months Remaining', value=12.0)
    turnover = st.number_input('Employee Turnover (%)', value=35.0)

with col3:
    st.markdown('*Psychological & Environmental Metrics*')
    climate = st.selectbox(
        'Economic Climate',
        ['Stable Economy', 'Funding Winter', 'Bull Market', 'Recession'],
    )
    burnout_level = st.selectbox('Burnout Level', ['Low', 'Moderate', 'Severe'])
    mental_support = st.selectbox('Seeks Mental Health Support?', ['No', 'Yes'])
    founder_burnout_flag = st.selectbox('Founder Burnout Flag', [0, 1])
    stress_score = st.slider('Stress Score (1-10)', 1.0, 10.0, 5.0)
    fatigue_score = st.slider('Decision Fatigue Score (1-10)', 1.0, 10.0, 4.5)
    burnout_score = st.slider('Burnout Score (1-10)', 1.0, 10.0, 3.4)
    pressure_score = st.slider('Investor Pressure Score (1-10)', 1.0, 10.0, 6.0)
    conflict_score = st.slider('Cofounder Conflict Score (0-10)', 0.0, 10.0, 4.0)
    pmf_score = st.slider('Product-Market Fit Score (1-10)', 1.0, 10.0, 5.3)
    wlb_score = st.slider('Work-Life Balance Score (1-10)', 1.0, 10.0, 6.7)

# ==========================================
# 4. Prediction & Output Display
# ==========================================
st.markdown('---')

if st.button('🔮 Predict Startup Failure Risk'):
    user_input_df = pd.DataFrame(
        [
            {
                'Founder_Type': founder_type,
                'Economic_Climate': climate,
                'Founder_Age': founder_age,
                'Founder_Experience_Years': founder_exp,
                'Industry': industry,
                'Funding_Stage': funding_stage,
                'Work_Mode': work_mode,
                'Team_Size': team_size,
                'Startup_Age_Months': startup_age,
                'Weekly_Work_Hours': work_hours,
                'Sleep_Hours': sleep_hours,
                'Exercise_Days_Per_Week': exercise_days,
                'Vacation_Days_Taken': vacation_days,
                'Investor_Pressure_Score': pressure_score,
                'Cofounder_Conflict_Score': conflict_score,
                'Stress_Score': stress_score,
                'Decision_Fatigue_Score': fatigue_score,
                'Burnout_Score': burnout_score,
                'Burnout_Level': burnout_level,
                'Founder_Burnout_Flag': founder_burnout_flag,
                'Monthly_Revenue_Growth_Percent': monthly_growth,
                'Runway_Months_Remaining': runway,
                'Product_Market_Fit_Score': pmf_score,
                'Employee_Turnover_Percent': turnover,
                'Work_Life_Balance_Score': wlb_score,
                'Seeks_Mental_Health_Support': mental_support,
            }
        ]
    )

    try:
        prediction = selected_model.predict(user_input_df)[0]

        st.subheader('📊 Prediction Result:')

        if prediction == 1:
            st.error('⚠️ *Startup Failure: YES* (High Risk of Failure)')
        else:
            st.success('✅ *Startup Failure: NO* (Low Risk / Likely to Succeed)')

    except Exception as err:
        st.error(f'An error occurred during prediction: {err}')