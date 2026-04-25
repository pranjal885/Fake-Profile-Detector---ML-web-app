import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Fake Profile Detector", page_icon="🕵️", layout="wide")

# Custom CSS for UI Enhancement
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }

    /* Styling the header */
    .title-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0px 8px 32px rgba(59, 130, 246, 0.4);
        margin-bottom: 3rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .title-header h1 {
        margin: 0;
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
        color: white;
    }
    .title-header p {
        margin-top: 10px;
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 300;
    }

    /* Column/Section Card separation for the first row of columns */
    div[data-testid="stHorizontalBlock"] {
        gap: 1.5rem;
    }

    div[data-testid="stHorizontalBlock"]:first-of-type div[data-testid="column"] {
        background: rgba(30, 58, 138, 0.2);
        border: 1px solid rgba(59, 130, 246, 0.3);
        padding: 1.5rem 1.5rem 2rem 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease-in-out;
    }

    div[data-testid="stHorizontalBlock"]:first-of-type div[data-testid="column"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(59, 130, 246, 0.3);
        border-color: rgba(59, 130, 246, 0.7);
        background: rgba(30, 58, 138, 0.3);
    }

    /* Styling the Predict Button to match blue theme */
    .stButton>button {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        padding: 1rem 2rem !important;
        font-weight: 800 !important;
        font-size: 1.3rem !important;
        letter-spacing: 1px;
        width: 100% !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4) !important;
    }
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 10px 25px rgba(37, 99, 235, 0.6) !important;
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
    }

    /* Simple styling for the result cards */
    .result-card {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------
# Header Section
# --------------------------------
st.markdown("""
<div class="title-header">
    <h1>🕵️ Fake Profile Detection System</h1>
    <p>Leveraging Machine Learning to identify synthetic, spam, and bot accounts</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------
# Load trained model and scaler
# --------------------------------
@st.cache_resource
def load_models():
    model = pickle.load(open("fake_profile_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, scaler

model, scaler = load_models()

# --------------------------------
# Dataset & Model Info
# --------------------------------
with st.expander("📊 Dataset & Model Information", expanded=False):
    st.markdown("""
    **Dataset Size:** 1,000 profiles  
    **Classes:** Real, Fake  

    **Features Used:**
    - Followers, Following
    - Posts & Posts per Day
    - Engagement & Similarity Scores
    - Account Age
    - Spam & Generic Comments
    - Suspicious Activity Indicators  

    **Model Used:** Logistic Regression
    """)

# --------------------------------
# Input Validation Function
# --------------------------------
def validate_inputs(followers, following, posts, posts_per, account_age):
    if followers < 0 or following < 0 or posts < 0:
        return False, "Numeric values cannot be negative."
    if posts == 0 and posts_per > 0:
        return False, "Posts per day cannot be greater than 0 when total posts are 0."
    if account_age == 0 and posts > 0:
        return False, "Account age cannot be 0 if posts exist."
    return True, ""

# --------------------------------
# User Input Section (Dashboard Layout)
# --------------------------------
st.markdown("### 📋 Enter Profile Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 🔹 Basic Info")
    platform = st.selectbox("Platform", ["Twitter", "Instagram", "Facebook"])
    username = st.radio("Suspicious Username?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    has_profile = st.radio("Has Profile Photo?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    verified = st.radio("Verified Account?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    bio_length = st.number_input("Bio Length (chars)", min_value=0, value=50)
    account_a = st.number_input("Account Age (days)", min_value=0, value=365)

with col2:
    st.markdown("#### 📈 Network & Activity")
    followers = st.number_input("Followers", min_value=0, value=100)
    following = st.number_input("Following", min_value=0, value=100)
    follower_f = st.number_input("Follower / Following Ratio", format="%.4f", value=1.0)
    follow_un = st.number_input("Unusual Following Count", min_value=0, value=0)
    suspicious = st.radio("Suspicious Activity?", [0, 1], format_func=lambda x: "Yes" if x else "No")

with col3:
    st.markdown("#### 📝 Content Metrics")
    posts = st.number_input("Total Posts", min_value=0, value=10)
    posts_per = st.number_input("Posts per Day", format="%.4f", value=0.5)
    caption_si = st.number_input("Caption Similarity Score", format="%.4f", value=0.1)
    content_si = st.number_input("Content Similarity Score", format="%.4f", value=0.1)
    spam_com = st.number_input("Spam Comments Count", min_value=0, value=0)
    generic_cc = st.number_input("Generic Comments Count", min_value=0, value=0)

platform_map = {
    "Twitter": 0,
    "Instagram": 1,
    "Facebook": 2
}

st.markdown("---")

# --------------------------------
# Prediction Button & Results
# --------------------------------
_, center_col, _ = st.columns([1, 2, 1])

with center_col:
    if st.button("🔎 Detect Profile"):
        is_valid, error_msg = validate_inputs(
            followers, following, posts, posts_per, account_a
        )

        if not is_valid:
            st.error(f"⚠️ {error_msg}")
        else:
            with st.spinner("Analyzing profile metrics..."):
                input_data = np.array([[ 
                    platform_map[platform],
                    has_profile,
                    bio_length,
                    username,
                    followers,
                    following,
                    follower_f,
                    account_a,
                    posts,
                    posts_per,
                    caption_si,
                    content_si,
                    follow_un,
                    spam_com,
                    generic_cc,
                    suspicious,
                    verified
                ]])

                input_scaled = scaler.transform(input_data)
                prediction = model.predict(input_scaled)[0]
                probability = model.predict_proba(input_scaled)[0][prediction]

            # Results Display
            st.markdown("<br>", unsafe_allow_html=True)
            if prediction == 1:
                st.error(f"### ❌ Fake Profile Detected\n**Confidence Score:** `{probability*100:.2f}%`")
                st.info("This account exhibits behaviors commonly associated with bots, spam, or fake accounts.")
            else:
                st.success(f"### ✅ Real Profile Detected\n**Confidence Score:** `{probability*100:.2f}%`")
                st.info("The metrics suggest that this is a genuine user account.")
