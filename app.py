import streamlit as st
import numpy as np
import pickle

# --------------------------------
# Load trained model and scaler
# --------------------------------
model = pickle.load(open("fake_profile_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Fake Profile Detector", layout="centered")

st.title("🔍 Fake Profile Detection System")
st.write("Enter profile details to check whether the account is **Real or Fake**.")

# --------------------------------
# Dataset & Model Info (ADD-ON 5 ✅)
# --------------------------------
with st.expander("📊 Dataset & Model Information"):
    st.write("""
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
# Input Validation Function (ADD-ON 4 ✅)
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
# User Input Section
# --------------------------------
platform = st.selectbox("Platform", ["Twitter", "Instagram", "Facebook"])
has_profile = st.radio("Has Profile Photo?", [0, 1])
bio_length = st.number_input("Bio Length", min_value=0)
username = st.radio("Suspicious Username?", [0, 1])
followers = st.number_input("Followers", min_value=0)
following = st.number_input("Following", min_value=0)
follower_f = st.number_input("Follower / Following Ratio", format="%.4f")
account_a = st.number_input("Account Age (days)", min_value=0)
posts = st.number_input("Total Posts", min_value=0)
posts_per = st.number_input("Posts per Day", format="%.4f")
caption_si = st.number_input("Caption Similarity Score", format="%.4f")
content_si = st.number_input("Content Similarity Score", format="%.4f")
follow_un = st.number_input("Unusual Following Count", min_value=0)
spam_com = st.number_input("Spam Comments Count", min_value=0)
generic_cc = st.number_input("Generic Comments Count", min_value=0)
suspicious = st.radio("Suspicious Activity?", [0, 1])
verified = st.radio("Verified Account?", [0, 1])

# --------------------------------
# Encode platform
# --------------------------------
platform_map = {
    "Twitter": 0,
    "Instagram": 1,
    "Facebook": 2
}

# --------------------------------
# Prediction Button
# --------------------------------
if st.button("🔎 Detect Profile"):

    # 🔐 Validate inputs BEFORE prediction
    is_valid, error_msg = validate_inputs(
        followers, following, posts, posts_per, account_a
    )

    if not is_valid:
        st.error(error_msg)
        st.stop()

    # Prepare input
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

    st.subheader("Result")

    if prediction == 1:
        st.error(f"❌ Fake Profile Detected (Confidence: {probability*100:.2f}%)")
    else:
        st.success(f"✅ Real Profile Detected (Confidence: {probability*100:.2f}%)")
