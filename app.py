import pickle
import streamlit as st
import pandas as pd

# =====================
# CONFIG
# =====================
st.set_page_config(
    page_title="Vehicle Insurance Prediction",
    page_icon="🚗",
    layout="wide"
)

# =====================
# LOAD MODEL
# =====================
@st.cache_resource
def load_model():
    with open("xgb_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# =====================
# HEADER
# =====================
st.markdown(
    """
    <h1 style="text-align:center;">🚗 Vehicle Insurance Cross-Sell Prediction</h1>
    <p style="text-align:center; font-size:18px;">
    XGBoost model deployment with Streamlit
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# =====================
# SIDEBAR
# =====================
st.sidebar.title("⚙️ Prediction Mode")
mode = st.sidebar.radio(
    "Choose input method",
    ["Manual Input", "Upload CSV / Excel"]
)

# =====================
# ENCODING FUNCTIONS
# =====================
def encode_inputs(
    Gender,
    Age,
    Driving_License,
    Region_Code,
    Previously_Insured,
    Vehicle_Age,
    Vehicle_Damage,
    Annual_Premium,
    Policy_Sales_Channel,
    Vintage,
    Annual_Premium_Adjusted
):
    # Encode categorical variables
    Gender = 1 if Gender == "Male" else 0

    vehicle_age_map = {
        "< 1 Year": 0,
        "1-2 Year": 1,
        "> 2 Years": 2
    }
    Vehicle_Age = vehicle_age_map[Vehicle_Age]

    Vehicle_Damage = 1 if Vehicle_Damage == "Yes" else 0

    return pd.DataFrame([{
        "Gender": Gender,
        "Age": Age,
        "Driving_License": Driving_License,
        "Region_Code": Region_Code,
        "Previously_Insured": Previously_Insured,
        "Vehicle_Age": Vehicle_Age,
        "Vehicle_Damage": Vehicle_Damage,
        "Annual_Premium": Annual_Premium,
        "Policy_Sales_Channel": Policy_Sales_Channel,
        "Vintage": Vintage,
        "Annual_Premium_Adjusted": Annual_Premium_Adjusted
    }])

# =====================
# MANUAL INPUT MODE
# =====================
if mode == "Manual Input":
    st.subheader("📝 Manual Input")

    col1, col2, col3 = st.columns(3)

    with col1:
        Gender = st.selectbox("Gender", ["Male", "Female"])
        Age = st.number_input("Age", 18, 100, 30)
        Driving_License = st.selectbox("Driving License", [0, 1])

    with col2:
        Region_Code = st.number_input("Region Code", 0, 100, 10)
        Previously_Insured = st.selectbox("Previously Insured", [0, 1])
        Vehicle_Age = st.selectbox(
            "Vehicle Age",
            ["< 1 Year", "1-2 Year", "> 2 Years"]
        )

    with col3:
        Vehicle_Damage = st.selectbox("Vehicle Damage", ["Yes", "No"])
        Annual_Premium = st.number_input("Annual Premium", 0.0, 100000.0, 30000.0)
        Policy_Sales_Channel = st.number_input("Policy Sales Channel", 0, 200, 30)
        Vintage = st.number_input("Vintage", 0, 500, 150)
        Annual_Premium_Adjusted = st.number_input(
            "Annual Premium Adjusted",
            0.0,
            100000.0,
            30000.0
        )

    if st.button("🔮 Predict"):
        sample = encode_inputs(
            Gender,
            Age,
            Driving_License,
            Region_Code,
            Previously_Insured,
            Vehicle_Age,
            Vehicle_Damage,
            Annual_Premium,
            Policy_Sales_Channel,
            Vintage,
            Annual_Premium_Adjusted
        )

        pred = model.predict(sample)[0]
        prob = model.predict_proba(sample)[0][1]

        st.success("Prediction completed")
        st.metric("Prediction (0 = No, 1 = Yes)", int(pred))
        st.metric("Probability", round(prob, 3))

# =====================
# FILE UPLOAD MODE
# =====================
else:
    st.subheader("📂 Upload CSV / Excel")

    uploaded_file = st.file_uploader(
        "Upload file",
        type=["csv", "xlsx"]
    )

    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.write("📄 Uploaded data preview:")
        st.dataframe(df.head())

        required_cols = [
            "Gender",
            "Age",
            "Driving_License",
            "Region_Code",
            "Previously_Insured",
            "Vehicle_Age",
            "Vehicle_Damage",
            "Annual_Premium",
            "Policy_Sales_Channel",
            "Vintage",
            "Annual_Premium_Adjusted"
        ]

        if not all(col in df.columns for col in required_cols):
            st.error(
                "Dataset must contain columns:\n"
                + ", ".join(required_cols)
            )
        else:
            encoded_rows = []

            for _, row in df.iterrows():
                sample = encode_inputs(
                    row["Gender"],
                    row["Age"],
                    row["Driving_License"],
                    row["Region_Code"],
                    row["Previously_Insured"],
                    row["Vehicle_Age"],
                    row["Vehicle_Damage"],
                    row["Annual_Premium"],
                    row["Policy_Sales_Channel"],
                    row["Vintage"],
                    row["Annual_Premium_Adjusted"]
                )
                encoded_rows.append(sample)

            X_all = pd.concat(encoded_rows, ignore_index=True)

            preds = model.predict(X_all)
            probs = model.predict_proba(X_all)[:, 1]

            df["prediction"] = preds
            df["probability"] = probs.round(3)

            st.success("Prediction completed for uploaded file")
            st.dataframe(df)

            st.download_button(
                "⬇️ Download result",
                df.to_csv(index=False),
                "prediction_result.csv",
                "text/csv"
            )

# =====================
# FOOTER
# =====================
st.markdown(
    """
    <hr>
    <p style="text-align:center; font-size:14px;">
    Big Data Midterm Project • Vehicle Insurance Prediction
    </p>
    """,
    unsafe_allow_html=True
)
