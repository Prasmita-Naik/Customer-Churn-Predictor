import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Churn Predictor", page_icon="📱")

st.title("📱 Customer Churn Predictor")
st.write("Predict whether a customer is likely to leave the service.")

@st.cache_data
def load_data():
    return pd.read_csv("Churn_Dateset_new.csv")

ch = load_data()

# Remove ID
ch = ch.drop("Anonymous Customer ID", axis=1)

# Missing values
for col in ch.columns:
    if col == "Churn":
        ch = ch.dropna(subset=["Churn"])
    elif ch[col].isnull().sum() > 0:
        if col in ["Complains", "Seconds of Use",
                   "Frequency of SMS", "Tariff Plan", "Status"]:
            ch[col] = ch[col].fillna(ch[col].mode()[0])
        else:
            ch[col] = ch[col].fillna(ch[col].median())

X = ch.drop("Churn", axis=1)
y = ch["Churn"]

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=123)

model = RandomForestClassifier(random_state=123)
model.fit(X_train, y_train)

st.subheader("👤 Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    call_failure = st.number_input("Call Failure", 0.0, 50.0, 5.0)
    complains = st.selectbox("Complains", [0, 1])
    subscription = st.number_input("Subscription Length", 1.0, 100.0, 35.0)
    charge = st.number_input("Charge Amount", 0.0, 20.0, 5.0)
    seconds = st.number_input("Seconds of Use", 0.0, 10000.0, 2000.0)
    frequency = st.number_input("Frequency of Use", 0.0, 200.0, 50.0)

with col2:
    sms = st.number_input("Frequency of SMS", 0.0, 1000.0, 20.0)
    numbers = st.number_input("Distinct Called Numbers", 0.0, 100.0, 20.0)
    age = st.number_input("Age Group", 1.0, 10.0, 3.0)
    tariff = st.number_input("Tariff Plan", 1.0, 10.0, 1.0)
    status = st.number_input("Status", 1.0, 10.0, 1.0)
    value = st.number_input("Customer Value", 0.0, 1000.0, 100.0)

if st.button("🔮 Predict Churn"):

    customer = pd.DataFrame([[
        call_failure,
        complains,
        subscription,
        charge,
        seconds,
        frequency,
        sms,
        numbers,
        age,
        tariff,
        status,
        value
    ]], columns=X.columns)

    prediction = model.predict(customer)[0]
    probability = model.predict_proba(customer)[0][1]

    st.divider()

    if prediction == 1:
        st.error("🔴 Customer is likely to CHURN")
    else:
        st.success("🟢 Customer is likely to STAY")

    st.metric("Churn Probability", f"{probability:.1%}")