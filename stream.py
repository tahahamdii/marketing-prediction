import streamlit as st
import requests

# Financial-themed styling
st.set_page_config(
    page_title="Predict Average Balance",
    page_icon="💳",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    body {
        background-color: #f4f6f9;
    }
    .main {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar for input fields
st.sidebar.title("Input Details")
st.sidebar.write("Enter the details below:")

# Input fields in the sidebar
mailer_type = st.sidebar.selectbox("Mailer Type", ["Type A", "Type B", "Type C"])
income_level = st.sidebar.number_input("Income Level", min_value=0, step=1000)
bank_accounts_open = st.sidebar.number_input("# Bank Accounts Open", min_value=0, step=1)
overdraft_protection = st.sidebar.selectbox("Overdraft Protection", ["Yes", "No"])
credit_rating = st.sidebar.selectbox("Credit Rating", ["High", "Medium", "Low", "Poor"])
credit_cards_held = st.sidebar.number_input("# Credit Cards Held", min_value=0, step=1)
homes_owned = st.sidebar.number_input("# Homes Owned", min_value=0, step=1)
household_size = st.sidebar.number_input("Household Size", min_value=1, step=1)
own_your_home = st.sidebar.selectbox("Own Your Home", ["Yes", "No"])

# Main page content
st.title("💳 Predict Average Balance")
st.write("This application predicts the average balance based on the details you provide. Use the sidebar to enter your details and click **Predict**.")

# Submit button
if st.sidebar.button("Predict"):
    # Prepare the data payload
    data = {
        "Mailer Type": mailer_type,
        "Income Level": income_level,
        "# Bank Accounts Open": bank_accounts_open,
        "Overdraft Protection": overdraft_protection,
        "Credit Rating": credit_rating,
        "# Credit Cards Held": credit_cards_held,
        "# Homes Owned": homes_owned,
        "Household Size": household_size,
        "Own Your Home": own_your_home,
    }

    # Flask API endpoint
    flask_url = "http://127.0.0.1:5000/predict"  # Change this to your Flask endpoint

    # Send request to the Flask app
    try:
        response = requests.post(flask_url, json=data)
        if response.status_code == 200:
            result = response.json()
            st.success(f"The predicted average balance is: ${result['average_balance']:.2f}")
        else:
            st.error("Error in prediction. Please check your input or the server.")
    except Exception as e:
        st.error(f"Failed to connect to the prediction server. Error: {e}")
