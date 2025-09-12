import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Load Data
df = pd.read_csv("diabetes.csv")

st.title('🩺 Diabetes Prediction ')

st.subheader('Training Data Overview')
st.write(df.describe())

# Split
x = df.drop(['Outcome'], axis=1)
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Sidebar sliders for user input
def user_report():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3, key="preg")
    glucose = st.sidebar.slider('Glucose', 0, 200, 120, key="gluc")
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 70, key="bp")
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 100, 20, key="skin")
    insulin = st.sidebar.slider('Insulin', 0, 846, 79, key="insulin")
    bmi = st.sidebar.slider('BMI', 0, 67, 20, key="bmi")
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47, key="dpf")
    age = st.sidebar.slider('Age', 21, 88, 33, key="age")

    user_report = {
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [bp],
        'SkinThickness': [skinthickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [dpf],
        'Age': [age]
    }
    return pd.DataFrame(user_report, index=[0])

# ✅ Create user_data for prediction
user_data = user_report()

# --- Dynamic Visualization (Dataset vs Your Input) ---
st.subheader(" Dataset Average vs Your Data")

avg_data = pd.DataFrame(df.drop("Outcome", axis=1).mean()).T
avg_data["Label"] = "Average"

user_data_vis = user_data.copy()
user_data_vis["Label"] = "You"

compare_df = pd.concat([avg_data, user_data_vis])
compare_df = compare_df.set_index("Label")

st.bar_chart(compare_df.T)

# --- Model selection ---
model_choice = st.sidebar.selectbox(
    "Choose Classifier",
    ("Logistic Regression", "Decision Tree", "Random Forest", "SVM", "KNN"),
    key="model_choice"
)

if model_choice == "Logistic Regression":
    option = st.sidebar.selectbox("Model Style", ["Simple", "Balanced", "Flexible"], key="log_reg_style")
    if option == "Simple":
        model = LogisticRegression(C=0.5, max_iter=100)
    elif option == "Balanced":
        model = LogisticRegression(C=1.0, max_iter=200)
    else:
        model = LogisticRegression(C=2.0, max_iter=300)

elif model_choice == "Decision Tree":
    option = st.sidebar.selectbox("Tree Style", ["Shallow", "Medium", "Deep"], key="tree_style")
    if option == "Shallow":
        model = DecisionTreeClassifier(max_depth=3)
    elif option == "Medium":
        model = DecisionTreeClassifier(max_depth=6)
    else:
        model = DecisionTreeClassifier(max_depth=None)

elif model_choice == "Random Forest":
    option = st.sidebar.selectbox("Forest Style", ["Small", "Standard", "Large"], key="forest_style")
    if option == "Small":
        model = RandomForestClassifier(n_estimators=50, max_depth=5)
    elif option == "Standard":
        model = RandomForestClassifier(n_estimators=100, max_depth=10)
    else:
        model = RandomForestClassifier(n_estimators=200, max_depth=15)

elif model_choice == "SVM":
    option = st.sidebar.selectbox("SVM Style", ["Strict", "Balanced", "Flexible"], key="svm_style")
    if option == "Strict":
        model = SVC(C=0.5, kernel="linear", probability=True)
    elif option == "Balanced":
        model = SVC(C=1.0, kernel="rbf", probability=True)
    else:
        model = SVC(C=2.0, kernel="poly", probability=True)

elif model_choice == "KNN":
    option = st.sidebar.selectbox("KNN Style", ["Very Sensitive", "Moderate", "Smooth"], key="knn_style")
    if option == "Very Sensitive":
        model = KNeighborsClassifier(n_neighbors=3)
    elif option == "Moderate":
        model = KNeighborsClassifier(n_neighbors=7)
    else:
        model = KNeighborsClassifier(n_neighbors=12)

# --- Train & Predict ---
model.fit(x_train, y_train)
accuracy = accuracy_score(y_test, model.predict(x_test))

# 🔹 Show Metrics Dashboard
col1, col2, col3 = st.columns(3)
col1.metric("Model Accuracy", f"{accuracy*100:.2f}%")

from sklearn.metrics import classification_report

st.subheader("Classification Report")

# Generate classification report as a dict
report = classification_report(y_test, model.predict(x_test), output_dict=True)

# Convert to DataFrame
report_df = pd.DataFrame(report).transpose()

# Show as a nice table
st.dataframe(report_df.style.background_gradient(cmap="Blues").format(precision=2))




# 🔹 Personalized Health Insights
st.subheader("Health Insights")
if user_data["Glucose"][0] > 140:
    st.warning("High glucose detected! ⚠️ Consider reducing sugar intake.")
if user_data["BMI"][0] > 30:
    st.warning("Your BMI indicates obesity. Regular exercise is recommended.")
if user_data["BloodPressure"][0] > 90:
    st.warning("High blood pressure detected. Monitor regularly.")
if user_data["Age"][0] > 50:
    st.info("Age is a risk factor. Regular health checkups advised.")

# --- Prediction Report ---
user_result = model.predict(user_data)
st.subheader('Your Report:')
if user_result[0] == 0:
    st.success('You Are Healthy ✅')
else:
    st.error('You Are Not Healthy ⚠️')

# --- Model Comparison Dashboard ---
st.subheader("Model Comparison (Accuracy Scores)")
models = {
    "Logistic Regression": LogisticRegression(max_iter=300),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier()
}
accuracy_scores = {}
for name, m in models.items():
    m.fit(x_train, y_train)
    accuracy_scores[name] = accuracy_score(y_test, m.predict(x_test))
st.bar_chart(pd.DataFrame.from_dict(accuracy_scores, orient='index', columns=['Accuracy']))


















