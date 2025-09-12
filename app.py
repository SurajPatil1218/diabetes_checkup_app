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

st.title('Diabetes Checkup')

st.subheader('Training Data')
st.write(df.describe())

# Split
x = df.drop(['Outcome'], axis=1)
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Sidebar sliders for user input
def user_report():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 70)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0, 67, 20)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47)
    age = st.sidebar.slider('Age', 21, 88, 33)

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
st.subheader("Visualisation (Dataset Average vs Your Data)")

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
    ("Logistic Regression", "Decision Tree", "Random Forest", "SVM", "KNN")
)

if model_choice == "Logistic Regression":
    option = st.sidebar.selectbox("Model Style", ["Simple", "Balanced", "Flexible"])
    if option == "Simple":
        model = LogisticRegression(C=0.5, max_iter=100)
    elif option == "Balanced":
        model = LogisticRegression(C=1.0, max_iter=200)
    else:
        model = LogisticRegression(C=2.0, max_iter=300)

elif model_choice == "Decision Tree":
    option = st.sidebar.selectbox("Tree Style", ["Shallow", "Medium", "Deep"])
    if option == "Shallow":
        model = DecisionTreeClassifier(max_depth=3)
    elif option == "Medium":
        model = DecisionTreeClassifier(max_depth=6)
    else:
        model = DecisionTreeClassifier(max_depth=None)

elif model_choice == "Random Forest":
    option = st.sidebar.selectbox("Forest Style", ["Small", "Standard", "Large"])
    if option == "Small":
        model = RandomForestClassifier(n_estimators=50, max_depth=5)
    elif option == "Standard":
        model = RandomForestClassifier(n_estimators=100, max_depth=10)
    else:
        model = RandomForestClassifier(n_estimators=200, max_depth=15)

elif model_choice == "SVM":
    option = st.sidebar.selectbox("SVM Style", ["Strict", "Balanced", "Flexible"])
    if option == "Strict":
        model = SVC(C=0.5, kernel="linear", probability=True)
    elif option == "Balanced":
        model = SVC(C=1.0, kernel="rbf", probability=True)
    else:
        model = SVC(C=2.0, kernel="poly", probability=True)

elif model_choice == "KNN":
    option = st.sidebar.selectbox("KNN Style", ["Very Sensitive", "Moderate", "Smooth"])
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
col2.metric("Your Glucose", user_data["Glucose"][0])
col3.metric("Your BMI", user_data["BMI"][0])

# 🔹 Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, model.predict(x_test))
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# 🔹 Classification Report
st.subheader("Classification Report")
st.text(classification_report(y_test, model.predict(x_test)))

# 🔹 ROC Curve
if hasattr(model, "predict_proba"):
    y_pred_prob = model.predict_proba(x_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    st.subheader("ROC Curve")
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], 'r--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

# 🔹 Feature Importance (for tree-based models)
if hasattr(model, "feature_importances_"):
    st.subheader("Feature Importance")
    importance = pd.DataFrame({
        "Feature": x.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    st.bar_chart(importance.set_index("Feature"))

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

# 🔹 Downloadable Report
result_text = f"""
Model: {model_choice}
Accuracy: {accuracy*100:.2f}%
Prediction: {"Not Healthy ⚠️" if user_result[0]==1 else "Healthy ✅"}
"""
buffer = io.BytesIO()
buffer.write(result_text.encode())
st.download_button("Download Report", buffer, file_name="diabetes_report.txt")

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


user_data_vis = user_data.copy()   # safe copy for charts
user_data_vis["Label"] = "You"

compare_df = pd.concat([avg_data, user_data_vis])
compare_df = compare_df.set_index("Label")

st.bar_chart(compare_df.T)

# Model selection
model_choice = st.sidebar.selectbox(
    "Choose Classifier",
    ("Logistic Regression", "Decision Tree", "Random Forest", "SVM", "KNN")
)

# Beginner-friendly tuning options
if model_choice == "Logistic Regression":
    option = st.sidebar.selectbox("Model Style", ["Simple", "Balanced", "Flexible"])
    if option == "Simple":
        model = LogisticRegression(C=0.5, max_iter=100)
    elif option == "Balanced":
        model = LogisticRegression(C=1.0, max_iter=200)
    else:  # Flexible
        model = LogisticRegression(C=2.0, max_iter=300)

elif model_choice == "Decision Tree":
    option = st.sidebar.selectbox("Tree Style", ["Shallow", "Medium", "Deep"])
    if option == "Shallow":
        model = DecisionTreeClassifier(max_depth=3)
    elif option == "Medium":
        model = DecisionTreeClassifier(max_depth=6)
    else:  # Deep
        model = DecisionTreeClassifier(max_depth=None)

elif model_choice == "Random Forest":
    option = st.sidebar.selectbox("Forest Style", ["Small", "Standard", "Large"])
    if option == "Small":
        model = RandomForestClassifier(n_estimators=50, max_depth=5)
    elif option == "Standard":
        model = RandomForestClassifier(n_estimators=100, max_depth=10)
    else:  # Large
        model = RandomForestClassifier(n_estimators=200, max_depth=15)

elif model_choice == "SVM":
    option = st.sidebar.selectbox("SVM Style", ["Strict", "Balanced", "Flexible"])
    if option == "Strict":
        model = SVC(C=0.5, kernel="linear")
    elif option == "Balanced":
        model = SVC(C=1.0, kernel="rbf")
    else:  # Flexible
        model = SVC(C=2.0, kernel="poly")

elif model_choice == "KNN":
    option = st.sidebar.selectbox("KNN Style", ["Very Sensitive", "Moderate", "Smooth"])
    if option == "Very Sensitive":
        model = KNeighborsClassifier(n_neighbors=3)
    elif option == "Moderate":
        model = KNeighborsClassifier(n_neighbors=7)
    else:  # Smooth
        model = KNeighborsClassifier(n_neighbors=12)

# Train & Predict
model.fit(x_train, y_train)
accuracy = accuracy_score(y_test, model.predict(x_test))

st.subheader(f'Model Accuracy ({model_choice}):')
st.write(f"{accuracy * 100:.2f}%")

user_result = model.predict(user_data)

st.subheader('Your Report:')
if user_result[0] == 0:
    st.success('You Are Healthy ✅')
else:
    st.error('You Are Not Healthy ⚠️')











