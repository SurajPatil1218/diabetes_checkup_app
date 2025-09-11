import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Load Data
df = pd.read_csv("diabetes.csv")

st.title('Diabetes Checkup')

st.subheader('Training Data')
st.write(df.describe())

st.subheader("Visualisation with Your Data")

df_vis = df.copy()
df_vis["Label"] = "Dataset"
user_data["Label"] = "You"

df_plot = pd.concat([df_vis, user_data])
df_plot = df_plot.set_index("Label")

st.bar_chart(df_plot.drop("Outcome", axis=1).T)


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

user_data = user_report()

# Model selection (must come before tuning)
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
        model = DecisionTreeClassifier(max_depth=None)  # grow fully

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







