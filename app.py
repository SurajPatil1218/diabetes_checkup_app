import streamlit as st
import pandas as pd
from PIL import Image
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

st.subheader('Visualisation')
st.bar_chart(df)

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

# Model tuning options - simplified wording
if model_choice == "Logistic Regression":
    complexity = st.sidebar.slider("Model Complexity (lower = simpler, higher = more flexible)", 1, 10, 5)
    training_time = st.sidebar.slider("Training Effort (more effort = better fit)", 100, 500, 200)
    model = LogisticRegression(C=complexity, max_iter=training_time)

elif model_choice == "Decision Tree":
    depth = st.sidebar.slider("Tree Depth (how detailed the tree is)", 1, 20, 5)
    min_samples = st.sidebar.slider("Minimum Samples to Split (higher = simpler tree)", 2, 20, 2)
    model = DecisionTreeClassifier(max_depth=depth, min_samples_split=min_samples)

elif model_choice == "Random Forest":
    trees = st.sidebar.slider("Number of Trees (more trees = more stable)", 10, 300, 100)
    depth = st.sidebar.slider("Tree Depth (how detailed the forest trees are)", 1, 20, 5)
    model = RandomForestClassifier(n_estimators=trees, max_depth=depth)

elif model_choice == "SVM":
    flexibility = st.sidebar.slider("Model Flexibility (low = strict, high = flexible)", 1, 10, 5)
    kernel = st.sidebar.selectbox("Type of Decision Boundary", ("linear", "rbf", "poly", "sigmoid"))
    model = SVC(C=flexibility, kernel=kernel)

elif model_choice == "KNN":
    neighbors = st.sidebar.slider("Number of Neighbors (lower = sensitive, higher = smoother)", 1, 20, 5)
    model = KNeighborsClassifier(n_neighbors=neighbors)



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




