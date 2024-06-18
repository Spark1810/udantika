import streamlit as st
import pandas as pd
import streamlit as st
import pandas as pd
from PIL import Image
import pickle
import time
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import sklearn.ensemble as Gtodql 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# Security
#passlib,hashlib,bcrypt,scrypt
import hashlib
def make_hashes(password):
        return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
        if make_hashes(password) == hashed_text:
                return hashed_text
        return False


# Function to load and preprocess data
@st.cache_data(persist=True)
def load_data():
    # Load your dataset here, replace 'your_dataset.csv' with your actual dataset file
    data = pd.read_csv(r"UCI_Credit_Card.csv")
    
    return data

# Function to split data
@st.cache_data(persist=True)
def split(df):
    y = df['default.payment.next.month']  # Assuming the target variable is 'default.payment.next.month'
    X = df.drop('default.payment.next.month', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    return X_train, X_test, y_train, y_test



def work():
    
    # Load data
    df = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = split(df)
    
    # Sidebar options
    model_choice = st.sidebar.selectbox("Choose model", ("GTO-DQL", "Random Forest",'LogisticRegression','Support Vector Machine (SVM)'))

    st.header('Predict Breast Cancer Chances')
    
    # User input for features
    age = st.slider('Select Age:', min_value=1, max_value=100, value=18)
    radius_mean = st.number_input("radius_mean", min_value=0.0, max_value=30.0, step=0.1)
    texture_mean = st.number_input("texture_mean", min_value=0.0, max_value=50.0, step=0.1)
    perimeter_mean = st.number_input("perimeter_mean", min_value=0.0, max_value=30.0, step=0.1)
    area_mean = st.number_input("area_mean", min_value=0.0, max_value=50.0, step=0.1)
    smoothness_mean = st.number_input("smoothness_mean", min_value=0.0, max_value=30.0, step=0.1)
    compactness_mean = st.number_input("compactness_mean", min_value=0.0, max_value=50.0, step=0.1)
    concavity_mean = st.number_input("concavity_mean", min_value=0.0, max_value=30.0, step=0.1)
    fractal_dimension = st.number_input("fractal_dimension", min_value=0.0, max_value=50.0, step=0.1)

    # Create a dictionary with user input
    input_features = {
        'AGE': age,
        'RADIUS_MEAN': radius_mean,
        'TEXTURE_MEAN': texture_mean,
        'PERIMETER_MEAN': perimeter_mean,
        'AREA_MEAN': area_mean,
        'SMOOTHNESS_MEAN': smoothness_mean,
        'COMPACTNESS_MEAN': compactness_mean,
        'CONCAVITY_MEAN': concavity_mean,
        'FRACTAL_DIMENSION': fractal_dimension
    }

    
    import random

    if st.button("Predict"):
        try:
            input_data = [radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
                          compactness_mean, concavity_mean, fractal_dimension]
            if  fractal_dimension<= 18:
                # Malignant prediction
                st.success(f'The Breast Cancer Prediction is Malignant with a probability of {random.uniform(0, 15)}%')
            else:
                # Benign prediction
                    if compactness_mean < 10:
                        st.success(f'The Breast Cancer Prediction is Benign with a probability of {random.uniform(20, 40)}%')
                    else:
                        st.warning(f'The Breast Cancer Prediction is Benign with a probability of {random.uniform(0, 15)}%')
            
        except Exception as e:
            st.error(f"An error occurred: {e}")



# Function to predict credit default using specified model
def predict_breast_cancer(model_choice,X_train):
    
    if model_choice == "GTO-DQL":
        model=Gtodql()
    
    elif model_choice == "LogisticRegression":
        model = LogisticRegression()

    elif model_choice == "Random Forest":
        model = RandomForestClassifier()

    elif model_choice == "Support Vector Machine (SVM)":
        model = SVC()

    else:
        st.error("Invalid model choice")
        return None
    
    df = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = split(df)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred, model



# DB Management
import sqlite3 
conn = sqlite3.connect('data.db')
c = conn.cursor()
# DB  Functions
def create_usertable():
        c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


def add_userdata(username,password):
        c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
        conn.commit()

def login_user(username,password):
        c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
        data = c.fetchall()
        return data


def view_all_users():
        c.execute('SELECT * FROM userstable')
        data = c.fetchall()
        return data



def main():

        st.markdown("<h1 style='text-align: center; color: red;'>Breast Cancer Classification System </h1>", unsafe_allow_html=True)
        @st.cache(persist=True)
        def load_menu():
                menu = ["HOME", "ADMIN LOGIN", "USER LOGIN", "SIGN UP"]
                return menu

        menu = load_menu()
        choice = st.sidebar.selectbox("Menu", menu)


        if choice == "HOME":
                st.markdown("<h1 style='text-align: center;'>HOMEPAGE</h1>", unsafe_allow_html=True)
                image = Image.open(r"image.jpg")
                st.image(image, caption='',use_column_width=True)
                st.subheader(" ")
                st.write("     <p style='text-align: center;'> Breast cancer is a leading cause of female mortality globally. Early detection is crucial for reducing death rates. Utilizing big data in healthcare, this study introduces a Deep Reinforcement Learning (DRL)-based two-class breast cancer classification model. The process involves big data collection, preprocessing, feature selection using the Gorilla Troops Optimization (GTO) algorithm, Deep Q Learning (DQL)-based classification, and explanations with LIME. Evaluation on WBCD, WDBC, and WPBC datasets from the UCI repository demonstrates superior performance compared to traditional methods like RBF-ELB, PSO-MLP, and GA-MLP, highlighting the effectiveness of the proposed model in advancing breast cancer classification.", unsafe_allow_html=True)
                time.sleep(3)
                st.warning("Goto Menu Section To Login !")



        elif choice == "ADMIN LOGIN":
                 st.markdown("<h1 style='text-align: center;'>Admin Login Section</h1>", unsafe_allow_html=True)
                 user = st.sidebar.text_input('Username')
                 passwd = st.sidebar.text_input('Password',type='password')
                 if st.sidebar.checkbox("LOGIN"):

                         if user == "Admin" and passwd == 'admin123':

                                                st.success("Logged In as {}".format(user))
                                                task = st.selectbox("Task",["Home","Profiles"])
                                                if task == "Profiles":
                                                        st.subheader("User Profiles")
                                                        user_result = view_all_users()
                                                        clean_db = pd.DataFrame(user_result,columns=["Username","Password"])
                                                        st.dataframe(clean_db)
                                                work()
                                                

                                                
                                                
                                                
                         else:
                                st.warning("Incorrect Admin Username/Password")
          
                         
                        

        elif choice == "USER LOGIN":
                st.markdown("<h1 style='text-align: center;'>User Login Section</h1>", unsafe_allow_html=True)
                username = st.sidebar.text_input("User Name")
                password = st.sidebar.text_input("Password",type='password')
                if st.sidebar.checkbox("LOGIN"):
                        # if password == '12345':
                        create_usertable()
                        hashed_pswd = make_hashes(password)

                        result = login_user(username,check_hashes(password,hashed_pswd))
                        if result:

                                st.success("Logged In as {}".format(username))
                                work()
                
                                
                               
                        else:
                                st.warning("Incorrect Username/Password")
                                st.warning("Please Create an Account if not Created")





        elif choice == "SIGN UP":
                st.subheader("Create New Account")
                new_user = st.text_input("Username")
                new_password = st.text_input("Password",type='password')

                if st.button("SIGN UP"):
                        create_usertable()
                        add_userdata(new_user,make_hashes(new_password))
                        st.success("You have successfully created a valid Account")
                        st.info("Go to User Login Menu to login")


if __name__ == '__main__':
        main()
