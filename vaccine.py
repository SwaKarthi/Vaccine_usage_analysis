import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib


st.set_page_config(page_title="Vaccine Usage analysis",
                   layout="wide",
                   initial_sidebar_state="expanded",
                   menu_items={'About': """# This dashboard app is created for Vaccine Usage Analysis!"""})

# Load data
def load_data():
    vaccine = pd.read_csv("https://raw.githubusercontent.com/nethajinirmal13/Training-datasets/main/Vaccine.csv")
    return vaccine

vaccine = load_data()

st.sidebar.header(":wave: :red[**Welcome to the Swathi's dashboard!**]")
with st.sidebar:
    selected = option_menu("Menu", ["Home","Vaccine analysis","Personal & Behaviour","Prediction"], 
                icons=["house", "graph-up-arrow", "person", "bullseye"],
                menu_icon= "menu-button-wide",
                default_index=0,
                styles={"nav-link": {"font-size": "18px", "text-align": "left", "margin": "-2px", "--hover-color": "#EFC3CA"},
                        "nav-link-selected": {"background-color": "#D20103"}})

if selected == "Home":
    st.markdown("# :red[Vaccine Usage analysis and Prediction]")
    st.markdown("## :red[A User-Friendly Tool Using Streamlit and Matplotlib]") 
    col1,col2 = st.columns([2,2],gap="small")
  
    with col1:
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.markdown("### :green[Domain :] Data Analysis and Machine Learning")
        st.markdown("### :green[Technologies used :] Python, Pandas, Streamlit and Matplotlib")
        st.markdown("### :green[Overview :]  This streamlit app can be used to visualize the Vaccine Usage analysis. Bar charts, Pie charts are used to get insights.")

    with col2:
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.image(Image.open("C:\\Users\\krkar\\Downloads\\vaccine.jpg"), width=400)

if selected == "Vaccine analysis":
    col1,col2 = st.columns([2,2],gap="large")
    with col1:
    # Data Summary
        fig, ax = plt.subplots()
        vaccine['h1n1_worry'].value_counts().plot(kind='bar', ax=ax)
        st.write("## Count of H1N1 Worry")
        # Set labels
        plt.xlabel('H1N1 Worry')
        plt.ylabel('Count')
        st.pyplot(fig)

    with col2:
        # Display the count plot
        fig, ax = plt.subplots()
        st.write("## H1N1 Awareness")
        sns.countplot(x='h1n1_awareness', data=vaccine, hue='h1n1_vaccine', ax=ax)
        plt.xlabel('h1n1 Awareness')
        plt.ylabel('Count')
        plt.legend(title='h1n1 Vaccine')
        st.pyplot(fig)

    with col1:
        #Antiviral Medication
        counts = vaccine['antiviral_medication'].value_counts()
        st.write("## Distribution of Antiviral Medication")
        # Plot the pie chart
        fig, ax = plt.subplots(figsize=[5, 5])
        counts.plot(kind='pie', autopct='%0.2f%%', explode=[0, 0.2], ax=ax)
        st.pyplot(fig)

    #H1N1 Vaccine Effective or Seas Vaccine Effective
    colors = ['#CAFF70', '#FF1493', '#00BFFF', '#FFD700', '#836FFF']
    colors1 = ['#FF7F24', '#FFB90F', '#A2CD5A', '#BF3EFF', '#EEAEEE']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[10, 5])
    st.write("## H1N1 Vaccine Effective or Seas Vaccine Effective")
    # Plot the first pie chart
    ax1.pie(vaccine['is_h1n1_vacc_effective'].value_counts(), labels=vaccine['is_h1n1_vacc_effective'].value_counts().index,
            autopct='%0.2f%%', explode=[0.1, 0, 0, 0, 0], colors=colors, shadow=True)

    # Plot the second pie chart
    ax2.pie(vaccine['is_seas_vacc_effective'].value_counts(), labels=vaccine['is_seas_vacc_effective'].value_counts().index,
            autopct='%0.2f%%', explode=[0.1, 0, 0, 0, 0], colors=colors1, shadow=True)

    # Set titles for the subplots
    ax1.set_title('is_h1n1_vacc_effective')
    ax2.set_title('is_seas_vacc_effective')
    st.pyplot(fig)

    #H1N1 Risky or Seas Risky
    fig, ax = plt.subplots(1, 2, figsize=[10, 5])
    st.write("## H1N1 Risky or Seas Risky")
    # Plot the first pie chart
    vaccine['is_h1n1_risky'].value_counts().plot(kind='pie', autopct='%0.2f%%', explode=[0.05, 0, 0, 0, 0], cmap='RdYlGn', ax=ax[0])

    # Plot the second pie chart
    vaccine['is_seas_risky'].value_counts().plot(kind='pie', autopct='%0.2f%%', explode=[0.05, 0, 0, 0, 0], cmap='Paired', ax=ax[1])

    ax[0].set_title('is_h1n1_risky')
    ax[1].set_title('is_seas_risky')
    st.pyplot(fig)


    #Sick from H1N1 Vaccine or Seas Vaccine
    st.write("## Sick from H1N1 Vaccine or Seas Vaccine")
    fig, ax = plt.subplots(1, 2, figsize=[10, 5])
    
    # Plot the first pie chart
    vaccine['sick_from_h1n1_vacc'].value_counts().plot(kind='pie', autopct='%0.2f%%', explode=[0.05, 0, 0, 0, 0], cmap='Spectral', ax=ax[0])

    # Plot the second pie chart
    vaccine['sick_from_seas_vacc'].value_counts().plot(kind='pie', autopct='%0.2f%%', explode=[0.05, 0, 0, 0, 0], cmap='twilight', ax=ax[1])

    # Set titles for the subplots
    ax[0].set_title('sick_from_h1n1_vacc')
    ax[1].set_title('sick_from_seas_vacc')
    st.pyplot(fig)

if selected == "Personal & Behaviour":
    col1,col2 = st.columns([2,2],gap="large")
    with col1:
        fig, ax = plt.subplots()
        st.write("## Person who bought Face Mask")
        vaccine['bought_face_mask'].value_counts().plot(kind = 'bar', color = 'orange', ax=ax)
        plt.xlabel('bought_face_mask')
        plt.ylabel('count')
        plt.show()
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        st.write("## Wash Hands Frequently")
        # Plot the count plot
        sns.countplot(x='wash_hands_frequently', data=vaccine, ax=ax)
        # Customize labels and title
        plt.xlabel('Wash Hands Frequently')
        plt.ylabel('Count')
        st.pyplot(fig)

    with col1:
        fig, ax = plt.subplots()
        st.write("## Children under 6 months")
        vaccine['cont_child_undr_6_mnths'].value_counts().plot(kind='bar', cmap='rainbow', edgecolor='b', ax=ax)
        # Customize labels and title
        plt.xlabel('Count')
        plt.ylabel('Cont Child Undr 6 Mnths')
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=[8, 6])
        st.write("## Distribution of Housing Status")
        # Plot the bar chart
        vaccine['housing_status'].value_counts().plot(kind='bar', color='b', edgecolor='r', ax=ax)

        # Set labels and title
        plt.xlabel('Housing Status')
        plt.ylabel('Count')
        st.pyplot(fig)

    with col1:
        fig, ax = plt.subplots(figsize=[5, 5])
        vaccine['income_level'].value_counts().plot(kind='pie', autopct='%0.2f%%', cmap='magma', explode=[0, 0.1, 0], shadow=True, ax=ax)
        ax.set_title('Income Level')
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=[5, 5])

        # Plot the pie chart
        vaccine['employment'].value_counts().plot(kind='pie', autopct='%0.2f%%', ax=ax)
        plt.title('Employment Distribution')
        st.pyplot(fig)

    with col1:
        fig, ax = plt.subplots(1, 2, figsize=[7, 5], sharey=True)
        st.write(" ")
        st.write("## No of Adults vs No of Children")
        # Plot the count plots
        sns.countplot(x='no_of_adults', data=vaccine, ax=ax[0])
        sns.countplot(x='no_of_children', data=vaccine, ax=ax[1])
        ax[0].set_title('Number of Adults')
        ax[1].set_title('Number of Children')
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        st.write("## Contact Avoidance")
        sns.countplot(x='contact_avoidance', data=vaccine, color='green', ax=ax)
        plt.ylabel('Count')
        plt.xlabel('Contact Avoidance')
        st.pyplot(fig)

if selected == "Prediction":
  
    # Drop rows with missing values
    vaccine.dropna(inplace=True)

    # Encode categorical variables
    label_encoder = LabelEncoder()

    # Fit and transform categorical variables except 'housing_status' and 'employment'
    vaccine['age_bracket'] = label_encoder.fit_transform(vaccine['age_bracket'])
    vaccine['qualification'] = label_encoder.fit_transform(vaccine['qualification'])
    vaccine['sex'] = label_encoder.fit_transform(vaccine['sex'])
    vaccine['race'] = label_encoder.fit_transform(vaccine['race'])
    vaccine['income_level'] = label_encoder.fit_transform(vaccine['income_level'])
    vaccine['marital_status'] = label_encoder.fit_transform(vaccine['marital_status'])


    # Transform 'housing_status' and 'employment' columns
    # Handle unseen labels by assigning a unique label for them
    vaccine['housing_status'] = label_encoder.fit_transform(vaccine['housing_status'])
    vaccine['employment'] = label_encoder.fit_transform(vaccine['employment'])
    vaccine['census_msa'] = label_encoder.fit_transform(vaccine['census_msa'])

    # Define features and target variable
    X = vaccine.drop(columns=['h1n1_worry', 'h1n1_vaccine','unique_id','race', 'is_h1n1_vacc_effective','is_h1n1_risky','is_seas_vacc_effective','has_health_insur', 'is_seas_risky','qualification','income_level','housing_status','employment',])
    y = vaccine['h1n1_vaccine']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    # Standardize features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the logistic regression model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    # Evaluation
    train_accuracy = accuracy_score(y_train, model.predict(X_train_scaled))
    test_accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
    report = classification_report(y_test, model.predict(X_test_scaled))

    st.title('Vaccination Prediction Model')
    st.write('Model Training Complete!')
    st.write('Training Accuracy:', train_accuracy)
    st.write('Test Accuracy:', test_accuracy)

    # Convert classification report to DataFrame
    classification_report_dict = classification_report(y_test, model.predict(X_test_scaled), output_dict=True)
    classification_report_df = pd.DataFrame(classification_report_dict).T

    # Display the classification report DataFrame
    st.write('Classification Report:')
    st.table(classification_report_df)

    # Save the model
    joblib.dump(model, 'vaccine_prediction_model.pkl')

    #header
    st.header('Vaccination Prediction')

    # Sidebar options
    selected_option = st.selectbox('Select an option', ['Train Model', 'Make Prediction'])

    # Train Model
    if selected_option == 'Train Model':
        st.title('Train a Logistic Regression Model')
        st.write('This model predicts vaccination outcomes based on given features.')

        # Display training accuracy and report
        st.write('Training Accuracy:', train_accuracy)

        # Convert classification report to DataFrame
        classification_report_dict = classification_report(y_test, model.predict(X_test_scaled), output_dict=True)
        classification_report_df = pd.DataFrame(classification_report_dict).T

        # Display the classification report DataFrame
        st.write('Classification Report:')
        st.table(classification_report_df)

    # Make Prediction
    elif selected_option == 'Make Prediction':
        st.title('Make Prediction')
        st.write('Enter the features below to predict vaccination outcome.')

        # Load the model
        try:
            model = joblib.load('vaccine_prediction_model.pkl')
        except FileNotFoundError:
            st.error('Model file not found. Please train the model first.')

        if 'model' in locals():
            # Input features
            col_1, col_2 = st.columns([4, 4])
            with col_1:
                h1n1_awareness = st.selectbox('h1n1_awareness', [0, 1, 2], index=1)
                antiviral_medication = st.selectbox('antiviral_medication', [0, 1], index=1)
                contact_avoidance = st.selectbox('contact_avoidance', [0, 1], index=1)
                bought_face_mask = st.selectbox('bought_face_mask', [0, 1], index=1) 
                wash_hands_frequently = st.selectbox('wash_hands_frequently', [0, 1], index=1)
                avoid_large_gatherings = st.selectbox('avoid_large_gatherings', [0, 1], index=1)
                reduced_outside_home_cont = st.selectbox('reduced_outside_home_cont', [0, 1], index=1)
                avoid_touch_face = st.selectbox('avoid_touch_face', [0, 1], index=1)
                dr_recc_h1n1_vacc = st.selectbox('dr_recc_h1n1_vacc', [0, 1], index=1) 
                dr_recc_seasonal_vacc = st.selectbox('dr_recc_seasonal_vacc', [0, 1], index=1)
                chronic_medic_condition = st.selectbox('chronic_medic_condition', [0, 1], index=1)
                
                
            with col_2:
                cont_child_undr_6_mnths = st.selectbox('cont_child_undr_6_mnths', [0, 1], index=1)
                is_health_worker = st.selectbox('is_health_worker', [0, 1], index=1)
                sick_from_h1n1_vacc = st.selectbox('sick_from_h1n1_vacc', [0, 1, 2, 3, 4], index=1)
                sick_from_seas_vacc = st.selectbox('sick_from_seas_vacc', [0, 1, 2, 3, 4], index=1) 
                age_bracket = st.number_input('age_bracket', min_value=0, max_value=100, value=25) 
                sex = st.selectbox('sex', [0, 1], index=1)
                marital_status = st.selectbox('marital_status', [0, 1], index=1)
                census_msa = st.selectbox('census_msa', [0, 1, 2], index=1)
                no_of_adults = st.selectbox('no_of_adults', [0, 1, 2, 3], index=1)
                no_of_children = st.selectbox('no_of_children', [0, 1, 2, 3], index=1)

            # Make prediction
            if st.button('Predict'):
                prediction = model.predict([[h1n1_awareness, antiviral_medication, contact_avoidance, bought_face_mask,
                                        wash_hands_frequently, avoid_large_gatherings, reduced_outside_home_cont, avoid_touch_face,
                                        dr_recc_h1n1_vacc, dr_recc_seasonal_vacc, chronic_medic_condition, cont_child_undr_6_mnths,
                                        is_health_worker, sick_from_h1n1_vacc, sick_from_seas_vacc, age_bracket, sex, marital_status, census_msa, no_of_adults, no_of_children]])

                if prediction == 1:
                    st.write('Recommendation: Take the vaccine')
                else:
                    st.write('Recommendation: Do not take the vaccine')