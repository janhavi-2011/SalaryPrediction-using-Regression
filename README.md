💰 Salary Prediction Using Regression

👩‍💻 Author

Name: Janhavi Maurya

📌 Project Title

Salary Prediction Using Regression with Streamlit Dashboard

📖 Project Description

This project aims to predict employee salaries using machine learning techniques based on various features such as Age, Gender, Education Level, Job Title, and Years of Experience.

The project uses Linear Regression and other regression models to analyze the relationship between these features and salary. A Streamlit dashboard is developed to visualize the dataset, train models, compare model performance, and allow users to input employee details to predict salary in real time.

The system helps demonstrate how machine learning can be used for data-driven salary prediction and decision making.

⚙️ Methodology

The project follows a complete Machine Learning workflow:

1️⃣ Data Collection

The dataset contains employee information such as age, gender, education level, job title, experience, and salary.

2️⃣ Data Preprocessing

Handling missing values

Data cleaning

Feature selection

Encoding categorical variables using One Hot Encoding

3️⃣ Feature Engineering

Input features used in the model:

Age

Gender

Education Level

Job Title

Years of Experience

Target variable:

Salary

4️⃣ Model Training

The dataset is divided into training and testing data using:

Train-Test Split
Training Data → Model Learning
Testing Data → Model Evaluation
5️⃣ Machine Learning Model

The project mainly uses Linear Regression to predict salary.

6️⃣ Model Evaluation

Model performance is evaluated using:

R² Score

Mean Squared Error (MSE)

7️⃣ Visualization

Different visualizations are created to understand the dataset:

Salary vs Experience Scatter Plot

Actual vs Predicted Salary Graph

Training and Testing Data Visualization

8️⃣ Streamlit Dashboard

A user-friendly dashboard is developed where users can:

View dataset information

Train machine learning models

Compare models

Enter employee details

Predict salary instantly

📊 Results

The trained machine learning model achieved the following performance:

R² Score: ≈ 0.90 (90%)

This means the model explains about 90% of the variance in salary prediction, indicating strong predictive performance.

The dashboard also allows users to input details such as:

Age

Gender

Education Level

Job Title

Years of Experience

and receive a predicted salary output in real time.

🛠 Technologies Used

Python

Pandas

NumPy

Scikit-learn

Matplotlib

Streamlit

🚀 How to Run the Project
1️⃣ Install required libraries
pip install pandas numpy scikit-learn matplotlib streamlit
2️⃣ Run the Streamlit dashboard
streamlit run app.py

📌 Conclusion

This project demonstrates how Machine Learning and Data Visualization can be combined with Streamlit to build an interactive prediction system.

The developed dashboard provides a simple way to explore the dataset and predict employee salaries using machine learning models.
