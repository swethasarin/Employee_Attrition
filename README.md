# Project Title: Employee Attrition Analysis and Prediction


**Step by step Approach:**
In this project, I have built a complete employee attrition prediction pipeline. First, I loaded and cleaned the dataset, dropping constant or irrelevant columns and standardizing column names. Handled outliers in numeric features and encoded categorical variables using Label Encoding for binary columns and One-Hot Encoding for multi-class columns. Next, I have applied feature scaling to numeric features using StandardScaler. The dataset was then split into training and testing sets, and I have trained two models: Logistic Regression and Random Forest, evaluating them using metrics such as Accuracy, Precision, Recall, F1-score, ROC-AUC, and visualizing confusion matrices and ROC curves. The best-performing model was selected based on ROC-AUC and saved along with the scaler and feature columns. Finally, I have developed a Streamlit web application with two sections: an interactive dashboard for EDA and a prediction module where users can input key employee details to receive attrition predictions, complete with probability visualization and a summary of the input data.

# Technologies used:

**Python** – Main programming language
**Pandas** – Data manipulation and preprocessing
**NumPy** – Numerical operations
**Matplotlib & Seaborn** – Data visualization and exploratory analysis
**Scikit-learn** – Machine Learning Model Development, preprocessing, metrics
model evaluation 
Logistic Regression,Random Forest Classifier
StandardScaler, LabelEncoder, train_test_split
**Pickle** – Saving and loading trained models and preprocessing objects
**Streamlit** – Web application framework for building interactive dashboards and prediction interface
**Jupyter Notebook** – Step-by-step development and exploratory data analysis
