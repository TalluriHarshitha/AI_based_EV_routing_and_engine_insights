# AI based EV Routing and Engine Insights
The project showcases an innovative solution that combines EV route optimization and vehicle engine condition prediction. It highlights the use of advanced algorithms for efficient route planning for electric vehicles, considering battery levels and charging stations. Additionally, the system predicts engine health based on various vehicle parameters, helping identify maintenance needs. The application integrates these features for enhanced vehicle performance and travel efficiency.

The project involves two main functionalities: **EV Route Optimization** and **Vehicle Engine Prediction**.
- Setting up Streamlit for Web App:
- The streamlit library is used to create an interactive web interface. The app provides two options: EV Route Optimization and Vehicle Engine Prediction.
- A custom CSS style is applied to make the app visually appealing (Olympic Blue background, black and white buttons, etc.).

**Part 1: EV Route Optimization**
- The app allows users to select a source and destination from predefined places (like Place A, Place B, etc.).
- Users can enter vehicle parameters like battery level, charging rate, discharge rate, speed, and capacity.
- The Dijkstra’s Algorithm is used to find the shortest path between the source and destination, considering the vehicle’s current battery level.
- If a charging station is encountered along the route, the vehicle will "charge" before continuing. This is handled by a Depth First Search (DFS) algorithm.
- After the route is computed, the app displays the optimal route, total time, and remaining battery.

**Part 2: Vehicle Energy Data Analysis & Engine Status Prediction**
In this part, vehicle engine data is analyzed and a machine learning model is trained to predict engine health.
Data Loading and Exploration:
-   The engine_data.csv file is loaded into a pandas DataFrame for analysis.
-   The describe() function gives an overview of the data, such as the count, mean, and standard deviation for each column.
-   A new feature Engine_power is derived by multiplying Engine rpm and Lub oil pressure.

Feature Engineering:
- A new feature Temperature_difference is created by subtracting lub oil temp from coolant temp, as it could be a valuable feature for predicting engine condition.
- The Engine_power column is dropped as it is no longer needed.
  
Preparing Data for Machine Learning:
- The target variable (Engine Condition) is separated from the features.
- Data is split into training and testing sets (60% for training and 40% for testing).
  
Training a Gradient Boosting Model (GBM):
- A Gradient Boosting Classifier (GBM) model is trained on the training data. This model is an ensemble method that combines several weak models to make a stronger, more accurate model.
- The model is trained with the following parameters: 100 estimators (trees), a learning rate of 0.1, and a maximum depth of 3.

Model Evaluation:
- The model's accuracy is calculated on the test data using accuracy_score, which measures how often the model's predictions match the actual values.
- An F1-score report is generated using classification_report, which includes precision, recall, and F1-score for both classes (0 = Normal, 1 = Needs Maintenance).
- Precision is the proportion of true positive predictions out of all positive predictions.
- Recall is the proportion of true positive predictions out of all actual positives.
- F1-score is the harmonic mean of precision and recall, balancing the two.
- The model's performance is assessed using these metrics, with an overall accuracy of about 67% and a better recall for predicting engine maintenance.

**Block diagram**
![image](https://github.com/user-attachments/assets/46812437-a7c1-435e-8b30-b6906ef0e65a)



