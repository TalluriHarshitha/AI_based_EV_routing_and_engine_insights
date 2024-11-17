# AI based EV Routing and Engine Insights
The project showcases an innovative solution that combines EV route optimization and vehicle engine condition prediction. It highlights the use of advanced algorithms for efficient route planning for electric vehicles, considering battery levels and charging stations. Additionally, the system predicts engine health based on various vehicle parameters, helping identify maintenance needs. The application integrates these features for enhanced vehicle performance and travel efficiency.

The project involves two main functionalities: **EV Route Optimization** and **Vehicle Engine Prediction**.
  1. **EV Route Optimization Code:**
   - The code implements an algorithm for determining the most efficient route for an electric vehicle (EV) based on its current battery level, charging rate, discharge rate, speed, and capacity. 
   - It uses **Dijkstra's algorithm** to calculate the shortest path between source and destination, considering available charging stations where the EV can recharge if needed. The route is optimized for time, factoring in the vehicle's battery consumption and recharging time.

  2. **Vehicle Engine Prediction Code:**
   - The engine prediction model uses machine learning to predict the health of a vehicle's engine based on various input parameters like engine RPM, oil pressure, fuel pressure, coolant pressure, and temperatures.
   - A **Gradient Boosting Machine (GBM)** model is trained on historical engine data to classify the engine's condition (normal or requiring maintenance). The model leverages features such as engine power and temperature differences to assess the likelihood of engine failure.

Together, the two components provide a comprehensive solution for efficient electric vehicle route planning and proactive engine maintenance prediction.


