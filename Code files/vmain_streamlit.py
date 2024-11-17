import streamlit as st
import heapq
import pickle
import numpy as np

# Custom CSS for styling
st.markdown("""
    <style>
    /* App background - Olympic Blue */
    .stApp {
        background-color: #008ECC; /* Olympic Blue */
    }

    /* Sidebar background */
    .css-1d391kg {
        background-color: #FFFFFF !important; /* White background for the sidebar */
    }

    /* Header text - Title Heading */
    h1 {
        color: #FFFFFF !important; /* White color for the main heading */
    }

    /* Main content text */
    .stTextInput, .stNumberInput, .stSelectbox {
        color: #000000 !important; /* Black text for input fields */
        background-color: #FFFFFF; /* White background for inputs */
    }

    /* Buttons */
    button {
        color: #FFFFFF !important; /* White text for buttons */
        background-color: #000000; /* Black button background */
        border-radius: 8px;
        padding: 0.5em 1em;
        font-size: 16px;
    }

    /* Button hover effect */
    button:hover {
        background-color: #FFFFFF; /* White button background */
        color: #000000 !important; /* Black text for hover effect */
        border: 1px solid #000000; /* Black border */
    }

    /* Page text */
    .stMarkdown, .stMarkdown p, .stSelectbox > div > div {
        color: #000000 !important; /* Black text for other content */
    }

    /* Ensure white text for headings */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #FFFFFF !important;
    }

    /* Ensuring that page titles and content have black text */
    .stApp h1, .stApp h2, .stApp h3, .stApp p {
        color: #000000 !important; /* Black text everywhere except the header */
    }
    </style>
""", unsafe_allow_html=True)

# Predefined data for EV Route Optimization
PLACES = {
    0: "Place A",
    1: "Place B",
    2: "Place C",
    3: "Place D",
    4: "Place E",
    5: "Place F",
    6: "Place G",
    7: "Place H",
    8: "Place I",
    9: "Place J",
}

GRAPH = [
    [(1, 10), (2, 15)],  # Connections from Place A
    [(0, 10), (3, 12)],  # Connections from Place B
    [(0, 15), (3, 10), (4, 25)],  # Connections from Place C
    [(1, 12), (2, 10), (5, 20)],  # Connections from Place D
    [(2, 25), (6, 8)],  # Connections from Place E
    [(3, 20), (7, 18)],  # Connections from Place F
    [(4, 8), (8, 14)],  # Connections from Place G
    [(5, 18), (9, 11)],  # Connections from Place H
    [(6, 14), (9, 9)],  # Connections from Place I
    [(7, 11), (8, 9)],  # Connections from Place J
]

CHARGING_STATIONS = {2, 5, 8}  # Charging stations at Place C, Place F, and Place I

# Dijkstra's algorithm for shortest path
def dijkstra(graph, start, end):
    n = len(graph)
    distances = [float('inf')] * n
    distances[start] = 0
    prev_nodes = [-1] * n
    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node]:
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                prev_nodes[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))

    path = []
    at = end
    while at != -1:
        path.append(at)
        at = prev_nodes[at]
    path.reverse()

    return distances[end], path

# Optimal Route Function
def optimal_route(graph, source, destination, battery, charge_rate, discharge_rate, speed, capacity, stations):
    current_battery = battery
    visited = set()
    results = []

    def dfs(node, battery, time, path):
        if node == destination:
            results.append({'time': time, 'path': path[:], 'remaining_battery': battery})
            return

        for neighbor, distance in graph[node]:
            if neighbor in visited:
                continue
            required_battery = distance / discharge_rate
            if battery >= required_battery:
                visited.add(neighbor)
                dfs(neighbor, battery - required_battery, time + distance / speed, path + [neighbor])
                visited.remove(neighbor)
            elif neighbor in stations:  # Charging required
                charge_needed = required_battery - battery
                charging_time = charge_needed / charge_rate
                visited.add(neighbor)
                dfs(neighbor, capacity - required_battery, time + distance / speed + charging_time, path + [neighbor])
                visited.remove(neighbor)

    visited.add(source)
    dfs(source, current_battery, 0, [source])

    if results:
        best_result = min(results, key=lambda x: x['time'])
        return best_result
    else:
        return None

# Load the trained model for Vehicle Engine Prediction
with open("engine_model.pkl", 'rb') as file:
    engine_model = pickle.load(file)

# Streamlit App
def main():
    st.title("AI-Powered EV Routing and Vehicle Engine Insights")

    # Navigation menu
    option = st.sidebar.selectbox("Choose Application", ["Home", "EV Route Optimization", "Vehicle Engine Prediction"])

    # Home Page
    if option == "Home":
        st.write("""
        ## Welcome to the combined application of EV Routing and Vehicle Engine Insights
        Choose one of the following from the sidebar:
        - **EV Route Optimization**: Plan the most efficient route for your electric vehicle.
        - **Vehicle Engine Prediction**: Predict the condition of your vehicle's engine based on input parameters.
        """)

    # EV Route Optimization
    elif option == "EV Route Optimization":
        st.subheader("EV Route Optimization")
        source = st.selectbox("Source", options=list(PLACES.keys()), format_func=lambda x: PLACES[x])
        destination = st.selectbox("Destination", options=list(PLACES.keys()), format_func=lambda x: PLACES[x])

        st.subheader("Vehicle Parameters")
        battery = st.number_input("Current Battery Level (Range: 0.0 - 20.0)", min_value=0.0, max_value=20.0, value=10.0)
        charge_rate = st.number_input("Charging Rate (Range: 0.1 - 10.0 units/hour)", min_value=0.1, max_value=10.0, value=2.0)
        discharge_rate = st.number_input("Discharge Rate (Range: 0.1 - 5.0 units/distance)", min_value=0.1, max_value=5.0, value=1.0)
        speed = st.number_input("Vehicle Speed (Range: 1.0 - 200.0 distance/hour)", min_value=1.0, max_value=200.0, value=60.0)
        capacity = st.number_input("Battery Capacity (Range: 5.0 - 50.0 units)", min_value=5.0, max_value=50.0, value=20.0)

        if st.button("Calculate Optimal Route"):
            result = optimal_route(GRAPH, source, destination, battery, charge_rate, discharge_rate, speed, capacity, CHARGING_STATIONS)

            if result:
                st.success("Optimal Route Found!")
                path_names = [PLACES[node] for node in result['path']]
                st.write(f"**Path:** {' -> '.join(path_names)}")
                st.write(f"**Total Time:** {result['time']:.2f} hours")
                st.write(f"**Remaining Battery:** {result['remaining_battery']:.2f} units")
            else:
                st.error("No feasible route found!")

    # Vehicle Engine Prediction
    elif option == "Vehicle Engine Prediction":
        st.subheader("Vehicle Engine Prediction")

        # Engine RPM Input (Range: 61.0 to 2239.0)
        engine_rpm = st.number_input("Engine RPM (Range: 61.0 - 2239.0)", min_value=61.0, max_value=2239.0, value=1150.0, step=1.0)

        # Lub Oil Pressure Input (Range: 0.003384 to 7.265566)
        lub_oil_pressure = st.number_input("Lub Oil Pressure (Range: 0.003384 - 7.265566)", min_value=0.003384, max_value=7.265566, value=3.0, step=0.001)

        # Fuel Pressure Input (Range: 0.003187 to 21.138326)
        fuel_pressure = st.number_input("Fuel Pressure (Range: 0.003187 - 21.138326)", min_value=0.003187, max_value=21.138326, value=10.0, step=0.001)

        # Coolant Pressure Input (Range: 0.002483 to 7.478505)
        coolant_pressure = st.number_input("Coolant Pressure (Range: 0.002483 - 7.478505)", min_value=0.002483, max_value=7.478505, value=3.5, step=0.001)

        # Lub Oil Temperature Input (Range: 71.32 to 89.58)
        lub_oil_temp = st.number_input("Lub Oil Temperature (Range: 71.32 - 89.58)", min_value=71.32, max_value=89.58, value=80.0, step=0.1)

        # Coolant Temperature Input (Range: 61.67 to 195.53)
        coolant_temp = st.number_input("Coolant Temperature (Range: 61.67 - 195.53)", min_value=61.67, max_value=195.53, value=120.0, step=0.1)

        # Temperature Difference Input (Range: -22.67 to 119.01)
        temp_difference = st.number_input("Temperature Difference (Range: -22.67 - 119.01)", min_value=-22.67, max_value=119.01, value=50.0, step=0.1)

        if st.button("Predict Engine Condition"):
            input_data = np.array([engine_rpm, lub_oil_pressure, fuel_pressure, coolant_pressure, lub_oil_temp, coolant_temp, temp_difference]).reshape(1, -1)
            prediction = engine_model.predict(input_data)
            confidence = engine_model.predict_proba(input_data)[:, 1]  # For binary classification

            if prediction[0] == 0:
                st.info(f"Engine is in **Normal Condition**. Confidence: {confidence[0]:.2f}")
            else:
                st.warning(f"Engine may require **Maintenance**. Confidence: {confidence[0]:.2f}")

if __name__ == "__main__":
    main()
