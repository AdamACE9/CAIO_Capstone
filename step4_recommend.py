import pickle
import numpy as np

# Load the best model and scaler
print("Loading best model and scaler...")
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

print("[SUCCESS] Model and scaler loaded!")

# Create recommendation function
def recommend_actions(predicted_energy, server_load, cooling_efficiency):
    """
    Generate recommendations based on predicted energy and system metrics

    Parameters:
    - predicted_energy: Predicted energy consumption in kWh
    - server_load: Server load percentage
    - cooling_efficiency: Cooling efficiency percentage

    Returns:
    - List of recommendations
    """
    recommendations = []

    # Energy-based recommendations
    if predicted_energy > 450:
        recommendations.append("[!] HIGH ENERGY - Reduce server load 15-20%")
    elif predicted_energy > 420:
        recommendations.append("[!] ELEVATED ENERGY - Monitor cooling")
    else:
        recommendations.append("[OK] OPTIMAL ENERGY - Maintain settings")

    # Cooling efficiency recommendations
    if cooling_efficiency < 75:
        recommendations.append("[MAINTENANCE] Improve cooling efficiency")

    # Server load recommendations
    if server_load > 80:
        recommendations.append("[PERFORMANCE] Reduce server load")

    return recommendations

# Test with 5 sample inputs
print("\n" + "="*70)
print("TESTING RECOMMENDATION SYSTEM")
print("="*70)

test_cases = [
    {'Server_Load_percent': 90, 'Ambient_Temperature_C': 28, 'Cooling_Efficiency_percent': 72, 'Hour': 14, 'Day_of_Week': 2, 'Month': 7},
    {'Server_Load_percent': 65, 'Ambient_Temperature_C': 22, 'Cooling_Efficiency_percent': 88, 'Hour': 3, 'Day_of_Week': 5, 'Month': 3},
    {'Server_Load_percent': 85, 'Ambient_Temperature_C': 26, 'Cooling_Efficiency_percent': 78, 'Hour': 18, 'Day_of_Week': 1, 'Month': 8},
    {'Server_Load_percent': 55, 'Ambient_Temperature_C': 20, 'Cooling_Efficiency_percent': 92, 'Hour': 8, 'Day_of_Week': 3, 'Month': 11},
    {'Server_Load_percent': 95, 'Ambient_Temperature_C': 29, 'Cooling_Efficiency_percent': 70, 'Hour': 16, 'Day_of_Week': 4, 'Month': 6}
]

for i, test_input in enumerate(test_cases, 1):
    print(f"\nTest Case {i}:")
    print(f"  Server Load: {test_input['Server_Load_percent']}%")
    print(f"  Ambient Temperature: {test_input['Ambient_Temperature_C']}C")
    print(f"  Cooling Efficiency: {test_input['Cooling_Efficiency_percent']}%")
    print(f"  Hour: {test_input['Hour']}, Day: {test_input['Day_of_Week']}, Month: {test_input['Month']}")

    # Prepare input for prediction
    input_array = np.array([[
        test_input['Server_Load_percent'],
        test_input['Ambient_Temperature_C'],
        test_input['Cooling_Efficiency_percent'],
        test_input['Hour'],
        test_input['Day_of_Week'],
        test_input['Month']
    ]])

    # Scale and predict
    input_scaled = scaler.transform(input_array)
    predicted_energy = model.predict(input_scaled)[0]

    print(f"  Predicted Energy: {predicted_energy:.2f} kWh")

    # Get recommendations
    recommendations = recommend_actions(
        predicted_energy,
        test_input['Server_Load_percent'],
        test_input['Cooling_Efficiency_percent']
    )

    print("  Recommendations:")
    for rec in recommendations:
        print(f"    - {rec}")

print("\n" + "="*70)
print("[SUCCESS] Recommendation system tested successfully!")
print("="*70)
