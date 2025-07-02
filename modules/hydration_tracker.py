def calculate_water_intake(weight_kg, activity_level, temperature_c):
    # Base water intake: 35 ml per kg body weight
    base_intake_ml = weight_kg * 35
    
    # Increase for active individuals
    if activity_level.lower() == "active":
        base_intake_ml += 500
    
    # Increase for high temperature: add 200 ml for each degree above 25°C
    if temperature_c > 25:
        base_intake_ml += (temperature_c - 25) * 200
    
    return base_intake_ml / 1000  # Convert ml to liters

# Standalone test
if __name__ == '__main__':
    weight = 70        # in kg
    activity = "active"
    temperature = 30   # in °C
    intake = calculate_water_intake(weight, activity, temperature)
    print(f"Recommended daily water intake: {intake:.2f} liters")
