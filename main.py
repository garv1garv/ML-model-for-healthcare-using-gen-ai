import datetime
from modules import (
    prediction,
    report_analyzer,
    location_alarm,
    appointment_scheduler,
    posture_detector,
    hydration_tracker,
    cycle_predictor
)

def display_header():
    print("\n" + "="*50)
    print(f"=== Health Guardian System [{datetime.date.today().strftime('%Y-%m-%d')}] ===".center(50))
    print("="*50)

def get_float_input(prompt):
    """Helper function for safe float input"""
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Invalid input. Please enter a number.")

def get_int_input(prompt, min_val=None, max_val=None):
    """Helper function for safe integer input"""
    while True:
        try:
            value = int(input(prompt))
            if (min_val is not None and value < min_val) or (max_val is not None and value > max_val):
                raise ValueError
            return value
        except ValueError:
            print(f"Invalid input. Please enter an integer{' between ' + str(min_val) + '-' + str(max_val) if min_val is not None else ''}.")

def main():
    while True:
        display_header()
        print("\nMain Menu:")
        print("1. Health Risk Prediction (ML + AI)")
        print("2. Health Report Analyzer")
        print("3. Location-Based Health Alerts")
        print("4. Doctor Appointment Scheduler")
        print("5. Real-Time Posture Detection")
        print("6. Smart Hydration Tracker")
        print("7. Cycle Tracker")
        print("0. Exit System")
        
        choice = input("\nSelect option (0-7): ").strip()
        
        if choice == "1":
            try:
                print("\n=== Health Risk Assessment ===")
                calories = get_float_input("Daily calorie intake: ")
                activity = get_int_input("Activity level (1-5 where 5=very active): ", 1, 5)
                water = get_float_input("Daily water intake (liters): ")
                genetic = get_int_input("Genetic risk (0=low, 1=high): ", 0, 1)
                query = input("Genetic predispositions (e.g., diabetes, hypertension): ")

                ml_result = prediction.predict_health_risk([calories, activity, water, genetic])
                ai_result = prediction.predict_potential_diseases(query, activity)

                print("\n--- Health Risk Results ---")
                print(f"ML Prediction: {'High risk' if ml_result == 1 else 'Low risk'}")
                print("\nAI Health Insights:")
                print(ai_result or "No AI analysis available")

            except Exception as e:
                print(f"\nError: {str(e)}")

        elif choice == "2":
            try:
                image_path = input("\nPath to health report image: ")
                plan = report_analyzer.analyze_report(image_path)
                print("\n--- Custom Health Plan ---")
                print(plan)
            except FileNotFoundError:
                print("Error: Image file not found")

        elif choice == "3":
            location = input("\nEnter your location: ")
            alert = location_alarm.check_location_alarm(location)
            print(f"\nHealth Alerts for {location}:")
            print(alert)

        elif choice == "4":
            try:
                print("\n=== Schedule Appointment ===")
                doctor = input("Doctor's name: ")
                patient = input("Patient's name: ")
                time_str = input("Appointment time (YYYY-MM-DD HH:MM): ")
                appointment_time = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M")
                
                if appointment_time < datetime.datetime.now():
                    raise ValueError("Cannot schedule appointments in the past")
                
                appointment_scheduler.schedule_appointment(doctor, patient, appointment_time)
                print("\nAppointment successfully scheduled!")
                
            except ValueError as e:
                print(f"Invalid input: {str(e)}")

        elif choice == "5":
            print("\nStarting posture monitoring... (Press Q to exit)")
            posture_detector.run_posture_detection()

        elif choice == "6":
            try:
                print("\n=== Hydration Calculator ===")
                weight = get_float_input("Weight (kg): ")
                activity_desc = input("Activity level (sedentary/active): ").lower()
                temp = get_float_input("Ambient temperature (°C): ")
                
                recommendation = hydration_tracker.calculate_water_intake(weight, activity_desc, temp)
                print(f"\nRecommended daily water intake: {recommendation:.2f} liters")
                
            except ValueError as e:
                print(f"Calculation error: {str(e)}")

        elif choice == "7":
            try:
                print("\n=== Cycle Tracker ===")
                cycle_dates = []
                print("Enter past cycle start dates (YYYY-MM-DD). Type 'done' when finished:")
                
                while True:
                    date_input = input("Date: ").strip()
                    if date_input.lower() == 'done':
                        break
                    try:
                        date = datetime.datetime.strptime(date_input, "%Y-%m-%d").date()
                        if date > datetime.date.today():
                            raise ValueError("Future dates not allowed")
                        cycle_dates.append(date)
                    except ValueError as e:
                        print(f"Invalid date: {str(e)}")
                
                if len(cycle_dates) >= 2:
                    predicted_date, avg_length = cycle_predictor.predict_next_cycle(cycle_dates)
                    print(f"\nAverage cycle length: {avg_length} days")
                    print(f"Predicted next cycle start: {predicted_date.strftime('%Y-%m-%d')}")
                else:
                    print("Insufficient data for prediction (need at least 2 dates)")
                    
            except Exception as e:
                print(f"Prediction error: {str(e)}")

        elif choice == "0":
            print(f"\nThank you for using Health Guardian. Exiting at {datetime.datetime.now().strftime('%H:%M')}")
            break

        else:
            print("\nInvalid selection. Please choose 0-7")

        input("\nPress Enter to continue...")

if __name__ == '__main__':
    main()