import datetime
import json
import os

APPOINTMENTS_FILE = r"C:\Users\lenovo\Desktop\HACKATHON\modules\data\appointments.json"
appointments = []

def load_appointments():
    """Load appointments from the JSON file into the global list."""
    global appointments
    if os.path.exists(APPOINTMENTS_FILE):
        try:
            with open(APPOINTMENTS_FILE, "r") as f:
                appointments = json.load(f)
                for appt in appointments:
                    if isinstance(appt.get("time"), str):
                        appt["time"] = datetime.datetime.fromisoformat(appt["time"])
        except Exception as e:
            print("Error loading appointments:", e)
            appointments = []
    else:
        appointments = []

def save_appointments():
    """Save the global appointments list to the JSON file."""
    serializable_appointments = []
    for appt in appointments:
        serializable_appt = appt.copy()
        if isinstance(serializable_appt.get("time"), datetime.datetime):
            serializable_appt["time"] = serializable_appt["time"].isoformat()
        serializable_appointments.append(serializable_appt)
    try:
        directory = os.path.dirname(APPOINTMENTS_FILE)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(APPOINTMENTS_FILE, "w") as f:
            json.dump(serializable_appointments, f, indent=4)
    except Exception as e:
        print("Error saving appointments:", e)

def schedule_appointment(doctor, patient, appointment_time):
    """Schedule a new appointment, save it to file, and return the appointment."""
    for appt in appointments:
        if appt["doctor"] == doctor and appt["time"] == appointment_time:
            print("Conflict detected: This doctor already has an appointment at this time.")
            return None
        if appt["patient"] == patient and appt["time"] == appointment_time:
            print("Conflict detected: This patient already has an appointment at this time.")
            return None

    appointment = {
        "doctor": doctor,
        "patient": patient,
        "time": appointment_time  
    }
    appointments.append(appointment)
    save_appointments()
    print("Appointment scheduled successfully.")
    return appointment

def list_appointments():
    """Return the list of scheduled appointments."""
    if not appointments:
        print("No appointments scheduled.")
    else:
        for appt in appointments:
            print(f"Doctor: {appt['doctor']}, Patient: {appt['patient']}, Time: {appt['time']}")

def get_user_input():
    """Prompt user for appointment details and schedule the appointment."""
    doctor = input("Enter doctor's name: ").strip()
    patient = input("Enter patient's name: ").strip()
    date_str = input("Enter appointment date and time (YYYY-MM-DD HH:MM): ").strip()
    try:
        appointment_time = datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M")
        schedule_appointment(doctor, patient, appointment_time)
    except ValueError:
        print("Invalid date and time format. Please use 'YYYY-MM-DD HH:MM'.")

def main_menu():
    """Display the main menu and handle user choices."""
    load_appointments()
    while True:
        print("\nAppointment Scheduler")
        print("1. Schedule a new appointment")
        print("2. List all appointments")
        print("3. Exit")
        choice = input("Enter your choice: ").strip()
        if choice == '1':
            get_user_input()
        elif choice == '2':
            list_appointments()
        elif choice == '3':
            print("Exiting the scheduler.")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == '__main__':
    main_menu()