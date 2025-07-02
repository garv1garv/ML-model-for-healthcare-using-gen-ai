import os
import datetime
from datetime import timedelta

def predict_next_cycle(cycle_starts):
    if len(cycle_starts) < 2:
        print("Not enough data to predict the cycle.")
        return None, None
    cycle_lengths = [(cycle_starts[i] - cycle_starts[i-1]).days for i in range(1, len(cycle_starts))]
    average_cycle = sum(cycle_lengths) / len(cycle_lengths)
    last_cycle = cycle_starts[-1]
    predicted_next = last_cycle + timedelta(days=average_cycle)
    return predicted_next, average_cycle

def get_user_cycle_data():
    cycle_starts = []
    print("Enter your past menstrual cycle start dates.")
    print("Type 'done' when you have finished entering dates.")
    while True:
        date_input = input("Enter a date (YYYY-MM-DD): ")
        if date_input.lower() == 'done':
            break
        try:
            date_obj = datetime.datetime.strptime(date_input, '%Y-%m-%d').date()
            cycle_starts.append(date_obj)
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD.")
    return cycle_starts

def locate_report_file(filename='report.png'):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    reports_directory = os.path.join(current_directory, 'reports')
    report_file_path = os.path.join(reports_directory, filename)
    return report_file_path

if __name__ == '__main__':
    cycle_starts = get_user_cycle_data()
    if len(cycle_starts) < 2:
        print("Not enough data to predict the cycle.")
    else:
        predicted, avg = predict_next_cycle(cycle_starts)
        print(f"Average cycle length: {avg:.2f} days")
        print(f"Predicted next cycle start date: {predicted}")
    
    report_path = locate_report_file()
    if os.path.exists(report_path):
        print(f"Report file found at: {report_path}")
    else:
        print(f"Report file not found at: {report_path}")
