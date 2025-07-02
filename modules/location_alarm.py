import requests
API_TOKEN = "ecc1296024f00b5e83efb07ba8b008bbba372a1c"

def get_air_quality_index(city):
    """
    Fetch the air quality index (AQI) for the given city using the AQICN API.
    Returns the AQI value as an integer or None if an error occurs.
    """
    url = f"https://api.waqi.info/feed/{city}/?token={API_TOKEN}"
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print("Error fetching data from AQICN API. HTTP status code:", response.status_code)
            return None

        data = response.json()
        if data.get("status") != "ok":
            print("Error in API response:", data.get("data"))
            return None
        aqi = data["data"].get("aqi")
        return aqi

    except Exception as e:
        print("Exception occurred while fetching AQI:", e)
        return None

def check_location_alarm(city):
    aqi = get_air_quality_index(city)
    if aqi is None:
        return "Unable to retrieve air quality data."

    print(f"Air Quality Index for {city}: {aqi}")
    if aqi > 100:
        alarm_message = "ALERT: Poor air quality detected! Consider taking precautions."
    else:
        alarm_message = "Air quality is acceptable."
    return alarm_message

if __name__ == '__main__':
    user_city = input("Enter the city name: ")
    message = check_location_alarm(user_city)
    print(message)

