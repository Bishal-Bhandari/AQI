import requests

def get_weather_data(api_key, location, units):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units={units}"

    r = requests.get(url)
    data = r.json()

    return {
        "weather_temp": data["main"]["temp"],
        "weather_humidity": data["main"]["humidity"],
        "wind_speed": data["wind"]["speed"],
        "wind_direction": data["wind"]["deg"]
    }
