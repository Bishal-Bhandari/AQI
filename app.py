from flask import Flask, send_from_directory, jsonify, render_template
import requests
import json
import os

app = Flask(__name__)

# Load API keys
with open('api_keys.json') as json_file:
    api_keys = json.load(json_file)
API_KEY = api_keys['Weather_API']['API_key']

AQI_URL = "http://api.openweathermap.org/data/2.5/air_pollution"

# Major cities in Nepal
CITIES = [
    {"name": "Kathmandu", "lat": 27.7172, "lon": 85.3240},
    {"name": "Pokhara", "lat": 28.2096, "lon": 83.9856},
    {"name": "Biratnagar", "lat": 26.4525, "lon": 87.2718},
    {"name": "Lalitpur", "lat": 27.6670, "lon": 85.3247},
]


app = Flask(__name__)

def get_aqi(lat, lon):
    params = {
        "lat": lat,
        "lon": lon,
        "appid": API_KEY
    }
    response = requests.get(AQI_URL, params=params)
    data = response.json()

    aqi = data["list"][0]["main"]["aqi"]
    components = data["list"][0]["components"]

    return {
        "aqi": aqi,
        "components": components
    }


@app.route("/aqi")
def aqi_data():
    results = []
    for city in CITIES:
        aqi_info = get_aqi(city["lat"], city["lon"])
        results.append({**city, **aqi_info})

    return jsonify(results)


@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)

@app.route("/")
def index():
    return send_from_directory(".", "templates/index.html")

@app.route("/style.css")
def css():
    return send_from_directory(".", "templates/style.css")

@app.route("/geojson")
def geojson():
    return send_from_directory(".", "data.geojson")

if __name__ == "__main__":
    app.run(debug=True)