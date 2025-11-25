from flask import Flask, send_from_directory
import requests
import json
import os

app = Flask(__name__)

# Fetch AQI from OpenAQ API for a country
def fetch_aqi(country_code="NP"):
    url = f"https://api.openaq.org/v2/latest?country={country_code}&limit=100"
    response = requests.get(url)
    data = response.json()

    aqi_points = []
    for item in data.get("results", []):
        lat = item["coordinates"]["latitude"]
        lon = item["coordinates"]["longitude"]
        city = item.get("city", "Unknown")
        location = item.get("location", "Unknown")

        # Extract PM2.5 or fall back
        pm25 = "N/A"
        for m in item["measurements"]:
            if m["parameter"] == "pm25":
                pm25 = m["value"]

        aqi_points.append({
            "city": city,
            "location": location,
            "pm25": pm25,
            "lat": lat,
            "lon": lon
        })

    return aqi_points

# Filter your country
COUNTRY = "Nepal"   # Change to any country
country_geo = geo[geo["ADMIN"] == COUNTRY]

# Save as local GeoJSON
country_geo.to_file("data.geojson", driver="GeoJSON")
print("data.geojson created successfully!")

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/style.css")
def css():
    return send_from_directory(".", "style.css")

@app.route("/geojson")
def geojson():
    return send_from_directory(".", "data.geojson")

if __name__ == "__main__":
    app.run(debug=True)