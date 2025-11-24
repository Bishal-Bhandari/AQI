from flask import Flask, send_from_directory
import geopandas as gpd
import json
import os

app = Flask(__name__)

# Load world GeoJSON
geo = gpd.read_file(
    "https://raw.githubusercontent.com/datasets/geo-boundaries-world-110m/master/countries.geojson"
)

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