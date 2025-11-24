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