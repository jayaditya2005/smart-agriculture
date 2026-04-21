import requests

def fetch_weather_by_coords(lat, lon, location_name="Custom Farm Location"):
    """
    Fetches real-time weather from Open-Meteo using exact latitude and longitude.
    This is highly accurate for rural areas and farms.
    """
    try:
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation&timezone=auto"
        w_resp = requests.get(weather_url)
        w_data = w_resp.json()
        
        current = w_data.get('current', {})
        
        return {
            "city": location_name,
            "country": "",
            "temperature": current.get('temperature_2m'),
            "humidity": current.get('relative_humidity_2m'),
            "precipitation": current.get('precipitation'),
            "lat": lat,
            "lon": lon
        }
    except Exception as e:
        return {"error": str(e)}

def fetch_weather(location_name):
    """
    Fetches real-time weather from Open-Meteo for a given location string.
    First uses geocoding to get lat, lon.
    Then gets the weather data.
    """
    try:
        # Step 1: Geocoding
        geocode_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location_name}&count=1&language=en&format=json"
        geo_resp = requests.get(geocode_url)
        geo_data = geo_resp.json()
        
        if not geo_data.get('results'):
            return {"error": f"Location '{location_name}' not found. Try entering a nearby district or exact coordinates."}
            
        location = geo_data['results'][0]
        lat = location['latitude']
        lon = location['longitude']
        
        # Step 2: Fetch Weather
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation&timezone=auto"
        w_resp = requests.get(weather_url)
        w_data = w_resp.json()
        
        current = w_data.get('current', {})
        
        return {
            "city": location['name'],
            "country": location.get('country', ''),
            "temperature": current.get('temperature_2m'),
            "humidity": current.get('relative_humidity_2m'),
            "precipitation": current.get('precipitation'),
            "lat": lat,
            "lon": lon
        }
        
    except Exception as e:
        return {"error": str(e)}
