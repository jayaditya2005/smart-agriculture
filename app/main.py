import streamlit as st
import sys
import os

# Add the app directory to sys.path so we can import utils safely
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from app.utils.weather_api import fetch_weather, fetch_weather_by_coords
from app.utils.model_inference import (
    analyze_leaf,
    predict_crop_and_fertilizer,
    predict_crop_yield,
    assess_soil_health
)

st.set_page_config(
    page_title="Smart Agriculture System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Premium CSS
css_file = os.path.join(os.path.dirname(__file__), "style.css")
if os.path.exists(css_file):
    with open(css_file, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def render_home():
    st.title("🌱 Smart Agriculture Hub")
    
    # Hero Section
    st.markdown("#### Empowering rural farms with AI-driven insights.")
    st.write("Welcome to your digital farming assistant. Our platform combines Deep Learning Vision models with real-time Open-Meteo environmental data to maximize your yields and protect your crops.")
    
    st.image("https://images.unsplash.com/photo-1586771107445-d3ca888129ff?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80", use_container_width=True)
    
    st.markdown("---")
    
    # Features Grid
    st.markdown("### 🚜 Available Modules")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("#### ⛅ Dynamic Vision & Weather")
        st.write("Upload a photo of any sick leaf. Our CNN automatically diagnoses the exact disease, cross-references your live local farm weather, and issues a customized treatment plan.")
        
        st.info("#### 📈 Yield Prediction")
        st.write("Predict your total expected harvest volume based on your specific acreage, crop type, and expected seasonal temperature factors before you even plant.")
        
    with col2:
        st.warning("#### 🌍 Soil Health Analysis")
        st.write("Turn your chemical soil test into an easy-to-read health score. Instantly know if your NPK, pH, or Organic Carbon levels need immediate intervention.")
        
        st.error("#### 🌾 Crop & Fertilizer Guide")
        st.write("Not sure what to plant? Input your soil traits and let the algorithm recommend the mathematically optimal crop and baseline fertilizer needs.")

def render_tabular_models():
    st.title("🌾 Crop & Fertilizer Recommendation")
    st.markdown("Enter your soil and environmental metrics below:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50)
        p = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=50)
        k = st.number_input("Potassium (K)", min_value=0, max_value=200, value=50)
    
    with col2:
        temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
    
    with col3:
        ph = st.number_input("pH level", min_value=0.0, max_value=14.0, value=6.5)
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)
        
    if st.button("Predict Optimal Crop and Fertilizer", use_container_width=True):
        st.info("Running Random Forest ML Prediction...")
        result = predict_crop_and_fertilizer(n, p, k, temp, humidity, ph, rainfall)
        st.success(f"**Recommended Crop:** {result['crop'].capitalize()}")
        st.write(f"**Fertilizer Advice:** {result['advice']}")

def render_yield_prediction():
    st.title("📈 Crop Yield Prediction")
    st.markdown("Estimate your total expected harvest based on crop details and farm size.")
    
    col1, col2 = st.columns(2)
    with col1:
        crop = st.selectbox("Select Crop:", ['Rice', 'Maize', 'Chickpea', 'Cotton', 'Sugarcane', 'Wheat'])
        season = st.selectbox("Season:", ['Kharif', 'Rabi', 'Summer'])
        area = st.number_input("Farm Area (in Acres):", min_value=0.1, value=5.0)
    
    with col2:
        rainfall = st.number_input("Expected Seasonal Rainfall (mm):", min_value=0.0, value=500.0)
        temp = st.number_input("Average Temperature (°C):", min_value=10.0, max_value=50.0, value=25.0)
        
    if st.button("Predict Yield", use_container_width=True):
        with st.spinner("Calculating expected yield..."):
            estimated_yield = predict_crop_yield(crop, season, area, rainfall, temp)
        st.success(f"**Estimated Harvest:** {estimated_yield} Tons")
        st.info(f"*(This averages out to {round(estimated_yield/area, 2)} Tons per Acre)*")

def render_soil_health():
    st.title("🌍 Soil Health Analysis")
    st.markdown("Assess the quality of your soil by entering its chemical properties.")
    
    col1, col2 = st.columns(2)
    with col1:
        n = st.number_input("Nitrogen (N) kg/ha", min_value=0, max_value=300, value=120)
        p = st.number_input("Phosphorus (P) kg/ha", min_value=0, max_value=300, value=40)
        k = st.number_input("Potassium (K) kg/ha", min_value=0, max_value=300, value=200)
    with col2:
        ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
        ec = st.number_input("Electrical Conductivity (dS/m)", min_value=0.0, max_value=5.0, value=1.2)
        oc = st.number_input("Organic Carbon (%)", min_value=0.0, max_value=3.0, value=0.6)
        
    if st.button("Analyze Soil Health", use_container_width=True):
        with st.spinner("Analyzing soil metrics..."):
            score, status = assess_soil_health(n, p, k, ph, ec, oc)
            
        if status == "Excellent":
            st.success(f"**Health Score:** {score}/100 - Your soil is in **Excellent** condition!")
        elif status == "Moderate":
            st.warning(f"**Health Score:** {score}/100 - Your soil is in **Moderate** condition. Consider supplementing Organic Carbon.")
        else:
            st.error(f"**Health Score:** {score}/100 - Your soil is in **Poor** condition. Needs immediate nutrient intervention.")

def render_dynamic_system():
    st.title("⛅ Dynamic Fertilizer & Disease System")
    st.markdown("Upload a leaf image to detect deficiencies and get weather-adjusted recommendations.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Farm Location")
        location_method = st.radio("Search location by:", ["Village/District Name", "Exact Coordinates"])
        
        if location_method == "Village/District Name":
            loc_name = st.text_input("Enter Location Name:", placeholder="e.g. Sangli, Palakkad")
            lat, lon = None, None
        else:
            loc_name = st.text_input("Custom Location Label:", placeholder="e.g. My Farm")
            v_col1, v_col2 = st.columns(2)
            lat = v_col1.number_input("Latitude", format="%.6f", value=19.0760)
            lon = v_col2.number_input("Longitude", format="%.6f", value=72.8777)
            
        st.subheader("2. Plant Leaf Image")
        uploaded_file = st.file_uploader("Upload an image of the plant leaf", type=["jpg", "jpeg", "png"])
        
    with col2:
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            
    if st.button("Analyze & Recommend", type="primary", use_container_width=True):
        if (location_method == "Village/District Name" and not loc_name) or uploaded_file is None:
            st.warning("Please provide both a valid location strategy and a leaf image.")
        else:
            with st.spinner("Analyzing image and fetching farm weather data..."):
                # 1. Image Classification
                leaf_class = analyze_leaf(uploaded_file.read())
                
                # 2. Weather Fetching
                if location_method == "Village/District Name":
                    weather = fetch_weather(loc_name)
                else:
                    weather = fetch_weather_by_coords(lat, lon, loc_name if loc_name else "Custom Farm Area")
                
            if "error" in weather:
                st.error(weather["error"])
            else:
                st.success(f"Classification Complete! Detected: **{leaf_class}**")
                
                st.subheader(f"Current Weather in {weather['city']}, {weather['country']}")
                cols = st.columns(3)
                cols[0].metric("Temperature", f"{weather['temperature']} °C")
                cols[1].metric("Humidity", f"{weather['humidity']} %")
                cols[2].metric("Precipitation", f"{weather['precipitation']} mm")
                
                # 3. Dynamic Rule Engine Recommendations
                st.subheader("Dynamic Fertilizer Plan")
                
                if leaf_class in ["Healthy crop leaf", "Healthy NPK crop leaf"]:
                    st.write("🌿 Your plant looks perfectly healthy! No immediate fertilizer or pesticide required.")
                else:
                    if leaf_class in ["Tomato leaf early blight disease", "Potato leaf late blight disease"]:
                        rec = "🧪 **Base Recommendation:** Administer a Copper-based Fungicide or Chlorothalonil to halt the blight spread."
                    elif leaf_class == "Apple leaf scab disease":
                        rec = "🧪 **Base Recommendation:** Apply Captan or Myclobutanil fungicide sprays. Ensure fallen leaves are swept away."
                    elif leaf_class == "Corn leaf rust disease":
                        rec = "🧪 **Base Recommendation:** Use foliar fungicides containing strobilurins or triazoles immediately."
                    elif leaf_class == "Crop leaf calcium deficiency":
                        rec = "🧪 **Base Recommendation:** Add Agricultural Lime or Gypsum to the soil to raise calcium levels."
                    elif leaf_class == "Crop leaf magnesium deficiency":
                        rec = "🧪 **Base Recommendation:** Apply Epsom Salt (Magnesium Sulfate) directly to the soil or as a foliar spray."
                    elif leaf_class == "Nitrogen (N) deficiency":
                        rec = "🧪 **Base Recommendation:** Apply Urea, Ammonium Nitrate, or Organic Manure immediately to boost Nitrogen levels and enhance vegetative growth."
                    elif leaf_class == "Phosphorus (P) deficiency":
                        rec = "🧪 **Base Recommendation:** Apply Superphosphate (SSP), Diammonium Phosphate (DAP), or Bone Meal to encourage root and flower development."
                    elif leaf_class == "Potassium (K) deficiency":
                        rec = "🧪 **Base Recommendation:** Apply Muriate of Potash (MOP), Potassium Sulfate, or Kelp Meal to improve disease resistance and fruit quality."
                    else:
                        rec = "🧪 **Base Recommendation:** Apply a balanced NPK fertilizer."
                        
                    st.write(rec)
                    
                    if weather['precipitation'] and weather['precipitation'] > 2.0:
                        st.warning(f"🌧️ **Weather Alert:** {weather['precipitation']}mm rain is expected. **DO NOT** apply fertilizer today as it will wash away into run-offs. Wait for dry weather.")
                    elif weather['temperature'] and weather['temperature'] > 35.0:
                        st.warning(f"☀️ **Weather Alert:** Extremely high temperature ({weather['temperature']}°C). Apply fertilizer only during early morning or late evening to prevent crop burn.")
                    else:
                        st.success("🌤️ **Weather Alert:** Current weather conditions are optimal for fertilizer application.")

def main():
    with st.sidebar:
        st.title("Navigation")
        app_mode = st.radio(
            "Select a Module:",
            [
                "Home", 
                "Crop & Fertilizer Guide", 
                "Dynamic Decision System",
                "Yield Prediction",
                "Soil Health Analysis"
            ]
        )
        st.markdown("---")
        st.caption("Developed for Mini Project")

    if app_mode == "Home":
        render_home()
    elif app_mode == "Crop & Fertilizer Guide":
        render_tabular_models()
    elif app_mode == "Dynamic Decision System":
        render_dynamic_system()
    elif app_mode == "Yield Prediction":
        render_yield_prediction()
    elif app_mode == "Soil Health Analysis":
        render_soil_health()

if __name__ == "__main__":
    main()
