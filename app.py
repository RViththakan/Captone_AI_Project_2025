from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Function to load pickle files safely
def load_pickle(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        print(f"Error: {file_path} not found.")
        return None

# Load models and encoders
hydroponics_model = load_pickle('plant_recommendation_model_hydrophonics.pkl')
hydroponics_encoders = load_pickle('label_encoders_hydrophonics.pkl')

market_model = load_pickle('market_trends_model_market.pkl')

target_file = 'Market_trends.csv'
df_market = pd.read_csv(target_file, encoding='Windows-1252') if os.path.exists(target_file) else None

large_scale_model = load_pickle('plant_recommendation_model_largescale.pkl')
large_scale_encoders = load_pickle('label_encoders_largescale.pkl')

exotic_model = load_pickle('plant_recommendation_model_exotic.pkl')
exotic_encoders = load_pickle('label_encoders_exotic.pkl')

home_model = load_pickle('plant_recommendation_model_home.pkl')
home_encoders = load_pickle('label_encoders_home.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_hydroponics', methods=['POST'])
def predict_hydroponics():
    if not hydroponics_model or not hydroponics_encoders:
        return jsonify({"error": "Model or encoders not loaded properly"})
    
    try:
        data = request.json
        user_input = pd.DataFrame([data])
        
        for col in user_input.columns:
            if col in hydroponics_encoders:
                le = hydroponics_encoders[col]
                if user_input[col][0] in le.classes_:
                    user_input[col] = le.transform(user_input[col])
                else:
                    user_input[col] = le.transform([le.classes_[0]])  # Default to most common class
        
        prediction = hydroponics_model.predict(user_input)
        watering, sunlight, maintenance, recommended_plant = prediction[0]
        
        return jsonify({
            "Watering": watering,
            "Sunlight": sunlight,
            "Maintenance": maintenance,
            "Recommended Plant": recommended_plant
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/predict_market', methods=['POST'])
def predict_market():
    if df_market is None:
        return jsonify({"error": "Market trends dataset not found"})
    
    try:
        crop_name = request.json.get('crop', '').strip().lower()
        print(f"Looking for crop: {crop_name}")  # Debugging line
        filtered_df = df_market[df_market['Crop'].str.lower() == crop_name]
        
        # Debugging the filtered dataframe
        print(f"Filtered DataFrame: {filtered_df}")  # Debugging line
        
        if not filtered_df.empty:
            crop_data = filtered_df.iloc[0]
            return jsonify({
                "Best Selling Months": crop_data['Best Selling Months'],
                "Peak Demand Regions": crop_data['Peak Demand Regions'],
                "Market Insights": crop_data['Market Insights']
            })
        return jsonify({"error": "Crop not found in dataset"})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/predict_large_scale', methods=['POST'])
def predict_large_scale():
    if not large_scale_model or not large_scale_encoders:
        return jsonify({"error": "Model or encoders not loaded properly"})
    
    try:
        data = request.json
        user_input = pd.DataFrame([data])
        
        for col in user_input.columns:
            if col in large_scale_encoders:
                le = large_scale_encoders[col]
                if user_input[col][0] in le.classes_:
                    user_input[col] = le.transform(user_input[col])
                else:
                    user_input[col] = le.transform([le.classes_[0]])  # Default to most common class
        
        prediction = large_scale_model.predict(user_input)
        
        return jsonify({
            "Recommended Plant": prediction[0][0],
            "Crop Yield Potential": prediction[0][1],
            "Pest and Disease Resistance": prediction[0][2]
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/predict_exotic', methods=['POST'])
def predict_exotic():
    if not exotic_model or not exotic_encoders:
        return jsonify({"error": "Model or encoders not loaded properly"})
    
    try:
        data = request.json
        user_input = pd.DataFrame([data])
        
        for col in user_input.columns:
            if col in exotic_encoders:
                le = exotic_encoders[col]
                if user_input[col][0] in le.classes_:
                    user_input[col] = le.transform(user_input[col])
                else:
                    user_input[col] = le.transform([le.classes_[0]])  # Default to most common class
        
        prediction = exotic_model.predict(user_input)
        return jsonify({"Recommended Plant": prediction[0]})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/predict_home', methods=['POST'])
def predict_home():
    if not home_model or not home_encoders:
        return jsonify({"error": "Model or encoders not loaded properly"})
    
    try:
        data = request.json
        user_input = pd.DataFrame([data])
        
        for col in user_input.columns:
            if col in home_encoders:
                le = home_encoders[col]
                if user_input[col][0] in le.classes_:
                    user_input[col] = le.transform(user_input[col])
                else:
                    user_input[col] = le.transform([le.classes_[0]])  # Default to most common class
        
        prediction = home_model.predict(user_input)
        return jsonify({"Recommended Plant": prediction[0]})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
