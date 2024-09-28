# from flask import Flask, request, jsonify, render_template
# import pandas as pd
# import numpy as np
# from model import train_model  # Import your training function

# app = Flask(__name__)

# # Load your trained model and label encoders
# model, label_encoders = train_model()  # Unpack the function to get the trained model and encoders

# # Read CSV dataset once during app initialization
# df = pd.read_csv(r"C:\Users\athar\Downloads\sampled_1000_values.csv")  # Update as necessary

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# # def predict():
# #     data = request.form['inputData']
    
# #     try:
# #         # Split the input data and convert to the correct type
# #         input_data = data.split(',')  # Split the input by commas

# #         # Ensure the input is in the expected format
# #         if len(input_data) != 6:
# #             return jsonify({'error': "Input must have exactly 6 values."}), 400
        
# #         # Create a DataFrame with the correct feature names
# #         feature_names = ['Source', 'Destination', 'Protocol', 'Length', 'Info', 'Malicious']  # Correct feature names
# #         input_df = pd.DataFrame([input_data], columns=feature_names)  # Create a DataFrame

# #         # Encode categorical features using the encoders
# #         for col in ['Source', 'Destination', 'Info', 'Protocol']:
# #             if col in label_encoders:  # Check if the encoder exists
# #                 input_df[col] = label_encoders[col].transform(input_df[col])  # Transform using the encoder

# #         # Convert the numerical values appropriately
# #         input_df['Length'] = input_df['Length'].astype(float)
# #         input_df['Malicious'] = input_df['Malicious'].astype(int)  # Convert to integer if needed

# #         # Make a prediction
# #         prediction = model.predict(input_df)  # Pass the DataFrame to the model
        
# #         return jsonify({'prediction': str(prediction[0])})
    
# #     except ValueError as ve:
# #         # Handle the case where the input is not a valid float
# #         return jsonify({'error': f"Invalid input: {data}. Error: {str(ve)}"}), 400
# #     except Exception as e:
# #         # Capture any other exceptions and return a message
# #         return jsonify({'error': str(e)}), 400
# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.form['inputData']
    
#     try:
#         # Split the input data into a list of floats
#         float_data = [float(x) for x in data.split(',')]  # Convert the input to a list of floats

#         # Reshape the input to the correct format (1 row, 6 columns)
#         input_array = np.array(float_data).reshape(1, -1)

#         # Preprocess the data as per your ML model requirements
#         prediction = model.predict(input_array)  # Pass the array directly to the model
        
#         return jsonify({'prediction': str(prediction[0])})
    
#     except ValueError:
#         # Handle the case where the input is not valid
#         return jsonify({'error': f"Invalid input: {data} is not valid"}), 400
#     except Exception as e:
#         # Capture any other exceptions and return a message
#         return jsonify({'error': str(e)}), 400


# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from model import train_model, preprocess_input_data  # Import your preprocessing function and model

app = Flask(__name__)

# Load your trained model
model = train_model()  # Call the function to get the trained model

# Read CSV dataset once during app initialization
df = pd.read_csv(r"sampled_1000_values.csv")  # Update as necessary

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form['inputData']
    
    try:
        # Split the input data into a list of strings (or floats if necessary)
        float_data = [x.strip() for x in data.split(',')]

        # Preprocess the input data (e.g., label encoding, scaling, etc.)
        input_data = preprocess_input_data(float_data)  # Apply the same preprocessing as during training

        # Convert to numpy array and reshape if necessary
        input_array = np.array(input_data).reshape(1, -1)

        # Make a prediction
        prediction = model.predict(input_array)[0]  # Get the prediction (0 or 1)

        # Map prediction to malicious or non-malicious
        result = "Malicious" if prediction == 1 else "Non Malicious"

        return jsonify({'prediction': result})  # Return the result as a JSON response
    
    except ValueError as e:
        # Handle invalid input
        return jsonify({'error': f"Invalid input: {data} is not valid: {str(e)}"}), 400
    except Exception as e:
        # Capture any other exceptions and return a message
        return jsonify({'error': str(e)}), 400



if __name__ == "__main__":
    app.run(debug=True)
