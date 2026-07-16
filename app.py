from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and preprocessor
model = joblib.load("D:/House_Price_Prediction/notebooks/models/house_price_model.pkl")
preprocessor = joblib.load("D:/House_Price_Prediction/notebooks/models/preprocessor.pkl")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    city = request.form["City"]
    property_type = request.form["Property_Type"]
    bhk = int(request.form["BHK"])
    bathroom = int(request.form["Bathroom"])
    parking = int(request.form["Parking"])
    total_area = float(request.form["Total_Area_in_Sqft"])

    input_df = pd.DataFrame({
        "City":[city],
        "Property_Type":[property_type],
        "BHK":[bhk],
        "Bathroom":[bathroom],
        "Parking":[parking],
        "Total_Area_in_Sqft":[total_area]
    })

    input_processed = preprocessor.transform(input_df)

    prediction = model.predict(input_processed)

    return render_template(
        "index.html",
        prediction=f"₹ {prediction[0]:,.2f}"
    )

if __name__ == "__main__":
    app.run(debug=True)