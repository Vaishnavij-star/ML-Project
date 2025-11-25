from flask import Flask, request, render_template
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Initialize Flask app
app = Flask(__name__)

# Home page route
@app.route('/')
def index():
    return render_template('index.html')  # Main landing page

# Prediction route
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')  # Show form

    else:
        try:
            # Collect data from form
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('reading_score')),
                writing_score=float(request.form.get('writing_score'))
            )

            # Convert input into dataframe
            pred_df = data.get_data_as_dataframe()
            print("Input DataFrame:\n", pred_df)

            # Load prediction pipeline and predict
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            print("Prediction Result:", results[0])

            # Return results to template
            return render_template('home.html', results=results[0])

        except Exception as e:
            return render_template('home.html', error=f"Prediction failed: {e}")


# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
