import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the ML model
with open('best_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/report')
def report():
    return render_template('report.html')

@app.route('/score')
def score():
    return render_template('report.html')

@app.route('/generate_report', methods=['POST'])
def generate_report():
    # Retrieve user inputs from the form
    country = request.form['country']
    region = request.form['region']
    happinessRank = float(request.form['happinessRank'])
    standardError = float(request.form['standardError'])
    gdp = float(request.form['gdp'])
    family = float(request.form['family'])
    lifeExpectancy = float(request.form['lifeExpectancy'])
    freedom = float(request.form['freedom'])
    corruption = float(request.form['corruption'])
    generosity = float(request.form['generosity'])
    distopiaResidual = float(request.form['distopiaResidual'])

    # Prepare the input features for prediction
    input_features = [country,region,happinessRank, standardError, gdp, family, lifeExpectancy, freedom, corruption, generosity, distopiaResidual]

    # Make a prediction using the model
    happiness_score = model.predict([input_features])[0]

    # Render the report.html page with the generated happiness score
    return render_template('report.html', happiness_score=happiness_score)

if __name__ == '__main__':
    app.run(debug=True)
