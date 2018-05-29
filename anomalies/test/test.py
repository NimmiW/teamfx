
import statistics
print(str(11.8381-6.8118))

list=[10,6]

mean_list = statistics.mean(list)
print(str(mean_list))

sd = statistics.stdev(list)
print(int(sd))





"""

@app.route("/anomalies/")
def index():
    return render_template('anomalies/index.html')

@app.route("/anomalies/input")
def get_input():
    return render_template('anomalies/get_input.html')

@app.route("/anomalies/selectfeatures", methods = ['POST', 'GET'])
def feature_selecion():

    year, from_month, to_month, currency, labels, price_values, gradients_values, volatility_values, volatility_gradients_values, length\
        = feature_selection.feature_selecion()
    return render_template('anomalies/feature_selection.html',
                           year = year, from_month = from_month, to_month = to_month, currency=currency ,labels=labels,
                           price_values=price_values, volatility_values = volatility_values, length=length,
                           volatility_gradients_values = volatility_gradients_values, gradients_values=gradients_values)

@app.route("/anomalies/detectlofmapper", methods = ['POST', 'GET'])
def detect_lof_mapper():
    year, from_month, to_month,currency, status = local_outlier_factor.detect_lof_mapper()
    return render_template('anomalies/local_outlier_factor_mapper.html',
                           year = year, from_month = from_month, to_month = to_month, currency= currency)

@app.route("/anomalies/detectlofreducer", methods = ['POST', 'GET'])
def detect_lof_reducer():
    year, from_month, to_month, currency, features = local_outlier_factor_reducer.detect_lof_reducer()
    return render_template('anomalies/local_outlier_factor_reducer.html',
                           year=year, from_month=from_month, to_month=to_month, currency=currency,
                           features = features.to_html())

@app.route("/anomalies/detectanomalies", methods = ['POST', 'GET'])
def detect_anomalies():
    year, from_month, to_month, currency, anomalies = anomaly_identification.detect_anomalies()
    return render_template('anomalies/anomalies.html',
                           year=year, from_month=from_month, to_month=to_month, currency=currency,
                           anomalies = anomalies.to_html())

@app.route("/anomalies/plotresults", methods = ['POST', 'GET'])
def plot_results():
    anomalies_result_visualization.plot_results()
    return render_template('anomalies/get_input.html')
"""