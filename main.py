from flask import Flask, render_template, make_response,request
from brd import plot_data,black_region_detection
from anomalies import feature_selection, local_outlier_factor, local_outlier_factor_reducer, anomaly_identification
from anomalies import anomalies_result_visualization
from anomalies.visualize import visualize as anomlies_visualize
from predictions.predict import predict_arima_ann
from predictions.test.hello_world import fun
from predictions.test.graph import grp
from predictions.evaluate import evaluate as evaluate_predictions
import plotly.plotly.plotly as py
import pandas as pd
import numpy as np
import json

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/login')
def login():
    return render_template('login.html')


#--------------------------------------------------------Test Routes----------------------------------------------------#

@app.route('/brd/plotly')
def plotly():
    ids, graphJSON = plot_data.plot_function()

    return render_template('brd/index.html',
                           ids=ids,
                           graphJSON=graphJSON)

@app.route("/brd/<num>")
def brd(num):
    labels, values = black_region_detection.detect(num)
    print(labels)
    print(values)
    return render_template('brd/brd.html', labels = labels, values = values)


@app.route('/brd/charts')
def charts():
    return render_template('brd/charts.html')

@app.route('/brd/charts_1')
def brd_():
    #labels = ["2012-01-01 00:00:00","2012-02-01 00:10:00","2012-03-01","2012-04-01","2012-04-03","2012-04-08","2012-07-01","2012-08-01",
    #          "2012-09-01","2012-10-01","2012-11-01","2012-12-01"]
    #values = [10000, 30162, 26263, 18394, 18287, 28682, 31274, 33259, 25849, 24159, 32651, 31984, 38451]
    #length=len(values)
    labels, values, length = black_region_detection.get_data()
    print(labels)
    print(values)
    return render_template('brd/brd.html', labels = labels, values = values, length=length)

@app.route('/brd/tables')
def tables():
    return render_template('brd/tables.html')

#--------------------------------------------------------anomalies Routes----------------------------------------------------#

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
    #eturn render_template('anomalies/feature_selection.html',
    #                       year = year, from_month = from_month, to_month = to_month, currency=currency ,labels=labels,
    #                       price_values=price_values, volatility_values = volatility_values, length=length,
    #                       volatility_gradients_values = volatility_gradients_values, gradients_values=gradients_values)
    return render_template('anomalies/feature_selection.html',
                           year=year, from_month=from_month, to_month=to_month, currency=currency, labels=labels,
                           price_values=price_values, length=length,
                           volatility_gradients_values=volatility_gradients_values)

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

@app.route("/anomalies/visualize", methods = ['POST', 'GET'])
def visualize_anormalies():
    if (request.form['page'] == 'anomalies_visualize_page'):

        ids, graphJSON = anomlies_visualize.get_visualize_view()

        return render_template('anomalies/visualize.html',
                               status = "with_data",
                               ids=ids,
                               graphJSON=graphJSON)
    else:
        return render_template('anomalies/visualize.html',
                               status = "without_data",
                               ids=["no_id"],
                               graphJSON=[]
                               )



"""@app.route("/anomalies/visualize/graph", methods = ['POST', 'GET'])
def visualize_anormalies_with_data():
    return render_template('anomalies/visualize_with_data.html')"""
#--------------------------------------------------------bactesting Routes----------------------------------------------------#



#--------------------------------------------------------optimization Routes----------------------------------------------------#



#--------------------------------------------------------predictions Routes----------------------------------------------------#
@app.route("/predictions/")
def prediction_index():
    return render_template('predictions/index.html')

@app.route("/predictions/input")
def prediction_inputs():
    return render_template('predictions/get_input.html')


@app.route("/predictions/predict", methods = ['POST', 'GET'])
def prediction_predict():

    print("inputs")

    print(request.form["currency"])
    prediction_type = request.form["prediction_type"]
    #date_time = "2012-10 -31 02:14:00"
    date_time = request.form["date"]+' '+request.form["time"]+':00'
    #count = int(request.form["count"])
    ids, graphJSON = predict_arima_ann(date_time, prediction_type)
    return render_template('predictions/predicted_graph.html', ids=ids,
                           graphJSON=graphJSON)

    """return render_template('anomalies/feature_selection.html',
                           year=year, from_month=from_month, to_month=to_month, currency=currency, labels=labels,
                           price_values=price_values, length=length,
                           volatility_gradients_values=volatility_gradients_values)"""

@app.route("/predictions/predict/result")
def prediction_graph():
    ids, graphJSON = grp()
    return render_template('predictions/predicted_graph.html',  ids=ids,
                           graphJSON=graphJSON)

@app.route("/predictions/evaluate")
def prediction_evaluate():
    return render_template('predictions/evaluate.html')

@app.route("/predictions/predict/evaluation_results")
def prediction_evaluate_result():

    results = evaluate_predictions.read_results()
    return results.to_json(orient='index')

#----------------------------------------------------------------------------------------------------------------------------
@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


if __name__ == '__main__':
  app.run(debug=True)
