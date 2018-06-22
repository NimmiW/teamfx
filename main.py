from flask import Flask, render_template, make_response,request
from brd import plot_data,black_region_detection
from anomalies import feature_selection, local_outlier_factor, local_outlier_factor_reducer, anomaly_identification
from anomalies import anomalies_result_visualization
from anomalies.visualize import visualize as anomlies_visualize

from anomalies.evaluator import evaluator_hours
import anomalies.config as config

from backtesting.backtester import application

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

    year, from_month, to_month, currency, length,ids, graphJSON\
        = feature_selection.feature_selecion()
    #eturn render_template('anomalies/feature_selection.html',
    #                       year = year, from_month = from_month, to_month = to_month, currency=currency ,labels=labels,
    #                       price_values=price_values, volatility_values = volatility_values, length=length,
    #                       volatility_gradients_values = volatility_gradients_values, gradients_values=gradients_values)

    print(ids)
    print(graphJSON)
    print("under")
    return render_template('anomalies/feature_selection.html',
                           year=year, from_month=from_month, to_month=to_month, currency=currency,
                           ids=ids, graphJSON=graphJSON)

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
    threshold = config.ANOMALY_PERCENTAGE
    nneighbours =  config.NEAREST_NEIGHBOURS
    year, from_month, to_month, currency, anomalies = anomaly_identification.detect_anomalies()

    anomalies = anomalies[['Ranking_Factor']]

    return render_template('anomalies/anomalies.html',
                           year=year, from_month=from_month, to_month=to_month, currency=currency,
                           anomalies = anomalies.to_html()
                           )

@app.route("/anomalies/plotresults", methods = ['POST', 'GET'])
def plot_results():
    anomalies_result_visualization.plot_results()
    return render_template('anomalies/get_input.html')

@app.route("/anomalies/visualize", methods = ['POST', 'GET'])
def visualize_anormalies():
    if (request.form['page'] == 'anomalies_visualize_page'):

        threshold = config.ANOMALY_PERCENTAGE
        nneighbours = config.NEAREST_NEIGHBOURS
        ids, graphJSON = anomlies_visualize.get_visualize_view(threshold,nneighbours)
        print(ids)
        print(graphJSON)

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

@app.route("/anomalies/evaluate", methods = ['POST', 'GET'])
def visualize_anormalies_with_data():

    print("on")

    ids, graphJSON = evaluator_hours.PR_curve_visualize()
    print(ids)
    print(graphJSON)
    print("under")

    return render_template('anomalies/evaluate.html',
                           status="with_data",
                           ids=ids,
                           graphJSON=graphJSON)

@app.route("/anomalies/evaluate/confusion_matrix",  methods = ['POST', 'GET'])
def show_cpnfusion_matrix_anomalies():
    threshold = request.form["threshold"]
    year = request.form["year"]
    currency = request.form["currency"]
    nneighbours = 2
    print(threshold)
    print(year)
    print(currency)
    results = evaluator_hours.show_evaluate_results(threshold, nneighbours, year, currency)
    print(results.to_json(orient='index'))
    return results.to_json(orient='index')


"""@app.route("/anomalies/visualize/graph", methods = ['POST', 'GET'])
def visualize_anormalies_with_data():
    return render_template('anomalies/visualize_with_data.html')"""
#--------------------------------------------------------bactesting Routes----------------------------------------------------#
@app.route("/backtesting/", methods = ['POST', 'GET'])
def backtesting():
    # anomalies_result_visualization.plot_results()
    return render_template('backtesting/index.html')

@app.route("/backtesting/input", methods = ['POST','GET'])
def backtestingInput():
    # anomalies_result_visualization.plot_results()
    return render_template('backtesting/input.html')

@app.route("/backtesting/visualize", methods = ['GET','POST'])
def postInput():
    graph = application.app()
    ids, graphJSON = application.app()

    return render_template('backtesting/visualize.html',
                           status="with_data",
                           ids=ids,
                           graphJSON=graphJSON)

# @app.route("/backtesting/evaluate", methods=['GET'])
# def evaluate():
#
#     return render_template('backtesting/evaluate.html')
#     application.app()


#--------------------------------------------------------optimization Routes----------------------------------------------------#



#--------------------------------------------------------predictions Routes----------------------------------------------------#



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
