from flask import Flask, render_template, make_response,request, session
from brd import plot_data,black_region_detection
from anomalies import feature_selection, local_outlier_factor, local_outlier_factor_reducer, anomaly_identification
from anomalies import anomalies_result_visualization
from anomalies.visualize import visualize as anomlies_visualize

from anomalies.evaluator import evaluator_hours
import anomalies.config as config

from backtesting.backtester import application

from optimization.Strategies import StrategyOptimizer_MA
from optimization.Strategies import StrategyOptimizer_MACD
from optimization.Strategies import Strategy_Optimizer_Bollinger
from optimization.Strategies import Strategy_Optimizer_Stochastic
from optimization.Strategies import Strategy_Optimizer_RSI
from optimization.Strategies import Strategy_Optimizer_FMA
from optimization import Risk_Calculator
from optimization import signal_Generator
from optimization import evaluate_optimization
from optimization import gen_parameters


import plotly.plotly.plotly as py
import pandas as pd
import numpy as np
import json

app = Flask(__name__)
app.config['SESSION_TYPE'] = 'memcached'
app.config['SECRET_KEY'] = 'super secret key'



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

@app.route("/app", methods = ['GET','POST'])
def overall_backtesting():

    blackregion = request.form["blackregions"]
    if(blackregion == "true"):
        ids, graphJSON, returns, idsreturn, returngraphJSON = application.app(True)
    else:
        ids, graphJSON, returns, idsreturn, returngraphJSON= application.app()

    return render_template('backtesting/visualize.html',status="with_data",
                           ids=ids,
                           graphJSON=graphJSON, returns = returns, returngraphJSON = returngraphJSON, idsreturn= idsreturn
                        )

@app.route("/backtesting/visualize", methods = ['GET','POST'])
def postInput():
    ids, graphJSON,returns,idsreturn, returngraphJSON   = application.app()
    returns = (returns.tail(200)).to_html()
    # session ['returngraphJSON'] =returngraphJSON
    # session ['idsreturn'] = idsreturn
    # print(ids)
    # print(' print(idsreturn)')
    # print(idsreturn)
    # print('print(ids)')
    # print(ids)
    return render_template('backtesting/visualize.html',
                           status="with_data",
                           ids=ids,
                           graphJSON=graphJSON, returns = returns, returngraphJSON = returngraphJSON, idsreturn= idsreturn )

@app.route("/backtesting/returnChart", methods = ['GET','POST'])
def returnChart():
    # graph,returns = application.app()
    returns = request.args.get('returns')
    return render_template('backtesting/returnChart.html', returns= returns)

@app.route("/backtesting/equityChart", methods = ['GET','POST'])
def equityChart():
    # graph,returns = application.app()
    returngraphJSON = request.args.get('returngraphJSON')
    idsreturn = request.args.get('idsreturn')
    print("idsreturn")
    print(idsreturn)
    return render_template('backtesting/equityChart.html', status="with_data",ids=[idsreturn], graphJSON = returngraphJSON )

@app.route("/backtesting/evaluate", methods=['GET'])
def showevaluate():
    return render_template('backtesting/evaluate.html')

@app.route("/backtesting/evaluation", methods=['GET','POST'])
def evaluate():
    sharp_ratio, cagr,max_daily_drawdown,graph,ids = application.app(False,"Evaluate")
    # print(sharp_ratio)
    # print(cagr)
    return render_template('backtesting/evaluation.html',status="with_data",sharp_ratio = sharp_ratio,cagr = cagr,
                           max_daily_drawdown =max_daily_drawdown,graphJSON=graph, ids = ids )





#--------------------------------------------------------optimization Routes----------------------------------------------------#
@app.route("/optimization/", methods = ['POST', 'GET'])
def load_index_page():
    return render_template('optimization/index.html')

@app.route("/optimization/input", methods = ['POST', 'GET'])
def load_optimize_interface():
    return render_template('optimization/optimize_interface.html')

@app.route("/optimization/strategyOptimizer", methods = ['POST', 'GET'])
def optimize():
    strategy = request.form['strategy']
    returns = []
    if(strategy == 'Moving Average'):
        strategyNum = 1
        returns = StrategyOptimizer_MA.initialize()
    elif (strategy == 'MACD'):
        strategyNum = 2
        returns = StrategyOptimizer_MACD.initialize()
    elif (strategy == 'Bollinger Band'):
        strategyNum = 3
        returns = Strategy_Optimizer_Bollinger.initialize()
    elif (strategy == 'Stochastic'):
        strategyNum = 4
        returns = Strategy_Optimizer_Stochastic.initialize()
    elif (strategy == 'RSI'):
        strategyNum = 5
        returns = Strategy_Optimizer_RSI.initialize()
    elif (strategy == 'Fuzzy Moving Average'):
        strategyNum = 6
        returns = Strategy_Optimizer_FMA.initialize()

    top10 = returns[:10]
    results = [(Risk_Calculator.calculateRisk(x[1],strategy), x, strategyNum) for x in top10]
    return render_template('optimization/Results.html',results=results)

@app.route('/plotChart', methods=['POST','GET'])
def plot_chart():
    data = request.form.get('custId')
    strategy = data[1:len(data) - 1][-1:len(data)]
    para = data.split('[')
    para = para[2][:len(para) - 9].split(',')
    para = [int(i) for i in para]
    ids, graphJSON = signal_Generator.app(para,strategy)

    return render_template('optimization/plot.html',status="with_data",ids=ids,graphJSON=graphJSON,para=para,strategy=strategy)

@app.route("/optimization/evaluate", methods = ['POST', 'GET'])
def load_optimize_eval_interface():
    return render_template('optimization/optimize_eval_interface.html')

@app.route("/optimization/evaluation_optimizer", methods = ['POST', 'GET'])
def opt_evaluation_results():
    strategy = request.form['strategy']
    print("strategy:",strategy)
    para = gen_parameters.getPara(strategy);
    print("parA",para)
    results = evaluate_optimization.calculateRisk(para,strategy)
    print(results)
    return render_template('optimization/eva_optimize_results.html',results=results,para=para)

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
