from flask import Flask, render_template, make_response
from brd import black_region_detection
from anomalies import feature_selection, local_outlier_factor, local_outlier_factor_reducer, anomaly_identification


app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/login')
def login():
    return render_template('login.html')


#--------------------------------------------------------Test Routes----------------------------------------------------#

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
    return render_template('anomalies/feature_selection.html', status=feature_selection.feature_selecion())

@app.route("/anomalies/detectlofmapper")
def detect_lof_mapper():
    return render_template('anomalies/local_outlier_factor_mapper.html', status = local_outlier_factor.detect_lof_mapper())

@app.route("/anomalies/detectlofreducer")
def detect_lof_reducer():
    features = local_outlier_factor_reducer.detect_lof_reducer()
    return render_template('anomalies/local_outlier_factor_reducer.html', features = features.to_html())

@app.route("/anomalies/detectanomalies")
def detect_anomalies():
    anomalies = anomaly_identification.detect_anomalies()
    return render_template('anomalies/anomalies.html', anomalies = anomalies.to_html())


#--------------------------------------------------------bactesting Routes----------------------------------------------------#



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
