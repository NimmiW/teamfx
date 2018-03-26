from flask import Flask, render_template, make_response
from brd import black_region_detection
from anomalies import feature_selection, local_outlier_factor, local_outlier_factor_reducer, anomaly_identification


app = Flask(__name__)

@app.route('/')
def hello_world():
  return 'Hello, World!'


#--------------------------------------------------------Test Routes----------------------------------------------------#
@app.route('/html')
def html():
    return render_template('hello.html')

@app.route('/chart/<pair>')
def chart(pair):
    return render_template('chart.html', pair = pair)

@app.route("/brd/<num>")
def brd(num):
    return render_template('brd/brd.html', black_regions = black_region_detection.detect(num))

#--------------------------------------------------------anomalies Routes----------------------------------------------------#

@app.route("/anomalies/selectfeatures")
def feature_selecion():
    return render_template('anomalies/feature_selection.html', status = feature_selection.feature_selecion())

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
