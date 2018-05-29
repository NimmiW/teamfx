from flask import Flask,redirect, url_for, request
import pandas as pd
import os
import gc

def detect_lof_reducer():

    if os.path.exists("static/anomalies/merged_local_outlier_factor_file.csv"):
        os.remove("static/anomalies/merged_local_outlier_factor_file.csv")
    fout=open("static/anomalies/merged_local_outlier_factor_file.csv","a")

    # first file:
    print('Process begin')
    for line in open("static/anomalies/local_outlier_factor0.csv"):
        fout.write(line)
        print(line)

    # now the rest:
    for num in range(1,6):
        f = open("static/anomalies/local_outlier_factor"+str(num)+".csv")
        lines = f.readlines()[1:]
        for line in lines:
            print(line)
            fout.write(line)
        f.close()

    fout.close()
    gc.collect()
    features = pd.read_csv('static/anomalies/features.csv')

    return request.form["year"], request.form["from_month"], request.form["to_month"], request.form["currency"], features.head(100)