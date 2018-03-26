import pandas as pd

def detect_lof_reducer():
    fout=open("static/anomalies/merged_local_outlier_factor_file.csv","a")

    # first file:
    print('Process begin')
    for line in open("static/anomalies/local_outlier_factor0.csv"):
        fout.write(line)
        print(line)

    # now the rest:
    for num in range(1,5):
        f = open("static/anomalies/local_outlier_factor"+str(num)+".csv")
        lines = f.readlines()[1:]
        for line in lines:
            print(line)
            fout.write(line)
        f.close()

    fout.close()

    features = pd.read_csv('static/anomalies/features.csv')
    return features