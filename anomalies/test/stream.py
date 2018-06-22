import plotly
import numpy as np
import plotly.plotly as py
import plotly.tools as tls
import plotly.graph_objs as go
plotly.tools.set_credentials_file(username='NimmiRashinika', api_key='kZbVdGmEOGobMEAu86le')

import plotly.plotly as py
import plotly.graph_objs as go

# trace0 = go.Scatter(
#     x=[1, 2, 3, 4],
#     y=[10, 15, 13, 17]
# )
# trace1 = go.Scatter(
#     x=[1, 2, 3, 4],
#     y=[16, 5, 11, 9]
# )
# data = [trace0, trace1]
#
# py.plot(data, filename = 'basic-line')

stream_ids = tls.get_credentials_file()['stream_ids']
print(stream_ids)