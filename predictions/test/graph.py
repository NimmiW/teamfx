import json
import plotly.plotly.plotly as py

def grp():
    graphs = [
        dict(
            data=[
                dict(
                    x=[1,2,3],
                    y=[2,4,5],
                    type='scatter',
                    legendgroup= "G1",
                    name= 'A'
                ),
                dict(
                    x=[8, 2, 3],
                    y=[11, 8, 3],
                    type='scatter',
                    legendgroup="G1",
                    name='B'
                ),
                dict(
                    x=[5, 5, 5],
                    y=[2, 4, 5],
                    type='scatter',
                    legendgroup="G2",
                    name='C'
                ),
                dict(
                    x=[1, 2, 5],
                    y=[11, 8, 3],
                    type='scatter',
                    legendgroup="G3",
                    name='D'
                ),
            ],
            layout=dict(
                title='Anomalies',
            )
        )
    ]

    # Add "ids" to each of the graphs to pass up to the client
    # for templating
    ids = ['graph-{}'.format(i) for i, _ in enumerate(graphs)]

    # Convert the figures to JSON
    # PlotlyJSONEncoder appropriately converts pandas, datetime, etc
    # objects to their JSON equivalents
    graphJSON = json.dumps(graphs, cls=py.utils.PlotlyJSONEncoder)


    return  ids, graphJSON
