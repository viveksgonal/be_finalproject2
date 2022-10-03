from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
from plotly.graph_objs import Scatter
import os
import json
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from plotly.subplots import make_subplots
# from fbprophet import Prophet
import pickle

def home(request):
    world = pd.read_csv('carbon_be_project/meatcattleworldco.csv')
    data = dict(
        type = 'choropleth',
        colorscale='greens',
        # reversescale = True,
        locations = world['Area'],
        locationmode = "country names",
        z = world['Value'],
        text = world['Area'],
        colorbar = {'title' : 'CO2 Emission due to Beef Cattle'},
        # wscale=False
        # autocolorscale=True
      ) 

    layout = dict(height=580,margin={"r":50,"t":10,"l":50,"b":50},
                geo = dict(showframe = False,projection = {'type':'natural earth'})
             )
    choromap = go.Figure(data = [data],layout = layout)
     
    #fig5 = iplot(choromap)
    fig5 = choromap.to_html()
    df2000 = pd.read_csv('carbon_be_project/dataset2000.csv')
    df2017 = pd.read_csv('carbon_be_project/dataset2017.csv')
    fig6 = make_subplots(1, 2, specs=[[{'type':'domain'}, {'type':'domain'}]],
                        subplot_titles=['2000', '2020'])
    fig6.add_trace(go.Pie(labels=df2000['Item'], values=df2000['Value'], scalegroup='one',
                        name="CO2 2000"), 1, 1)
    fig6.add_trace(go.Pie(labels=df2017['Item'], values=df2017['Value'], scalegroup='one',
                        name="CO2 2020"), 1, 2)
    fig6.update_layout(title_text='Carbon Emission in Meat Industry')
    plot_div6 = plot(fig6, output_type='div', include_plotlyjs=False)
    context = {'fig5': fig5,'plot_div6': plot_div6}    
    return render(request,'index.html', context)

def predict(model_arima,years):
    test = pickle.load(open(model_arima,'rb'))
    forecast = test.predict(start=1,end = 50+years,typ = 'levels').rename('Forecast')
    buff1 = [] 
    for x in range(0,50+years):
        buff1.append(x)
    buff1 = np.array(buff1)
    forecast.index=buff1  

    buf1 = [] 
    for x in range(1961, 2011+years):
        buf1.append(x)
    buf = np.array(buf1)
    
    # providing an index
    ser = pd.Series(buf)
    df = pd.DataFrame({'Year':ser, 'Value':forecast})
    df['Year'] = df['Year'].astype(str) + '/12/31'
    df['Year'] = df['Year'].str.replace('/','-')
    df['Year'] = pd.to_datetime(df['Year'])
    return df

def forecast_accuracy(forecast, actual):
    mape = round(np.mean(np.abs(forecast - actual)/np.abs(actual)),4)  # MApe
    rmse = round(np.mean((forecast - actual)**2)**.5,4)  # RMSE
    return mape,rmse

#country - 3 letter code items - 
def getPredictions(years,items,country):
    result = predict('carbon_be_project/arima-models/'+country+'_'+items+'.pkl',years)
    return result

def getData(items,country):
    dffao = pd.read_csv('carbon_be_project/datasets/'+country+'_'+items+'.csv')
    dffao['Year'] = dffao['Year'].astype(str) + '/12/31'
    dffao['Year'] = dffao['Year'].str.replace('/','-')
    dffao['Year'] = pd.to_datetime(dffao['Year'])
    dffao['Value'] = round(dffao['Value']/1000,2)
    dffao.drop(['Domain','Area','Element','Item'],axis=1,inplace=True)
    return dffao

def input_predict(request):    
    return render(request,'input_predict.html')

def visualize(request):
    return render(request,'visualize.html')

def solution(request):
    return render(request,'solution.html')
    
def result(request):
    years = int(request.GET['years'])
    items = request.GET['typeOfItem']    
    country = request.GET['country']
    result = getPredictions(years+12,items,country)
    actual = getData(items,country)
    result=pd.DataFrame(result)
    actual=pd.DataFrame(actual)
    forecast = getPredictions(0,items,country)
    mape,rmse = forecast_accuracy(forecast.Value,actual.Value.iloc[0:51])
    acc_list = [mape,rmse]
    img = plot({'data':[Scatter(x=result['Year'], y=result['Value'],mode='lines+markers', name='Predicted Data', opacity=0.8, marker_color='blue'),Scatter(x=actual['Year'], y=actual['Value'],mode='lines+markers', name='Actual Data', opacity=0.8, marker_color='red')],'layout': {'xaxis': {'title': 'Year'}, 'yaxis': {'title': 'Carbon Emission (in Mg)'},'height':600}}, output_type='div')
    result['Year'] = result['Year'].dt.strftime('%Y')
    result['actualdata']=round(actual['Value'],2)
    result['Value'] = round(result['Value'],2)
    json_records = result.reset_index().to_json(orient ='records')
    data = []
    data = json.loads(json_records)
    context = {'d': data,'img':img,'items':items,'acc_list':acc_list}
    return render(request, 'result.html', context)