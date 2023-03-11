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
        colorscale='reds',
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

def globalpredict(model_arima,years):
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

def getGlobalPredictions(years,country):
    result = globalpredict('carbon_be_project/arima-models-cumu-con/'+country+'.pkl',years)
    return result

def getData(items,country):
    dffao = pd.read_csv('carbon_be_project/datasets/'+country+'_'+items+'.csv')
    dffao['Year'] = dffao['Year'].astype(str) + '/12/31'
    dffao['Year'] = dffao['Year'].str.replace('/','-')
    dffao['Year'] = pd.to_datetime(dffao['Year'])
    dffao['Value'] = round(dffao['Value']/1000,2)
    #dffao.drop(['Domain','Area','Element','Item'],axis=1,inplace=True)
    return dffao

def getGlobalData(country):
    dffao = pd.read_csv('carbon_be_project/arima-models-cumu-con/'+country+'.csv')
    dffao['Year'] = dffao['Year'].astype(str) + '/12/31'
    dffao['Year'] = dffao['Year'].str.replace('/','-')
    dffao['Year'] = pd.to_datetime(dffao['Year'])
    dffao['Value'] = round(dffao['Value']/1000,2)
    #dffao.drop(['Domain','Area','ValueK'],axis=1,inplace=True)
    return dffao

def input_predict(request):    
    return render(request,'input_predict.html')

def input_compare(request):    
    return render(request,'input_compare.html')

def visualize(request):
    return render(request,'visualize.html')

def solution(request):
    return render(request,'solution.html')

def compare(request):
    years = int(request.GET['years'])
    country1 = request.GET['country1']
    country2 = request.GET['country2']
    typeOfItem = request.GET['typeOfItem']
    meat = request.GET['meat']
    gl_con = request.GET['gl_con']
    # print(request.GET['#country1'])
    if(meat=='indi' and gl_con=='gl_c'):
        actual1 = getData(typeOfItem,'gl')
        actual2 = getData(typeOfItem,country1)
        result1 = getPredictions(years+7,typeOfItem,'gl')
        result2 = getPredictions(years+7,typeOfItem,country1)
        result1=pd.DataFrame(result1)
        result2=pd.DataFrame(result2)
        actual1=pd.DataFrame(actual1)
        actual2=pd.DataFrame(actual2)
        result = pd.DataFrame(result1)
        forecast1 = getPredictions(0,typeOfItem,'gl')
        mape1,rmse1 = forecast_accuracy(forecast1.Value,actual1.Value.iloc[0:51])
        forecast2 = getPredictions(0,typeOfItem,country1)
        mape2,rmse2 = forecast_accuracy(forecast2.Value,actual2.Value.iloc[0:51])
        acc_list = [mape1,rmse1,mape2,rmse2]
        actual1=actual1.iloc[0:57]
        actual2=actual2.iloc[0:57]
        img2 = plot({'data':[Scatter(x=result1['Year'].iloc[57:], y=result1['Value'].iloc[57:],mode='lines+markers', name='Predicted Global Data', opacity=0.8, marker_color='blue'),Scatter(x=result2['Year'].iloc[57:], y=result2['Value'].iloc[57:],mode='lines+markers', name='Predicted Country Data', opacity=0.8, marker_color='red')],'layout': {'xaxis': {'title': 'Year'}, 'yaxis': {'title': 'Carbon Emission (in Mg)'},'height':600,'legend':dict(yanchor="top", y=1.145, xanchor="left", x=0.4),'title':'Predicted Carbon Emission'}}, output_type='div')
        img1 = plot({'data':[Scatter(x=actual1['Year'], y=actual1['Value'],mode='lines+markers', name='Actual Global Data', opacity=0.8, marker_color='blue'),Scatter(x=actual2['Year'], y=actual2['Value'],mode='lines+markers', name='Actual Country Data', opacity=0.8, marker_color='red')],'layout': {'xaxis': {'title': 'Year'}, 'yaxis': {'title': 'Carbon Emission (in Mg)'},'height':600,'legend':dict(yanchor="top", y=1.145, xanchor="left", x=0.4),'title':'Actual Carbon Emission'}}, output_type='div')
        trace1 = go.Bar(x=result1["Year"].iloc[57:], y=result1['Value'].iloc[57:],name='Predicted Global Data')
        trace2 = go.Bar(x=result2["Year"].iloc[57:], y=result2['Value'].iloc[57:],name='Predicted Country Data')
        layout1 = go.Layout(title="Predicted Carbon Emission", xaxis=dict(title="Year"),yaxis=dict(title="Value"),legend=dict(y=1.2,x=0.4,traceorder='normal'))
        data1 = [trace1,trace2]
        fig1 = go.Figure(data=data1, layout=layout1)
        img4 = plot(fig1, output_type='div', include_plotlyjs=False)
        trace3 = go.Bar(x=actual1["Year"], y=actual1['Value'],name='Actual Global Data')
        trace4 = go.Bar(x=actual2["Year"], y=actual2['Value'],name='Actual Country Data')
        layout2 = go.Layout(title="Actual Carbon Emission", xaxis=dict(title="Year"),yaxis=dict(title="Value"),legend=dict(y=1.2,x=0.4,traceorder='normal'))
        data2 = [trace3,trace4]
        fig2 = go.Figure(data=data2, layout=layout2)
        img3 = plot(fig2, output_type='div', include_plotlyjs=False)
        result['Year'] = result1['Year'].dt.strftime('%Y')
        # #result['actualdata']=round(actual['Value'],2)
        result['PGlobal'] = round(result2['Value'],2)
        result['PCountry'] = round(result1['Value'],2)
        result['AGlobal'] = round(actual2['Value'],2)
        result['ACountry'] = round(actual1['Value'],2)
        
    if(meat=='cumu' and gl_con=='gl_c'):
        actual1 = getGlobalData('gl')
        actual2 = getGlobalData(country1)
        result1 = getGlobalPredictions(years+7,'gl')
        result2 = getGlobalPredictions(years+7,country1)    
        #create dataframes of both dataframes
        result1=pd.DataFrame(result1)
        result2=pd.DataFrame(result2)
        result = pd.DataFrame(result1)
        actual1=pd.DataFrame(actual1)
        actual2=pd.DataFrame(actual2)
        forecast1 = getGlobalPredictions(0,'gl')
        mape1,rmse1 = forecast_accuracy(forecast1.Value,actual1.Value.iloc[0:51])
        forecast2 = getGlobalPredictions(0,country1)
        mape2,rmse2 = forecast_accuracy(forecast2.Value,actual2.Value.iloc[0:51])
        acc_list = [mape1,rmse1,mape2,rmse2]
        actual2=actual2.iloc[0:57]
        actual1=actual1.iloc[0:57]
        img2 = plot({'data':[Scatter(x=result1['Year'].iloc[57:], y=result1['Value'].iloc[57:],mode='lines+markers', name='Predicted Country Data', opacity=0.8, marker_color='blue'),Scatter(x=result2['Year'].iloc[57:], y=result2['Value'].iloc[57:],mode='lines+markers', name='Predicted Global Data', opacity=0.8, marker_color='red')],'layout': {'xaxis': {'title': 'Year'}, 'yaxis': {'title': 'Carbon Emission (in Mg)'},'height':600,'legend':dict(yanchor="top", y=1.145, xanchor="left", x=0.4),'title':'Predicted Carbon Emission'}}, output_type='div')
        img1 = plot({'data':[Scatter(x=actual1['Year'], y=actual1['Value'],mode='lines+markers', name='Actual Country Data', opacity=0.8, marker_color='blue'),Scatter(x=actual2['Year'], y=actual2['Value'],mode='lines+markers', name='Actual Global Data', opacity=0.8, marker_color='red')],'layout': {'xaxis': {'title': 'Year'}, 'yaxis': {'title': 'Carbon Emission (in Mg)'},'height':600,'legend':dict(yanchor="top", y=1.145, xanchor="left", x=0.4),'title':'Actual Carbon Emission'}}, output_type='div')
        # img4 = plot({'data':[Scatter(x=result1['Year'].iloc[60:], y=result1['Value'].iloc[60:],mode='bar', name='Country Data', opacity=0.8),Scatter(x=result2['Year'].iloc[60:], y=result2['Value'].iloc[60:],mode='bar', name='Global Data', opacity=0.8)],'layout': {'xaxis': {'title': 'Year'}, 'yaxis': {'title': 'Carbon Emission (in Mg)'},'height':600,'legend':dict(yanchor="top", y=1.145, xanchor="left", x=0.4)}}, output_type='div')
        # img3 = plot({'data':[Scatter(x=actual1['Year'], y=actual1['Value'],mode='bar', name='Actual Global Data', opacity=0.8),Scatter(x=actual2['Year'], y=actual2['Value'],mode='bar', name='Actual Country Data', opacity=0.8)],'layout': {'xaxis': {'title': 'Year'}, 'yaxis': {'title': 'Carbon Emission (in Mg)'},'height':600,'legend':dict(yanchor="top", y=1.145, xanchor="left", x=0.4)}}, output_type='div')
        #FOR BAR CHART
        trace1 = go.Bar(x=result1["Year"].iloc[57:], y=result1['Value'].iloc[57:],name='Predicted Global Data')
        trace2 = go.Bar(x=result2["Year"].iloc[57:], y=result2['Value'].iloc[57:],name='Predicted Country Data')
        layout1 = go.Layout(title="Predicted Carbon Emission", xaxis=dict(title="Year"),yaxis=dict(title="Value"),legend=dict(y=1.2,x=0.4,traceorder='normal'))
        data1 = [trace1,trace2]
        fig1 = go.Figure(data=data1, layout=layout1)
        img4 = plot(fig1, output_type='div', include_plotlyjs=False)
        
        trace3 = go.Bar(x=actual1["Year"], y=actual1['Value'],name='Actual Global Data')
        trace4 = go.Bar(x=actual2["Year"], y=actual2['Value'],name='Actual Country Data')
        layout2 = go.Layout(title="Actual Carbon Emission", xaxis=dict(title="Year"),yaxis=dict(title="Value"),legend=dict(y=1.2,x=0.4,traceorder='normal'))
        data2 = [trace3,trace4]
        fig2 = go.Figure(data=data2, layout=layout2)
        img3 = plot(fig2, output_type='div', include_plotlyjs=False)
        result['Year'] = result1['Year'].dt.strftime('%Y')
        # #result['actualdata']=round(actual['Value'],2)
        result['PGlobal'] = round(result2['Value'],2)
        result['PCountry'] = round(result1['Value'],2)
        result['AGlobal'] = round(actual2['Value'],2)
        result['ACountry'] = round(actual1['Value'],2)

    if(meat=='indi' and gl_con=='c_c'):
        actual1 = getData(typeOfItem,country1)
        actual2 = getData(typeOfItem,country2)
        result1 = getPredictions(years+7,typeOfItem,country1)
        result2 = getPredictions(years+7,typeOfItem,country2)
        result1=pd.DataFrame(result1)
        result2=pd.DataFrame(result2)
        actual1=pd.DataFrame(actual1)
        actual2=pd.DataFrame(actual2)
        result = pd.DataFrame()
        forecast1 = getPredictions(0,typeOfItem,country1)
        mape1,rmse1 = forecast_accuracy(forecast1.Value,actual1.Value.iloc[0:51])
        forecast2 = getPredictions(0,typeOfItem,country2)
        print(mape1,rmse1)
        mape2,rmse2 = forecast_accuracy(forecast2.Value,actual2.Value.iloc[0:51])
        acc_list = [mape1,rmse1,mape2,rmse2]
        img2 = plot({'data':[Scatter(x=result1['Year'].iloc[57:], y=result1['Value'].iloc[57:],mode='lines+markers', name='Predicted Country 1 Data', opacity=0.8, marker_color='blue'),Scatter(x=result2['Year'].iloc[57:], y=result2['Value'].iloc[57:],mode='lines+markers', name='Predicted Country 2 Data', opacity=0.8, marker_color='red')],'layout': {'xaxis': {'title': 'Year'}, 'yaxis': {'title': 'Carbon Emission (in Mg)'},'height':600,'legend':dict(yanchor="top", y=1.145, xanchor="left", x=0.4),'title':'Predicted Carbon Emission'}}, output_type='div')
        img1 = plot({'data':[Scatter(x=actual1['Year'], y=actual1['Value'],mode='lines+markers', name='Actual Country 1 Data', opacity=0.8, marker_color='blue'),Scatter(x=actual2['Year'], y=actual2['Value'],mode='lines+markers', name='Actual Country 2 Data', opacity=0.8, marker_color='red')],'layout': {'xaxis': {'title': 'Year'}, 'yaxis': {'title': 'Carbon Emission (in Mg)'},'height':600,'legend':dict(yanchor="top", y=1.145, xanchor="left", x=0.4),'title':'Actual Carbon Emission'}}, output_type='div')
        trace1 = go.Bar(x=result1["Year"].iloc[57:], y=result1['Value'].iloc[57:],name='Predicted Country 1 Data')
        trace2 = go.Bar(x=result2["Year"].iloc[57:], y=result2['Value'].iloc[57:],name='Predicted Country 2 Data')
        layout1 = go.Layout(title="Predicted Carbon Emission", xaxis=dict(title="Year"),yaxis=dict(title="Value"),legend=dict(y=1.2,x=0.4,traceorder='normal'))
        data1 = [trace1,trace2]
        fig1 = go.Figure(data=data1, layout=layout1)
        img4 = plot(fig1, output_type='div', include_plotlyjs=False)
        trace3 = go.Bar(x=actual1["Year"], y=actual1['Value'],name='Actual Country 1 Data')
        trace4 = go.Bar(x=actual2["Year"], y=actual2['Value'],name='Actual Country 2 Data')
        layout2 = go.Layout(title="Actual Carbon Emission", xaxis=dict(title="Year"),yaxis=dict(title="Value"),legend=dict(y=1.2,x=0.4,traceorder='normal'))
        data2 = [trace3,trace4]
        fig2 = go.Figure(data=data2, layout=layout2)
        img3 = plot(fig2, output_type='div', include_plotlyjs=False)
        result['Year'] = result1['Year'].dt.strftime('%Y')
        # #result['actualdata']=round(actual['Value'],2)
        result['PGlobal'] = round(result2['Value'],2)
        result['PCountry'] = round(result1['Value'],2)
        result['AGlobal'] = round(actual2['Value'],2)
        result['ACountry'] = round(actual1['Value'],2)

    if(meat=='cumu' and gl_con=='c_c'):
        actual1 = getGlobalData(country1)
        actual2 = getGlobalData(country2)
        result1 = getGlobalPredictions(years+7,country1)
        result2 = getGlobalPredictions(years+7,country2)    
        #create dataframes of both dataframes
        result1=pd.DataFrame(result1)
        result2=pd.DataFrame(result2)
        result = pd.DataFrame(result1)
        actual1=pd.DataFrame(actual1)
        actual2=pd.DataFrame(actual2)
        forecast1 = getGlobalPredictions(0,country1)
        mape1,rmse1 = forecast_accuracy(forecast1.Value,actual1.Value.iloc[0:51])
        forecast2 = getGlobalPredictions(0,country2)
        mape2,rmse2 = forecast_accuracy(forecast2.Value,actual2.Value.iloc[0:51])
        acc_list = [mape1,rmse1,mape2,rmse2]
        actual1 = actual1.iloc[0:57]
        actual2 = actual2.iloc[0:57]
        img2 = plot({'data':[Scatter(x=result1['Year'].iloc[57:], y=result1['Value'].iloc[57:],mode='lines+markers', name='Predicted Country 1 Data', opacity=0.8, marker_color='blue'),Scatter(x=result2['Year'].iloc[57:], y=result2['Value'].iloc[57:],mode='lines+markers', name='Predicted Country 2 Data', opacity=0.8, marker_color='red')],'layout': {'xaxis': {'title': 'Year'}, 'yaxis': {'title': 'Carbon Emission (in Mg)'},'height':600,'legend':dict(yanchor="top", y=1.145, xanchor="left", x=0.4),'title':'Predicted Carbon Emission'}}, output_type='div')
        img1 = plot({'data':[Scatter(x=actual1['Year'], y=actual1['Value'],mode='lines+markers', name='Actual Country 1 Data', opacity=0.8, marker_color='blue'),Scatter(x=actual2['Year'], y=actual2['Value'],mode='lines+markers', name='Actual Country 2 Data', opacity=0.8, marker_color='red')],'layout': {'xaxis': {'title': 'Year'}, 'yaxis': {'title': 'Carbon Emission (in Mg)'},'height':600,'legend':dict(yanchor="top", y=1.145, xanchor="left", x=0.4),'title':'Actual Carbon Emission'}}, output_type='div')
        # img4 = plot({'data':[Scatter(x=result1['Year'].iloc[60:], y=result1['Value'].iloc[60:],mode='bar', name='Country Data', opacity=0.8),Scatter(x=result2['Year'].iloc[60:], y=result2['Value'].iloc[60:],mode='bar', name='Global Data', opacity=0.8)],'layout': {'xaxis': {'title': 'Year'}, 'yaxis': {'title': 'Carbon Emission (in Mg)'},'height':600,'legend':dict(yanchor="top", y=1.145, xanchor="left", x=0.4)}}, output_type='div')
        # img3 = plot({'data':[Scatter(x=actual1['Year'], y=actual1['Value'],mode='bar', name='Actual Global Data', opacity=0.8),Scatter(x=actual2['Year'], y=actual2['Value'],mode='bar', name='Actual Country Data', opacity=0.8)],'layout': {'xaxis': {'title': 'Year'}, 'yaxis': {'title': 'Carbon Emission (in Mg)'},'height':600,'legend':dict(yanchor="top", y=1.145, xanchor="left", x=0.4)}}, output_type='div')
        #FOR BAR CHART
        trace1 = go.Bar(x=result1["Year"].iloc[57:], y=result1['Value'].iloc[57:],name='Predicted Country 1 Data')
        trace2 = go.Bar(x=result2["Year"].iloc[57:], y=result2['Value'].iloc[57:],name='Predicted Country 2 Data')
        layout1 = go.Layout(title="Predicted Carbon Emission", xaxis=dict(title="Year"),yaxis=dict(title="Value"),legend=dict(y=1.2,x=0.4,traceorder='normal'))
        data1 = [trace1,trace2]
        fig1 = go.Figure(data=data1, layout=layout1)
        img4 = plot(fig1, output_type='div', include_plotlyjs=False)
        
        trace3 = go.Bar(x=actual1["Year"], y=actual1['Value'],name='Actual Country 1 Data')
        trace4 = go.Bar(x=actual2["Year"], y=actual2['Value'],name='Actual Country 2 Data')
        layout2 = go.Layout(title="Actual Carbon Emission", xaxis=dict(title="Year"),yaxis=dict(title="Value"),legend=dict(y=1.2,x=0.4,traceorder='normal'))
        data2 = [trace3,trace4]
        fig2 = go.Figure(data=data2, layout=layout2)
        img3 = plot(fig2, output_type='div', include_plotlyjs=False)

        #Pie Chart for 2 Countries 
        fig3 = px.pie(values=actual1['ValueP'], names=actual1['MeatType'])
        fig4 = px.pie(values=actual2['ValueP'], names=actual2['MeatType'])
        img5 = plot(fig3, output_type='div', include_plotlyjs=False)
        img6 = plot(fig4, output_type='div', include_plotlyjs=False)



        result['Year'] = result1['Year'].dt.strftime('%Y')
        # #result['actualdata']=round(actual['Value'],2)
        result['PGlobal'] = round(result2['Value'],2)
        result['PCountry'] = round(result1['Value'],2)
        result['AGlobal'] = round(actual2['Value'],2)
        result['ACountry'] = round(actual1['Value'],2)
    
    #get country actual data
    

    json_records = result.reset_index().to_json(orient ='records')
    data = []
    data = json.loads(json_records)
    context = {'d': data,'img1':img1,'img2':img2,'img3':img3,'img4':img4,'img5':img5,'img6':img6,'acc_list':acc_list}
    return render(request, 'compare.html', context)

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