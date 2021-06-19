#import necessary packages
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.tsa.api as tsa
import itertools
import pandas as pd
import matplotlib.pyplot as plt



def plot_vax(ts, title=None, labels=None, color=None):
    '''
    Customized visualization for July 4, 2021 COVID-19 vaccine administration goal.
    
    Parameters:
    ts - pd.DataFrame with date in datetime format, set as index, and frequency defined
    title - str, title for the graph
    labels - dict, keys indicate existing terms and values indicate the text to replace them with
    color - str, column in ts to separate lines in the graph
    
    Outputs:
    Returns and displays plotly line graph
    
    '''
    #plot timeseries in range
    fig = px.line(ts, color=color, title=title, range_x=('2021-02-12', '2021-07-21'), labels=labels)

    #plot vertical line at 6/13/21 to indicate data observed by
    fig.add_trace(go.Scatter(
        x=['2021-06-13', '2021-06-13', '2021-06-13'],
        y=[0, 81, 78],
        mode="lines+text",
        line=go.scatter.Line(color='gray', dash='dash'),
        name='Data Observed as of 6/13',
        text=[None, "<-- Observed", 'Predicted -->'],
        textposition="top center",
        textfont={'size':10}))

    #plot horizontal line at goal of 70% first dose administered
    fig.add_trace(go.Scatter(
        x=[0, '2021-04-20', '2021-07-04'],
        y=[70, 70, 70],
        mode="lines+text",
        line=go.scatter.Line(color='black'),
        name="% goal - 70%",
        text=[None, "% goal - 70%", None],
        textposition="top center",
        textfont={'size':12}))

    #plot horizontal line at goal of 70% first dose administered
    fig.add_trace(go.Scatter(
        x=['2021-07-04', '2021-07-04'],
        y=[0, 70],
        mode="lines+text",
        line=go.scatter.Line(color='black'),
        name="Date goal - 7/4/21",
        text=[None, "Date goal - 7/4/21", None],
        textposition="top center",
        textfont={'size':12}))
    
    #styling
    fig.update_layout(plot_bgcolor='#f2f2f2', height=500, width=1000)
    
    #save fig
    fig.write_image(f'./images/{title}.jpg')
    
    return fig




def adftest(ts):
    '''
    Conduct Dickey-Fuller test and return results and stationarity assessment as a dataframe

    Parameters:
    ts - pd.DataFrame, with date in datetime format, set as index, and frequency defined

    Outputs:
    Returns results_df - pd.DataFrame comprised of test statistic, p-value, number of lags, number of observations, and 
                 boolean value assessing stationarity of data
    '''
    results = tsa.stattools.adfuller(ts)
    stats = ['Test Stat', 'p-value', 'k-lags', 'n-observations']
    results_dict = dict(zip(stats, results[:4]))
    results_dict['Stationary?'] = results_dict['p-value']<.05
    results_df = pd.DataFrame(results_dict, index=['AD Fuller Test'])
    return results_df




def stationarity_check(ts, column='Administered_Dose1_Recip_18PlusPop_Pct', window=8, color=None): 
    '''
    Test stationarity and visualize rolling mean and rolling standard deviation 

    Parameters:
    ts - pd.DataFrame, with date in datetime format, set as index, and frequency defined
    column (default='Administered_Dose1_Recip_18PlusPop_Pct') - str, name of target column in ts
    window (default=8) - int, number of observations used to calculate the statistic (mean, std)
    color (default=None) - str, name of column in ts to separate plots in visualization

    Outputs:
    Returns dataframe of Dickey-Fuller test results
    Displays multi-line plot of actual data, rolling mean, and rolling standard deviation
    '''
    # Calculate rolling statistics
    roll_mean = ts.rolling(window=window, center=False).mean()
    roll_std = ts.rolling(window=window, center=False).std()
    
    # Perform the Dickey Fuller Test
    dftest = adftest(ts)

    # Plot rolling statistics:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
                        x=ts.index,
                        y=ts[column],
                        mode='lines',
                        name='Actual',
                        line=go.scatter.Line(color='blue')))
    fig.add_trace(go.Scatter(
                        x=roll_mean.index,
                        y=roll_mean[column],
                        mode='lines',
                        name='Rolling Mean',
                        line=go.scatter.Line(color='orange', dash='dot')))
    fig.add_trace(go.Scatter(
                        x=roll_std.index,
                        y=roll_std[column],
                        mode='lines',
                        name='Rolling Standard Deviation',
                        line=go.scatter.Line(color='goldenrod', dash='dot')))

    #styling
    fig.update_layout(plot_bgcolor='#f2f2f2', height=600, width=1000,
                     title='Rolling Mean & Standard Deviation')

    display(dftest)
    fig.show()
    return dftest



def acf_pacf_plot(ts):
    '''
    Plots autocorrelation and partial autocorrelation
    
    Parameters:
     ts - pd.DataFrame, with date in datetime format, set as index, and frequency defined

    Outputs:
    Displays figure with two subplots:
        Autocorrelation plot - 
            Sharp drop after lag=k suggests an MA(k) model, k=q
            Correlation up until lag=k, then trailing off suggests an AR-k model, k=p
        Partial Autocorrelation plot - 
            Sharp drop after lag=k suggests an AR-k model, k=p
            Correlation up until lag=k, then trailing off suggests an MA-k model, k=q
    '''
    #plot ACF; cut=MA(q), trail=AR(p)
    fig, ax = plt.subplots(figsize=(16,3))
    tsa.graphics.plot_acf(ts, ax=ax, lags=50);
    #plot PACF; cut=AR(p), trail=MA(q)
    fig, ax = plt.subplots(figsize=(16,3))
    tsa.graphics.plot_pacf(ts, ax=ax, lags=50);



def grid_search_pdqs(ts_train, max_range, s):
    '''
    Grid search optimal p, d, q, and s terms for SARIMAX seasonal time series analysis model

    Inputs:
    ts_train - pd.DataFrame, training set time series with date in datetime format, set as index, and frequency defined
    max_range - int, top value to test for p, d, and q terms
    s - int, seasonal term

    Outputs:
    Returns results_df - pd.DataFrame describing results of the orders and seasonal orders tested, sorted by AIC ascending
    '''
    #set the range of p, d, and q terms to search based on PACF and ACF
    p = d = q = range(0,max_range) 
    #generate all different combinations of p, d and q orders
    pdq = list(itertools.product(p, d, q)) 
    #generate all different combinations of seasonal orders
    pdqs = [(x[0], x[1], x[2], s) for x in pdq] 

    #loop through the parameter combinations and collect AIC value
    results = []
    for comb in pdq:
        for combs in pdqs:
                model = tsa.SARIMAX(ts_train, order=comb, seasonal_order=combs,
                                    enforce_invertibility=False, enforce_stationarity=False)
                output = model.fit()
                results.append([comb, combs, output.aic])

    #convert results to a dataframe, sort by AIC smallest to largest
    results_df = pd.DataFrame(results, columns=['order', 'seasonal order', 'AIC']).sort_values('AIC')
    display(results_df)
    return results_df




def grid_search_pdq(ts_train, max_range):
    '''
    Grid search optimal p, d, and q terms for ARIMA time series analysis model

    Inputs:
    ts_train - pd.DataFrame, training set time series with date in datetime format, set as index, and frequency defined
    max_range - int, top value to test for p, d, and q terms

    Outputs:
    Returns results_df - pd.DataFrame describing results of the orders tested, sorted by AIC ascending
    '''
    #set the range of p, d, and q terms to search based on PACF and ACF
    p = d = q = range(0,max_range) 
    #generate all different combinations of p, d and q orders
    pdq = list(itertools.product(p, d, q)) 

    #loop through the parameter combinations and collect AIC value
    results = []
    for comb in pdq:
        model = tsa.arima.ARIMA(ts_train, order=comb, 
                                enforce_invertibility=False, enforce_stationarity=False)
        output = model.fit()
        results.append([comb, output.aic])

    #convert results to a dataframe, sort by AIC smallest to largest
    results_df = pd.DataFrame(results, columns=['order', 'AIC']).sort_values('AIC')
    display(results_df)
    return results_df




def validate_model(model, ts, steps=21, title=None, file_name=None):
    '''
    Use a given model to make predictions on the testing set, and graphically compare predictions to the actual values

    Inputs:
    model - model results wrapper item, trained and fit
    ts - pd.DataFrame, time series with date in datetime format, set as index, and frequency defined
    steps (default=21) - number of steps to predict; should be the same size as the testing set
    title (default=None) - str, optional title for graph
    file_name (default=None) - str, optional file name to save the figure as an image to the local 'images' folder
                                if None, image is not saved

    Outputs:
    Returns predictions and 95% confidence interval upper and lower bounds in a dataframe
    Displays line graph comparing predictions and confidence interval alongside actual values
    '''
    pred = model.get_forecast(steps=steps)
    pred_ci = pred.conf_int()
    pred_ci.columns = ['lower', 'upper']
    pred_ci['predictions'] = pred.predicted_mean
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
                        x=ts.index,
                        y=ts.Administered_Dose1_Recip_18PlusPop_Pct,
                        mode='lines',
                        name='Actual',
                        line=go.scatter.Line(color='black')))
    fig.add_trace(go.Scatter(
                        x=pred_ci.index,
                        y=pred_ci.predictions,
                        mode='lines',
                        name='Forecast',
                        line=go.scatter.Line(color='blue', dash='dot')))
    
    fig.add_trace(go.Scatter(
                        x=pred_ci.index,
                        y=pred_ci.upper,
                        mode='lines',
                        name='Upper CI',
                        line=go.scatter.Line(color='lightskyblue', dash='dot')))
    fig.add_trace(go.Scatter(
                        x=pred_ci.index,
                        y=pred_ci.lower,
                        mode='lines',
                        name='Lower CI',
                        line=go.scatter.Line(color='lightskyblue', dash='dot')))


    #styling
    fig.update_layout(plot_bgcolor='#f2f2f2', height=500, width=900,
                     title='Model Predictions vs Actual')
    
    #save image
    if file_name != None:
        fig.write_image(f'./images/{file_name}.jpg')

    fig.show()
    return pred_ci




def plot_forecast(model, ts, steps, title=None, file_name=None):
    '''
    Use a given model to forecast into the future, and graph alongside actual observations

    Inputs:
    model - model results wrapper item, trained and fit
    ts - pd.DataFrame, time series with date in datetime format, set as index, and frequency defined
    steps (default=21) - number of steps in the future to predict; should not be larger than the testing set
    title (default=None) - str, optional title for graph
    file_name (default=None) - str, optional file name to save the figure as an image to the local 'images' folder
                                if None, image is not saved

    Outputs:
    Returns the forecasted value for the target date, 7/4/21
    Displays line graph comparing predictions and 95% confidence interval alongside actual values
    '''
    # predict future values
    forecasted_future = model.get_forecast(steps=steps)
    # collect confidence interval for each predicted value
    forecasted_ci = forecasted_future.conf_int()
    # update column names
    forecasted_ci.columns = ['lower', 'upper']
    # combine into one dataframe
    forecasted_ci['predictions'] = forecasted_future.predicted_mean
    # print the value for the target date of 7/4/21
    print(f"Forecast for 7/4/21: {forecasted_ci.predictions['2021-07-04']}")
    
    fig = go.Figure()
    # plot actual observations
    fig.add_trace(go.Scatter(
                        x=ts.index,
                        y=ts.Administered_Dose1_Recip_18PlusPop_Pct,
                        mode='lines',
                        name='Actual',
                        line=go.scatter.Line(color='blue')))
    # plot predictions
    fig.add_trace(go.Scatter(
                        x=forecasted_ci.index,
                        y=forecasted_ci.predictions,
                        mode='lines',
                        name='Forecast',
                        line=go.scatter.Line(color='blue', dash='dot')))
    # plot confidence interval
    fig.add_trace(go.Scatter(
                        x=forecasted_ci.index,
                        y=forecasted_ci.upper,
                        mode='lines',
                        name='Upper CI',
                        line=go.scatter.Line(color='lightskyblue', dash='dot')))
    fig.add_trace(go.Scatter(
                        x=forecasted_ci.index,
                        y=forecasted_ci.lower,
                        mode='lines',
                        name='Lower CI',
                        line=go.scatter.Line(color='lightskyblue', dash='dot')))
    # shade the predicted region
    fig.add_shape(type='rect', 
                  xref='x', yref='y',
                 x0='2021-06-13',
                 y0=0,
                 x1='2021-07-04',
                 y1=70,
                 fillcolor='PaleTurquoise',
                 opacity=.2, layer='below')

    #plot vertical line at 6/13/21 to indicate data observed by
    fig.add_trace(go.Scatter(
                        x=['2021-06-13', '2021-06-13', '2021-06-13'],
                        y=[0, 81, 79],
                        mode="lines+text",
                        line=go.scatter.Line(color='gray', dash='dash'),
                        name='Data Observed as of 6/13',
                        text=[None, "<-- Observed", 'Predicted -->'],
                        textposition="top center",
                        textfont={'size':10}))

    #plot horizontal line at goal of 70% first dose administered
    fig.add_trace(go.Scatter(
                        x=['2021-02-13', '2021-04-20', '2021-07-04'],
                        y=[70, 70, 70],
                        mode="lines+text",
                        line=go.scatter.Line(color='black'),
                        name="% goal - 70%",
                        text=[None, "% goal - 70%", None],
                        textposition="top center",
                        textfont={'size':12}))

    #plot horizontal line at goal of 70% first dose administered
    fig.add_trace(go.Scatter(
                        x=['2021-07-04', '2021-07-04'],
                        y=[0, 70],
                        mode="lines+text",
                        line=go.scatter.Line(color='black'),
                        name="Date goal - 7/4/21",
                        text=[None, "Date goal - 7/4/21", None],
                        textposition="top center",
                        textfont={'size':12}))

    #styling
    fig.update_layout(plot_bgcolor='#f2f2f2', height=600, width=1000,
                     title=title)
    
    #save image
    if file_name != None:
        fig.write_image(f'./images/{file_name}.jpg')
    
    fig.show()