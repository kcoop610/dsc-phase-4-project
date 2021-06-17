def plot_vax(ts, title=None, labels=None, color=None):
    #plot timeseries in range
    fig = px.line(ts, color=color, title=title, range_x=('2021-02-12', '2021-07-21'), labels=labels)

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
    fig.update_layout(plot_bgcolor='#f2f2f2', height=600, width=1000)
    
    #save fig
    fig.write_image(f'./images/{title}.jpg')
    
    return fig



    