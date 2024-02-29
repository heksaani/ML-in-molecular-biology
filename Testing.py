#!/usr/bin/env python
# coding: utf-8

# ## <center>Pew Research Poll on Presidential Vote Preference<br>August, 2020</center>

# Data for each observed characteristic of the statistical  population is represented as a Sunburst chart, with the respective characteristic as root, and its categories as level 1 nodes. Each level 1 node has two children: one for Trump, and another for Biden. The level 3 nodes (the leafs) correspond to subgroups that expressed  strong support for Trump, respectively for Biden.

# In[1]:


import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode,  iplot
init_notebook_mode(connected=True)


# Set sector colors according to number of level 1 nodes:

# In[2]:


colors2 = ['#ffffff', '#aaaaaa',  '#dddddd'] + ['#C0223B', '#000096']*2 + ['#C0223B', '#000096']*2 
colors3 = ['#ffffff', '#777777','#aaaaaa',  '#dddddd'] +['#C0223B', '#000096']*3 + ['#C0223B', '#000096']*3
colors4 = ['#ffffff'] + ['#aaaaaa',  '#dddddd']*2 + ['#C0223B', '#000096']*4 + ['#C0223B', '#000096']*4
colors10 =['#ffffff'] + ['#777777','#aaaaaa', '#dddddd']*3 + ['#bbbbbb'] +  \
          ['#C0223B', '#000096']*10 + ['#C0223B', '#000096']*10 


def sunburst_data(df):
    if len(df.columns) != 7:
        raise valueError('Your df is not derived from Pew Research data')
       
    population_characteristic = df.columns[0]
    level1 = list(df[population_characteristic])
    n = len(level1)
    
    labels = [population_characteristic] + level1+   [df.columns[1], df.columns[3]]*n +\
             [df.columns[2], df.columns[4]]*n
    ids = [f'{k}' for k in range(1, len(labels)+1)]   
    level2_parents = []
    for k in range(2, n+2):
        level2_parents.extend([f'{k}', f'{k}'])
    parents = [''] + ['1']*n + level2_parents + [f'{k}' for k in range(n+2, 3*n+2)] 
    vals  =  df.values[:, 1:5]
    temp = np.copy(vals[:, 1])
    vals[:, 1] = vals[:, 2]
    vals[:, 2]= temp
    values = [1]*(n+1) + list(vals[:, :2].flatten()) + list(vals[:, 2:].flatten())
    text = labels[:n+1] + [f'{lab}<br>{val}%' for lab, val in zip(labels[n+1:], values[n+1:])]
    
    if n == 2:
        colors=colors2
    elif n == 3: 
        colors=colors3    
    elif n == 4: 
        colors=colors4
    elif n == 10:
        colors=colors10
    else:
        raise ValueError(f'Not implemented for {n} categories')       
    return go.Sunburst(labels=labels,
                       ids=ids,
                       parents=parents,
                       text=text,
                       textinfo='text',
                       hoverinfo='text',
                       values=values,
                       marker_colors=colors,
                       insidetextorientation='radial',
                       outsidetextfont_size=20
                       )


# In[3]:


fig = make_subplots(rows=3, cols=2,
                    vertical_spacing=0.005,
                    horizontal_spacing=0.005,
                    specs=[[{"type": "sunburst"}, {"type": "sunburst"}]]*3)

files = ['Gender.csv', 'Age.csv', 'USR.csv', 'Race-Ethnicity.csv', 'Education.csv', 'Religion.csv']
for k, f in enumerate(files):
    df = pd.read_csv('Data/'+f)
    j = k%2
    i = (k-j)//2
    fig.add_trace(sunburst_data(df), i+1, j+1)
fig.update_layout(title_text='Pew Research Poll on  Presidential Vote Preference<br>August, 2020', title_x=0.5,
                  font_family='Balto',
                  font_size=16,
                    height=1300, width=900,
                  margin_l=0
                 );  


# In[4]:


iplot(fig)