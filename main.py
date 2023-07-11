import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import statsmodels.api as sm
import numpy as np

# Titulo do app
st.title('Ações App')

# Barra lateral 1
st.sidebar.title('Selecione o stock')
ticker_symbol = st.sidebar.text_input('stock', 'AAPL', max_chars=10)

# Baixar dados 
data = yf.download(ticker_symbol, start='2020-01-01', end = '2023-06-26')

# Exibir os dados
st.subheader('Histórico')
st.dataframe(data)

# Exibir Gráfico
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y = data['Close'], name = 'Fechamento'))
fig.update_layout(title = f"{ticker_symbol}", xaxis_title = "Data", yaxis_title = "Preço")
st.plotly_chart(fig)

# Barra lateral para o segundo gráfico
st.sidebar.title('Selecione o stock - Gráfico 2')
ticker_symbol2 = st.sidebar.text_input('stock2', 'MSFT', max_chars=10)

# Baixar dados para o segundo gráfico
data2 = yf.download(ticker_symbol2, start='2020-01-01', end='2023-06-26')

# Exibir os dados do segundo gráfico
st.subheader('Histórico - Gráfico 2')
st.dataframe(data2)

# Exibir Gráfico de Fechamento para o segundo gráfico
fig_close2 = go.Figure()
fig_close2.add_trace(go.Scatter(x=data2.index, y=data2['Close'], name='Fechamento'))
fig_close2.update_layout(title=f"{ticker_symbol2}", xaxis_title="Data", yaxis_title="Preço")
st.plotly_chart(fig_close2)

# Calcular IRF para o segundo gráfico
returns2 = np.log(data2['Close']).diff().dropna()
model2 = sm.tsa.VAR(returns2)
results2 = model2.fit(maxlags=10, ic='aic')
irf2 = results2.irf(10)

# Exibir Gráfico de IRF para o segundo gráfico
fig_irf2 = go.Figure()
for i in range(len(returns2.columns)):
    fig_irf2.add_trace(go.Scatter(x=irf2.irfperiods, y=irf2.irfs[:, i, i], name=returns2.columns[i]))
fig_irf2.update_layout(title='Impulse Response Function - Gráfico 2', xaxis_title='Período', yaxis_title='IRF')
st.plotly_chart(fig_irf2)