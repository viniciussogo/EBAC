import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px

sns.set(context='talk', style='ticks')

st.set_page_config(
     page_title="Projeto Previsão de Renda",
     page_icon=":💸:",
     layout="wide",
)

st.write('# Análise exploratória da previsão de renda')

renda = pd.read_csv('./input/previsao_de_renda.csv')

#plots
fig = px.box(renda, x="sexo", y="renda", title="Distribuição da Renda por Sexo",
             labels={"sexo": "Sexo", "renda": "Renda"},
             color="sexo",
             color_discrete_map={"M": "#1f77b4", "F": "#ff7f0e"})


st.title("Análise da Relação entre Renda e Sexo")
st.plotly_chart(fig)



fig, ax = plt.subplots(7,1,figsize=(12,90))
renda[['posse_de_imovel','renda']].plot(kind='hist', ax=ax[0])
st.write('## Gráficos ao longo do tempo')
sns.lineplot(x='data_ref',y='renda', hue='posse_de_imovel',data=renda, ax=ax[1])
ax[1].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref',y='renda', hue='qtd_filhos',data=renda, ax=ax[2])
ax[2].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref',y='renda', hue='tipo_renda',data=renda, ax=ax[3])
ax[3].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref',y='renda', hue='educacao',data=renda, ax=ax[4])
ax[4].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref',y='renda', hue='estado_civil',data=renda, ax=ax[5])
ax[5].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref',y='renda', hue='tipo_residencia',data=renda, ax=ax[6])
ax[6].tick_params(axis='x', rotation=45)
sns.despine()
st.pyplot(plt)

st.write('## Gráficos bivariada')
fig, ax = plt.subplots(6,1,figsize=(10,60))
sns.barplot(x='posse_de_imovel',y='renda',data=renda, ax=ax[0])
sns.barplot(x='qtd_filhos',y='renda',data=renda, ax=ax[1])
sns.barplot(x='tipo_renda',y='renda',data=renda, ax=ax[2])
sns.barplot(x='educacao',y='renda',data=renda, ax=ax[3])
sns.barplot(x='estado_civil',y='renda',data=renda, ax=ax[4])
sns.barplot(x='tipo_residencia',y='renda',data=renda, ax=ax[5])
sns.despine()
st.pyplot(plt)




