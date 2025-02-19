import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

st.title('Meu primeiro App')

def carregar_dados():
    """Faz o upload de um arquivo CSV e retorna um DataFrame."""
    arquivo = st.file_uploader("Faça upload do arquivo CSV", type=["csv"])
    if arquivo is not None:
        return pd.read_csv(arquivo)
    return None

def mostrar_estatisticas(df):
    """Mostra estatísticas descritivas do DataFrame."""
    st.write(df.describe())

def mostrar_head(df):
    """Mostra as primeiras linhas do DataFrame."""
    st.write(df.head())

def visualizar_matriz_correlacao(df):
    """Exibe um heatmap de correlação."""
    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    st.pyplot(plt)

def gerar_histograma(df, coluna):
    """Gera um histograma para uma coluna específica."""
    plt.figure(figsize=(8,5))
    sns.histplot(df[coluna], bins=30, kde=True)
    st.pyplot(plt)

def gerar_boxplot(df, coluna):
    """Gera um boxplot para uma coluna específica."""
    plt.figure(figsize=(8,5))
    sns.boxplot(y=df[coluna])
    st.pyplot(plt)

def gerar_distribuicao(df, coluna):
    """Gera um gráfico de distribuição para uma coluna."""
    plt.figure(figsize=(8,5))
    sns.kdeplot(df[coluna], fill=True)
    st.pyplot(plt)

def visualizar_linha_tempo(df, coluna):
    """Gera um gráfico de linha para séries temporais."""
    plt.figure(figsize=(12,6))
    plt.plot(df.index, df[coluna], marker='o', linestyle='-')
    plt.xlabel('Índice')
    plt.ylabel(coluna)
    plt.title(f'Série Temporal: {coluna}')
    st.pyplot(plt)

def normalizar_dados(df, colunas):
    """Normaliza colunas usando MinMaxScaler."""
    scaler = MinMaxScaler()
    df[colunas] = scaler.fit_transform(df[colunas])
    return df

def padronizar_dados(df, colunas):
    """Padroniza colunas usando StandardScaler."""
    scaler = StandardScaler()
    df[colunas] = scaler.fit_transform(df[colunas])
    return df

def prever_series_temporais(df, coluna, ordem=(5,1,0)):
    """Treina um modelo ARIMA e faz previsões."""
    modelo = ARIMA(df[coluna], order=ordem)
    modelo_fit = modelo.fit()
    previsoes = modelo_fit.forecast(steps=10)
    return previsoes

def mostrar_colunas(df):
    """Exibe as colunas disponíveis no DataFrame."""
    return df.columns.tolist()

def escolher_coluna(df):
    """Permite ao usuário escolher uma coluna do DataFrame."""
    return st.selectbox("Escolha uma coluna", df.columns.tolist())

def aplicar_filtro(df, coluna, valor):
    """Filtra o DataFrame baseado em um valor específico de uma coluna."""
    return df[df[coluna] == valor]

def calcular_media(df, coluna):
    """Calcula a média de uma coluna."""
    return df[coluna].mean()

def calcular_mediana(df, coluna):
    """Calcula a mediana de uma coluna."""
    return df[coluna].median()

def calcular_desvio_padrao(df, coluna):
    """Calcula o desvio padrão de uma coluna."""
    return df[coluna].std()

def converter_para_datetime(df, coluna):
    """Converte uma coluna para formato datetime."""
    df[coluna] = pd.to_datetime(df[coluna])
    return df

def contar_valores_unicos(df, coluna):
    """Conta valores únicos em uma coluna."""
    return df[coluna].nunique()

def app():
    st.title("Análise Exploratória de Dados")
    df = carregar_dados()
    if df is not None:
        opcoes = [
            "Mostrar estatísticas", "Visualizar head", "Matriz de correlação", "Histograma",
            "Boxplot", "Distribuição", "Série temporal", "Normalizar dados", "Padronizar dados",
            "Previsão ARIMA", "Filtrar dados", "Calcular média", "Calcular mediana", "Calcular desvio padrão",
            "Converter para datetime", "Contar valores únicos"
        ]
        escolha = st.sidebar.selectbox("Escolha uma função", opcoes)
        
        if escolha == "Mostrar estatísticas":
            mostrar_estatisticas(df)
        elif escolha == "Visualizar head":
            mostrar_head(df)
        elif escolha == "Matriz de correlação":
            visualizar_matriz_correlacao(df)
        elif escolha == "Histograma":
            coluna = escolher_coluna(df)
            gerar_histograma(df, coluna)
        elif escolha == "Boxplot":
            coluna = escolher_coluna(df)
            gerar_boxplot(df, coluna)
        elif escolha == "Distribuição":
            coluna = escolher_coluna(df)
            gerar_distribuicao(df, coluna)
        elif escolha == "Série temporal":
            coluna = escolher_coluna(df)
            visualizar_linha_tempo(df, coluna)
        elif escolha == "Normalizar dados":
            colunas = st.multiselect("Escolha colunas", df.columns.tolist())
            df = normalizar_dados(df, colunas)
        elif escolha == "Padronizar dados":
            colunas = st.multiselect("Escolha colunas", df.columns.tolist())
            df = padronizar_dados(df, colunas)
        elif escolha == "Previsão ARIMA":
            coluna = escolher_coluna(df)
            previsoes = prever_series_temporais(df, coluna)
            st.write(previsoes)
        elif escolha == "Filtrar dados":
            coluna = escolher_coluna(df)
            valor = st.text_input("Digite o valor para filtrar")
            if valor:
                df_filtrado = aplicar_filtro(df, coluna, valor)
                st.write(df_filtrado)
        elif escolha == "Calcular média":
            coluna = escolher_coluna(df)
            st.write(calcular_media(df, coluna))
    else:
        st.warning("Por favor, carregue um arquivo CSV.")
if __name__ == "__main__":
    app()
