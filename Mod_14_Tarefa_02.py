import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import os
import sys

def plota_fig(df, value, index, func, ylabel, xlabel, opcao='nada'):
    if opcao=='nada':
        pd.pivot_table(df, values=value, index=index, aggfunc=func).plot(figsize=[15,5])
    elif opcao=='unstack':
        pd.pivot_table(df, values=value, index=index, aggfunc=func).unstack().plot(figsize=[15,5])
    elif opcao=='sort':
        pd.pivot_table(df, values=value, index=index, aggfunc=func).sort_values(value).plot(figsize=[15,5])
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    return None

mes = sys.argv[1]

sinasc = pd.read_csv('./input/SINASC_RO_2019_'+mes+'.csv')

max_data = sinasc.DTNASC.max()[:7]
print(max_data)

os.makedirs('./output/figs/'+max_data, exist_ok=True)

plota_fig(sinasc,'IDADEMAE','DTNASC','mean','média idade mãe por data', 'data nascimento')
plt.savefig('./output/figs/'+max_data+'/média idade mãe por data')

plota_fig(sinasc,'PESO','ESCMAE','median','peso mediano bebê', 'escolaridade mãe', 'sort')
plt.savefig('./output/figs/'+max_data+'/peso mediano por escolaridade mãe')

plota_fig(sinasc,'IDADEMAE',['DTNASC','SEXO'],'mean','média idade mãe', 'data nascimento', 'unstack')
plt.savefig('./output/figs/'+max_data+'/média idade mãe por sexo')

plota_fig(sinasc,'PESO',['DTNASC','SEXO'],'mean','média peso bebê', 'data nascimento', 'unstack')
plt.savefig('./output/figs/'+max_data+'/média peso bebê por sexo')

plota_fig(sinasc,'APGAR1','GESTACAO','mean','apgard médio', 'gestação', 'sort')
plt.savefig('./output/figs/'+max_data+'/média idade mãe por data')