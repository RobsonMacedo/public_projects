import re
import pickle
from datetime import date
import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import requests                  # [handles the http interactions](http://docs.python-requests.org/en/master/) 
from bs4 import BeautifulSoup    # beautiful soup handles the html to text conversion and more
import re                        # regular expressions are necessary for finding the crumb (more on crumbs later)
from datetime import datetime    # string to datetime object conversion
from time import mktime          # mktime transforms datetime objects to unix timestamps

def _get_crumbs_and_cookies(stock):
    """
    get crumb and cookies for historical data csv download from yahoo finance
    
    parameters: stock - short-handle identifier of the company 
    
    returns a tuple of header, crumb and cookie
    """
    
    url = 'https://finance.yahoo.com/quote/{}/history'.format(stock)
    with requests.session():
        header = {'Connection': 'keep-alive',
                   'Expires': '-1',
                   'Upgrade-Insecure-Requests': '1',
                   'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) \
                   AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36'
                   }
        
        website = requests.get(url, headers=header)
        soup = BeautifulSoup(website.text, 'lxml')
        crumb = re.findall('"CrumbStore":{"crumb":"(.+?)"}', str(soup))

        return (header, crumb[0], website.cookies)
    
def convert_to_unix(date):
    """
    converts date to unix timestamp
    
    parameters: date - in format (dd-mm-yyyy)
    
    returns integer unix timestamp
    """
    datum = datetime.strptime(date, '%Y-%m-%d')
    
    return int(mktime(datum.timetuple()))

def load_csv_data(stock, interval='1d', day_begin='01-03-2018', day_end='28-03-2018'):
    """
    queries yahoo finance api to receive historical data in csv file format
    
    parameters: 
        stock - short-handle identifier of the company
        
        interval - 1d, 1wk, 1mo - daily, weekly monthly data
        
        day_begin - starting date for the historical data (format: dd-mm-yyyy)
        
        day_end - final date of the data (format: dd-mm-yyyy)
    
    returns a list of comma seperated value lines
    """
    lista = []
    day_begin_unix = convert_to_unix(day_begin)
    day_end_unix = convert_to_unix(day_end)
    
    header, crumb, cookies = _get_crumbs_and_cookies(stock)
    
    with requests.session():
        url = 'https://query1.finance.yahoo.com/v7/finance/download/' \
              '{stock}?period1={day_begin}&period2={day_end}&interval={interval}&events=history&crumb={crumb}' \
              .format(stock=stock, day_begin=day_begin_unix, day_end=day_end_unix, interval=interval, crumb=crumb)
                
        website = requests.get(url, headers=header, cookies=cookies)
        
        
        lista =  website.text.split('\n')[:-1]
        return lista

papeis = []
parar = 's'
while parar not in 'Nn':
    acao = str(input('Digite o código de um papel: '))
    padrao = re.compile(r'\w{4}\d+')
    while padrao.match(acao) is None:
        acao = str(input('Digite o código de um papel válido: '))
    acao= str.upper(acao) # carrega o nome do papel de forma mais amigável
    acao = acao+'.SA'
    papeis.append(acao)
    parar = str(input('Deseja incluir mais algum papel? (S/N) ... ')).strip().upper()[0]
    while parar not in 'SsNn':
        parar = str(input('Entrada inválida! Deseja incluir mais algum papel? (S/N) ... ')).strip().upper()[0] 
        
for acao in papeis:
    #Carregando a base de dados
    base = load_csv_data(acao, interval='1d', day_begin='2011-01-01', day_end=str(date.today()))
    colunas = base[0].split(',')
    dados = base[1:]
    
    print(f'Tratando dados para {acao} ... ')
    
    # tratando dados
    lista = []
    for i in range(len(dados)):
        lista.append(dados[i].split(','))
    
    parametros = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    for i in range(len(parametros)):
        dados = []
        for j in range(len(lista)):
            dados.append(lista[j][i])
        parametros[i] = dados
    dados = {'Date':parametros[0], 'Open':parametros[1], 'High': parametros[2], 'Low':parametros[3], 'Close':parametros[4], 'Adj Close':parametros[5], 'Volume':parametros[6]}
    base = pd.DataFrame(dados, columns=colunas)
    base.drop(base[base.Close =='null'].index, inplace=True)
    
    
    
    #base = pd.read_csv(acao+'.csv', sep = ';'
    # converte as strings para float
    def acerta_base(base):
        lista = []
        parametros = ['Open', 'Close']
        for i in parametros:
            lista.clear()
            for j in base[i]:
                lista.append(float(j))
            base[i] = lista
    
    acerta_base(base)
    
    # Adicionando coluna das previsoes
    base['variacao'] = base.Open-base.Close
    
    
    def cria_situacao():
        lista = []
        for i in base.variacao:
            if (i > 0):
                lista.append("COMPRA")
            else:
                lista.append("VENDA")
        base['situacao'] = lista
    cria_situacao()
    
    
    #Criando as médias móveis
    def cria_mm9():
        lista = []
        lista.clear()
        ini = 0
        fim = 9
        for i in range(0, 9):
            lista.append(base.iloc[i, 4])
        for i in range(9, len(base.Close)):
            lista.append(base.iloc[ini:fim, 4].mean())
            ini += 1
            fim += 1
        base['mm9'] = lista
    cria_mm9()
    
    def cria_mm21():
        lista = []
        lista.clear()
        ini = 0
        fim = 21
        for i in range(0, 21):
            lista.append(base.iloc[i, 4])
        for i in range(21, len(base.Close)):
            lista.append(base.iloc[ini:fim, 4].mean())
            ini += 1
            fim += 1
        base['mm21'] = lista
    cria_mm21()
    
    
    p1 = base.iloc[-9:,4].mean()
    p2 = base.iloc[-21:,4].mean()
    
        
    def cria_situacao2():
        lista = []
        for i in range(len(base.Close)):
            if base.iloc[i, 4]>base.iloc[i,9] and base.iloc[i, 4]>base.iloc[i,10]:
                lista.append('COMPRA')
            else:
                lista.append('VENDA')
        base['situacao2'] = lista
    cria_situacao2()
    
    
    X = base.iloc[:, [9, 10]].values
    y2 = base.iloc[:,-1].values
    
    #Tratando as datas
    # resolver isso aqui
    '''def tratando_datas(datas):
        lista = []
        for i in range(len(datas)):
            #print(type(i))
            lista.append(datetime.datetime.strptime(datas[i], "%d/%m/%Y").date())
        base['nova_data'] = lista
        return base    
    
    tratando_datas(base.Date)'''
       
    
    #Caso queira escalonar
    #scaler = StandardScaler(with_mean = False)
    #X = scaler.fit_transform(X)
    
    #criando treino e teste para os diferentes cenários
    #X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size = 0.2)
    X_treino, X_teste, y2_treino, y2_teste = train_test_split(X, y2, test_size = 0.2)
    
    #Calculando previsoes
    print(f'Calculando previsões para {acao} ... ')
        
    #Trabalhando com classificação
    #naive bayes
    for i in range(40):
        
        kfold = StratifiedKFold(n_splits=20, shuffle=True, random_state = i)
        resultados_naive = []
        matrizes = []
        for id_treino, id_teste in kfold.split(X, np.zeros(shape=(X.shape[0], 1))):
            classificador_naive = GaussianNB()
            classificador_naive.fit(X[id_treino], y2[id_treino])
            predicao = classificador_naive.predict(X[id_teste])
            resultados_naive.append(accuracy_score(y2[id_teste], predicao))
            
    media_naive = np.array(resultados_naive).mean()
    desvio_naives = np.array(resultados_naive).std()
    
    # arvores de decisão
    for i in range(40):
        
        kfold = StratifiedKFold(n_splits=20, shuffle=True, random_state = i)
        resultados_arvores = []
        matrizes = []
        for id_treino, id_teste in kfold.split(X, np.zeros(shape=(X.shape[0], 1))):
            classificador_arvores = DecisionTreeClassifier()
            classificador_arvores.fit(X[id_treino], y2[id_treino])
            predicao = classificador_arvores.predict(X[id_teste])
            resultados_arvores.append(accuracy_score(y2[id_teste], predicao))
    
    media_decision_tree = np.array(resultados_arvores).mean()
    desvio_decision_tree = np.array(resultados_arvores).std()
    
    # Random_forest
    for i in range(40):
        
        kfold = StratifiedKFold(n_splits=20, shuffle=True, random_state = i)
        resultados_RFC = []
        matrizes = []
        for id_treino, id_teste in kfold.split(X, np.zeros(shape=(X.shape[0], 1))):
            classificador_RFC = RandomForestClassifier(n_estimators=10)
            classificador_RFC.fit(X[id_treino], y2[id_treino])
            predicao = classificador_RFC.predict(X[id_teste])
            resultados_RFC.append(accuracy_score(y2[id_teste], predicao))
      
    media_random_forest = np.array(resultados_RFC).mean()
    desvio_random_forest = np.array(resultados_RFC).std()      
    
    
    # KNN
    for i in range(40):
        
        kfold = StratifiedKFold(n_splits=20, shuffle=True, random_state = i)
        resultados_KNN_classifier = []
        matrizes = []
        for id_treino, id_teste in kfold.split(X, np.zeros(shape=(X.shape[0], 1))):
            classificador_KNN_classifier = KNeighborsClassifier()
            classificador_KNN_classifier.fit(X[id_treino], y2[id_treino])
            predicao = classificador_KNN_classifier.predict(X[id_teste])
            resultados_KNN_classifier.append(accuracy_score(y2[id_teste], predicao))
    
    media_knn = np.array(resultados_KNN_classifier).mean()
    desvio_knn = np.array(resultados_KNN_classifier).std() 
    
    # SVC
    
    for i in range(5):
        
        kfold = StratifiedKFold(n_splits=20, shuffle=True, random_state = i)
        resultados_SVC = []
        matrizes = []
        for id_treino, id_teste in kfold.split(X, np.zeros(shape=(X.shape[0], 1))):
            classificador_SVC = SVC(gamma = 'auto', probability = True)
            classificador_SVC.fit(X[id_treino], y2[id_treino])
            predicao = classificador_SVC.predict(X[id_teste])
            resultados_SVC.append(accuracy_score(y2[id_teste], predicao))
            
    media_svc = np.array(resultados_SVC).mean()
    desvio_svc = np.array(resultados_SVC).std() 
    
    
    # acertar esta parte.... criar loop e colocar as médias de acerto no txt
    classificadores = {'Naive':[classificador_naive, media_naive], 'Arvores': [classificador_arvores, media_decision_tree], 
                       'RFC': [classificador_RFC, media_random_forest], 'KNN': [classificador_KNN_classifier, media_knn],
                       'SVC': [classificador_SVC, media_svc]}
    #Passando os classificadores e gravando no txt
    soma = 0
    # Setando a hora atual para gravar no arquivo
    now = datetime.now()
    time = now.strftime("%H:%M:%S")    
    for i, k in classificadores.items():
        if k[0].predict([[p1, p2]])[0] == 'COMPRA':
            soma +=1
        resultados = open(acao+'.txt', 'a')
        resultados.write(f'{date.today()}, {time},  ' +acao+', '+i+f', {k[0].predict([[p1, p2]])[0]}, '+base.iloc[-1, -1])
        resultados.write(f', {k[1]}, ')
        resultados.write(f', {k[0].predict_proba([[p1, p2]])} \n')
        resultados.close()
    
    if soma >= 3:
        consolidado = open('consolidado.txt', 'a')
        consolidado.write(f'{date.today()}, {time}, ' +acao+', '+i+f', {k[0].predict([[p1, p2]])[0]}')
        consolidado.write(f', {k[1]}, ')
        consolidado.write(f', Probabilidade: , {k[0].predict_proba([[p1, p2]])}, COMPRA \n')
        consolidado.close()
    else:
        consolidado = open('consolidado.txt', 'a')
        consolidado.write(f'{date.today()}, {time},  ' +acao+', '+i+f', {k[0].predict([[p1, p2]])[0]}')
        consolidado.write(f', {k[1]}, ')
        consolidado.write(f', Probabilidade: , {k[0].predict_proba([[p1, p2]])}, VENDE \n')
        consolidado.close()
           
            
    print('Dados gravados em', acao+'.txt')
    print('Dados gravados em consolidado.txt')
    print('\n')
print('Previsões geradas com sucesso!')


