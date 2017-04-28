# Uncertainty Quality


Fatores de variação:
1) Dataset: MNIST com classes inside e outside
2) Inferência: Maximum Likelihood ou MAP, Dropout, VI, SGLD ou SGHMC
3) Treino: De acordo com o padrão da literatura para cada técnica
4) Arquitetura: Linear, MLP, Conv
Análise:
1) Incerteza: Anomalia, Calibração
2) ANOVA com e sem fator de acurácia de teste
Hipóteses:
1) Qual é o melhor método de inferência para incerteza?
2) Quem é mais importante, acurácia ou método?
Fatores em aberto:
1) Outros datasets
2) OSBA
3) Classes unknown com label distribuição uniforme
