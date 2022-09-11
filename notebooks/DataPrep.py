# %%
import SMOTE as SMOTE
import pandas as pd

# %%
dados = pd.read_csv('data/Disease-Prediction/Training.csv')
dados = dados.drop(columns=['Unnamed: 133'], axis=1)

# %%
atributos = dados.drop(columns=['prognosis'], axis=1)
classes = dados['prognosis']

classes_quantia = classes.value_counts()

# %%
resampler = SMOTE()
atributos, classes = resampler.fit_resample(atributos, classes)
