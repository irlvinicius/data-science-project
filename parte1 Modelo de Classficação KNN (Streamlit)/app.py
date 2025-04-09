import streamlit as st
import pandas as pd
import joblib

# Carregar o modelo e o scaler
modelo = joblib.load("./models_pkl/knn_model.pkl")
scaler = joblib.load("./models_pkl/normalizador.pkl")

# Coletar entrada do usuário
idade = st.slider("Idade", 10, 80, 30)
sexo_val = st.radio("Sexo", ["Masculino", "Feminino"])
sexo_val = 1 if sexo_val == "Masculino" else 0

min_espessura = 300.0
max_espessura = 700.0
espessura = st.number_input("Espessura Central (em microns)", min_value=min_espessura, max_value=max_espessura, step=1.0)
st.caption(f"Valores entre ({min_espessura} - {max_espessura})")

min_curvatura = 35.0
max_curvatura = 70.0
curvatura = st.number_input("Curvatura Máxima (em D)", min_value=min_curvatura, max_value=max_curvatura, step=0.1)
st.caption(f"Valores entre ({min_curvatura} - {max_curvatura})")

assimetria = st.slider("Assimetria Corneana", 0.0, 1.0, 0.3)

col1, col2, col3 = st.columns(3)

with col1:
    hist_val = st.radio("Histórico Familiar de Ceratocone", ["Sim", "Não"])
    hist_val = 1 if hist_val == "Sim" else 0

with col2:
    coceira_val = st.radio("Coceira nos Olhos Frequente", ["Sim", "Não"])
    coceira_val = 1 if coceira_val == "Sim" else 0

with col3:
    luz_val = st.radio("Sensibilidade à Luz", ["Sim", "Não"])
    luz_val = 1 if luz_val == "Sim" else 0

# Criar o DataFrame de entrada
entrada = pd.DataFrame([{
    'idade': idade,
    'sexo': sexo_val,
    'espessura_central': espessura,
    'curvatura_maxima': curvatura,
    'assimetria_corneana': assimetria,
    'historia_familiar': hist_val,
    'coceira_olhos_frequente': coceira_val,
    'sensibilidade_luz': luz_val
}])

# Normalizar os dados
entrada_normalizada = scaler.transform(entrada)


# Fazer a predição
if st.button("Classificar"):
    predicao = modelo.predict(entrada_normalizada)[0]
    resultado = "Com Ceratocone" if predicao == 1 else "Sem Ceratocone"
    st.success(f"Resultado da classificação: {resultado}")
