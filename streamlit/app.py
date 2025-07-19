import streamlit as st
import requests

st.set_page_config(page_title="Previsão de Obesidade", layout="centered")
st.title("🧠 Previsão de Nível de Obesidade")
st.write("Insira suas informações para prever seu nível de obesidade com base em características alimentares e físicas:")

# Entradas numéricas
Age = st.number_input("Idade", min_value=1, max_value=120, value=26, step=1, format="%d")
Height = st.number_input("Altura (em metros)", min_value=1.0, max_value=2.5, value=1.85, format="%.2f")
Weight = st.number_input("Peso (em kg)", min_value=30.0, max_value=200.0, value=86.0, format="%.2f")
FCVC = st.slider("Frequência de consumo de vegetais (0 a 3)", 1, 3, 1, step=1)
NCP = st.slider("Número de refeições por dia", 1, 4, 3, step=1)
CH2O = st.slider("Consumo de água por dia (copos)", 0, 3, 2, step=1)
FAF = st.slider("Atividade física (horas por semana)", 0, 3, 3, step=1)
TUE = st.slider("Tempo de uso de dispositivos eletrônicos (horas por dia)", 0, 2, 1, step=1)

# Entradas booleanas como selectbox
family_history = st.selectbox("Histórico familiar de sobrepeso?", ['yes', 'no'])
FAVC = st.selectbox("Come alimentos com alto teor calórico com frequência?", ['yes', 'no'])
SMOKE = st.selectbox("Você fuma?", ['yes', 'no'])
SCC = st.selectbox("Você monitora o consumo de calorias?", ['yes', 'no'])

# Frequências categóricas
CAEC = st.selectbox("Consumo de alimentos entre as refeições", ['no', 'Sometimes', 'Frequently', 'Always'])
CALC = st.selectbox("Consumo de bebidas alcoólicas", ['no', 'Sometimes', 'Frequently', 'Always'])

# Gênero
Gender = st.selectbox("Gênero", ['Male', 'Female'])

# Modo de transporte
MTRANS = st.selectbox("Modo de transporte mais utilizado", [
    'Public_Transportation', 'Walking', 'Motorbike', 'Bike', 'Automobile'
])

# Botão para enviar
if st.button("Prever Nível de Obesidade"):
    input_data = {
        "Age": Age,
        "Height": Height,
        "Weight": Weight,
        "FCVC": FCVC,
        "NCP": NCP,
        "CH2O": CH2O,
        "FAF": FAF,
        "TUE": TUE,
        "family_history": family_history,
        "FAVC": FAVC,
        "SMOKE": SMOKE,
        "SCC": SCC,
        "CAEC": CAEC,
        "CALC": CALC,
        "Gender": Gender,
        "MTRANS": MTRANS
    }

    try:
        # Faça a chamada para a API Flask
        response = requests.post("http://api:5000/predict", json=input_data)
        if response.status_code == 200:
            result = response.json()
            label_map = {
                0: 'Insufficient Weight',
                1: 'Normal Weight',
                2: 'Obesity Type I',
                3: 'Obesity Type II',
                4: 'Obesity Type III',
                5: 'Overweight Level I',
                6: 'Overweight Level II'
            }
            predicted_class = result['prediction']
            predicted_label = label_map.get(predicted_class, 'Desconhecido')

            st.success(f"🎯 Previsão: **{predicted_label}**")
            st.success(input_data)
        else:
            st.error("Erro ao realizar a predição. Verifique os valores de entrada.")
    except requests.exceptions.RequestException as e:
        st.error(f"Erro de conexão com a API: {e}")
