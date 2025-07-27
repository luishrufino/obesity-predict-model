import streamlit as st
import pandas as pd
import joblib  # Usando joblib para mais robustez
from utils import (
    FeatureEngineering, TrasformNumeric, MinMaxScalerFeatures, 
    LifestyleScore, ObesityMap, Model, DropNonNumeric, DropFeatures
)
import google.generativeai as genai


# 2. CONFIGURAÇÃO DA PÁGINA (PRIMEIRO COMANDO STREAMLIT)
st.set_page_config(page_title="ObesityFastCheck", layout="centered")

# 3. FUNÇÕES AUXILIARES
@st.cache_resource
def load_model():
    """Carrega o pipeline do arquivo uma única vez."""
    try:
        # Lembre-se de ajustar o caminho se 'app.py' estiver em uma subpasta
        pipeline = joblib.load('streamlit_cloud/obesity_model.joblib')
        return pipeline
    except FileNotFoundError:
        st.error("Arquivo do modelo 'obesity_model.joblib' não encontrado.")
        return None
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None



def gerar_analise_ia(imc, lifestyle_score, healthy_meal_ratio, activity_balance, transport_type, input_data):
    """
    Gera uma análise de saúde personalizada usando a API do Google Gemini.
    """
    # Configura a API key a partir dos segredos do Streamlit
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    except Exception:
        st.error("Chave da API do Google não encontrada. Verifique o arquivo secrets.toml.")
        return "Erro: Chave da API não configurada."

    # Cria o modelo
    model = genai.GenerativeModel('gemini-2.5-flash') # Usando o modelo Flash, que é rápido e eficiente

    # O prompt é a parte mais importante. Ele guia a IA para dar a resposta desejada.
    prompt = f"""
    Você é um assistente de saúde virtual do aplicativo ObesityFastCheck. Sua missão é fornecer uma análise prévia, educativa e motivacional com base nos dados do usuário, de forma empática e positiva.

    **Dados do Usuário:**
    - **IMC (Índice de Massa Corporal):** {imc:.2f}
    - **LifestyleScore (Pontuação de Estilo de Vida):** {lifestyle_score}
    - **HealthyMealRatio (Proporção de Refeição Saudável):** {healthy_meal_ratio:.2f}
    - **ActivityBalance (Balanço de Atividade Física vs. Tela):** {activity_balance}
    - **TransportType (Tipo de Transporte):** '{transport_type}'

    - ** Inputa do usuário: para avaliações mais precisas**
    - **Geral:** {input_data}

    **Instruções:**
    -   IMC (Índice de Massa Corporal)
        Fórmula: peso / altura²
        Bom: Entre 18.5 e 24.9 → Indica peso adequado em relação à altura.
        Ruim:Abaixo de 18.5 → Pode indicar desnutrição ou alimentação insuficiente. Acima de 25 → Pode indicar sobrepeso ou obesidade, aumentando risco de doenças crônicas.

    -   LifestyleScore (Pontuação de Estilo de Vida)
        Pontuação de zero a quatro, baseada em quatro hábitos saudáveis:
            Não fumar
            Controlar calorias
            Evitar alimentos muito calóricos
            Não ter histórico familiar de sobrepeso

        Score 4: Excelente → Estilo de vida muito saudável, com baixo risco metabólico.
        Score 2 ou 3: Regular → Há bons hábitos.
        Score 0 ou 1: Ruim → Estilo de vida de risco, associado a maus hábitos alimentares e comportamentos sedentários.

    -   HealthyMealRatio (Proporção de Refeição Saudável)
        Fórmula: consumo de vegetais / número de refeições        
        Bom: Acima de 0.5 → Indica que mais da metade das refeições incluem vegetais.
        Ruim: Abaixo de 0.3 → Pouca ingestão de vegetais, alimentação pobre em fibras e micronutrientes.

        Exemplo:
            FCVC = 3 e NCP = 3 → Ratio = 1 → Muito bom!
            FCVC = 1 e NCP = 4 → Ratio = 0.25 → Precisa melhorar!

    -   ActivityBalance (Balanço de Atividade Física vs. Tempo em Tela)
        Fórmula: FAF - TUE
        Bom: Acima de 1 → Atividade física supera o tempo em frente a telas.
        Neutro: Próximo de zero → Equilíbrio entre movimento e sedentarismo.
        Ruim: Negativo (ex: -1, -2) → Muito tempo parado, comportamento sedentário.

        Exemplo:
            FAF = 3, TUE = 1 → Balance = 2 → Excelente!
            FAF = 1, TUE = 2 → Balance = -1 → Precisa se movimentar mais!

    -   TransportType (Tipo de Transporte)
        Classificação de acordo com o nível de atividade física envolvido:
            active: Caminhada, Bicicleta → Excelente para manter rotina ativa.
            neutral: Transporte Público → Moderado, geralmente envolve caminhada parcial.
            sedentary: Automóvel, Motocicleta → Pouca ou nenhuma atividade física envolvida.


    **Sua Tarefa:**
    Com base nos dados acima, gere uma análise curta e coesa sobre o perfil de saúde do usuário. Combine as informações para dar uma visão holística. Por exemplo, se o IMC for alto, mas o LifestyleScore for bom, reconheça o esforço e sugira os próximos passos.

    **Regras Obrigatórias:**
    1.  **NUNCA forneça um diagnóstico médico formal.** Use termos como "sugere", "indica", "parece que".
    2.  **SEMPRE reforce que a ferramenta é apenas educativa** e que a consulta com um profissional de saúde (médico, nutricionista) é indispensável para um diagnóstico e plano de tratamento real.
    3.  **Use uma linguagem técnica, mas acessível.** Evite jargões médicos complexos e explique termos quando necessário. E não use nada motivacional ou de autoajuda, apenas análise.
    4.  **Formate a resposta para ser exibida no Streamlit.** Use negrito (`**`) para destacar os pontos mais importantes.
    5.  **Separe a análise de cada indicador por tópico. Deve ter 5 tópicos, 1 para cada indicador**
    6.  **Crie uma conclusão breve que resuma os pontos principais e incentive o usuário a buscar mais informações ou ajuda profissional.**
    7.  **Conside que o usuário está fazendo uma análise prévia, o objetivo da predição é auxiliar a tomada de decisão da equipe médica a diagnosticar a obesidade.**

        
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Ocorreu um erro ao contatar a IA: {e}")
        return "Não foi possível gerar a análise no momento."
    

# 4. CARREGAMENTO DO MODELO
pipeline = load_model()

# 5. INTERFACE DO USUÁRIO
st.title("🔬 ObesityFastCheck")
st.write("Insira suas informações para prever seu nível de obesidade com base em características alimentares e físicas:")

# Dicionários de mapeamento
yes_no_map = {'Sim': 'yes', 'Não': 'no'}
caec_map = {'Não': 'no', 'Às vezes': 'Sometimes', 'Frequentemente': 'Frequently', 'Sempre': 'Always'}
calc_map = caec_map.copy()
gender_map = {'Masculino': 'Male', 'Feminino': 'Female'}
mtrans_map = {
    'Transporte Público': 'Public_Transportation',
    'Caminhada': 'Walking',
    'Motocicleta': 'Motorbike',
    'Bicicleta': 'Bike',
    'Automóvel': 'Automobile'
}

# Entradas numéricas
Height = st.number_input("Altura (em metros)", min_value=1.0, max_value=2.5, value=1.85, format="%.2f")
Weight = st.number_input("Peso (em kg)", min_value=30.0, max_value=200.0, value=86.0, format="%.2f")
Gender_pt = st.radio("Gênero", list(gender_map.keys()), horizontal=True)

FCVC = st.slider("Frequência de consumo de vegetais (FCVC: 1 não consome, 2 uma ou duas vezes, 3 três vezes ou mais)", 1, 3, 1, step=1)
NCP = st.slider("Número de refeições principais por dia (NCP)", 1, 4, 3, step=1)
CH2O = st.slider("Consumo de água em litros por dia (CH2O)", 0, 3, 2, step=1)
FAF = st.slider("Atividade física semanal (FAF: 0 não prática, 1 uma vez por semana, 2 de duas a três vezes por semana, 3 mais de 3 vezes)", 0, 3, 3, step=1)
TUE = st.slider("Tempo de uso de dispositivos eletrônicos por dia (TUE: 0 pouco uso, 1 uso moderado, 2 muito uso)", 0, 2, 1, step=1)

# Entradas categóricas traduzidas
family_history_pt = st.radio("Histórico familiar de sobrepeso?", list(yes_no_map.keys()), horizontal=True)
FAVC_pt = st.radio("Consome alimentos calóricos frequentemente?", list(yes_no_map.keys()), horizontal=True)
SMOKE_pt = st.radio("Você fuma?", list(yes_no_map.keys()), horizontal=True)
SCC_pt = st.radio("Você monitora seu consumo calórico?", list(yes_no_map.keys()), horizontal=True)
CAEC_pt = st.selectbox("Consumo entre refeições (CAEC: lanches)", list(caec_map.keys()))
CALC_pt = st.selectbox("Consumo de bebidas alcoólicas (CALC)", list(calc_map.keys()))
MTRANS_pt = st.selectbox("Transporte mais usado", list(mtrans_map.keys()))


with st.sidebar:
    st.markdown("""
    <h3 style='font-size: 20px;'>ℹ️ <strong>Sobre o ObesityFastCheck</strong></h3>

    O <strong>ObesityFastCheck</strong> é uma aplicação interativa que prevê o nível de obesidade com base em hábitos alimentares, rotina física e perfil de saúde.  
    A análise utiliza variáveis originais e também variáveis derivadas, que enriquecem a interpretação e aumentam a precisão do modelo.

    <br><strong>📌 Variáveis derivadas explicadas:</strong><br>

    <strong>IMC</strong> = peso / altura²  
    Indicador clássico usado mundialmente para classificar o nível de obesidade.

    <br><strong>HealthyMealRatio</strong> = consumo de vegetais / número de refeições  
    Mede o equilíbrio alimentar, avaliando se há proporção adequada de vegetais nas refeições.

    <br><strong>ActivityBalance</strong> = atividade física - tempo em frente a telas  
    Representa o saldo entre movimentação ativa e comportamento sedentário.

    <br><strong>TransportType</strong> = tipo de transporte usado  
    Classificação:
    <ul>
        <li><strong>sedentary</strong>: Automóvel, Motocicleta</li>
        <li><strong>active</strong>: Bicicleta, Caminhada</li>
        <li><strong>neutral</strong>: Transporte público</li>
    </ul>

    <br><strong>LifestyleScore</strong> = escore de hábitos saudáveis  
    Pontua com base em:
    <ul>
        <li>Não fumar</li>
        <li>Controlar calorias (SCC)</li>
        <li>Evitar alimentos muito calóricos (FAVC)</li>
        <li>Ausência de histórico familiar de sobrepeso</li>
    </ul>
    Quanto maior, melhor o estilo de vida.

    <hr style="margin-top: 20px; margin-bottom: 10px;">
    <div style="color: red; font-size: 14px;">
        ⚠️ Esta ferramenta é apenas educativa e informativa.  
        <strong>Não substitui avaliação ou diagnóstico médico profissional.</strong>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    with st.expander("👤 Sobre o Autor"):
        st.write("""
        **Luis Rufino**  
        *https://github.com/luishrufino*
        """)


# 6. LÓGICA DO BOTÃO
if st.button("Prever Nível de Obesidade"):
    if pipeline is not None:
        input_data = {
            "Height": Height, "Weight": Weight, "FCVC": FCVC, "NCP": NCP,
            "CH2O": CH2O, "FAF": FAF, "TUE": TUE,
            "family_history": yes_no_map[family_history_pt],
            "FAVC": yes_no_map[FAVC_pt], "SMOKE": yes_no_map[SMOKE_pt],
            "SCC": yes_no_map[SCC_pt], "CAEC": caec_map[CAEC_pt],
            "CALC": calc_map[CALC_pt], "Gender": gender_map[Gender_pt],
            "MTRANS": mtrans_map[MTRANS_pt]
        }
        features_df = pd.DataFrame([input_data])

        try:
            transformed_df = pipeline[:-1].transform(features_df)
            prediction = pipeline.predict(features_df)
            # st.write("🔎 Dados enviados para a API:")
            # st.json(input_data)

            
            
            
            
            predicted_class = prediction[0] 
            

    
            imc = transformed_df['IMC'].iloc[0]
            lifestyle_score = transformed_df['LifestyleScore'].iloc[0]
            healthy_meal_ratio = transformed_df['HealthyMealRatio'].iloc[0]
            activity_balance = transformed_df['ActivityBalance'].iloc[0]
            transport_type = transformed_df['TransportType'].iloc[0]

            label_map = {
                0: 'Peso Insuficiente', 1: 'Peso Normal', 2: 'Sobrepeso Nível I',
                3: 'Sobrepeso Nível II', 4: 'Obesidade Tipo I', 5: 'Obesidade Tipo II',
                6: 'Obesidade Tipo III'
            }

            

            st.markdown("---")

            st.markdown(
                f"""
                <div style='background:rgba(200,200,200,0.5); padding:18px; border-radius:12px; margin-bottom:10px;'>
                    <span style='font-size:28px; color:#222; font-weight:bold;'>
                        Previsão: <strong>{label_map.get(predicted_class, 'N/A')}</strong>
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

        
            st.subheader("🤖 Análise Personalizada por IA")
            
            # Adiciona um spinner para mostrar que está processando
            with st.spinner('Gerando análise com base nos seus dados...'):
                analise_personalizada = gerar_analise_ia(
                    imc=imc,
                    lifestyle_score=lifestyle_score,
                    healthy_meal_ratio=healthy_meal_ratio,
                    activity_balance=activity_balance,
                    transport_type=transport_type,
                    input_data=input_data
                )
            st.markdown(analise_personalizada)
            st.markdown("---")
                    
        except Exception as e:
                st.error(f"Ocorreu um erro durante a análise: {e}")
                st.error("Verifique se todos os campos foram preenchidos corretamente.")
        pass 
else:
        st.error("O modelo não está carregado. O aplicativo não pode fazer predições.")