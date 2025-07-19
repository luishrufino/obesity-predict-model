from flask import Flask, request, jsonify, make_response
import pickle
import pandas as pd
from pydantic import BaseModel, ValidationError

app = Flask(__name__)

# Tenta carregar o pipeline
try:
    pipeline_path = '/model_data/pipeline.pkl'
    with open(pipeline_path, 'rb') as f:
        model_package = pickle.load(f)  # Carrega o pacote completo
        pipeline = model_package['pipeline']
        TRAIN_COLUMNS = model_package['feature_names']
        original_scaler = model_package['original_scaler']
    
    print("Modelo carregado com sucesso!")
    print(f"Total de colunas esperadas: {len(TRAIN_COLUMNS)}")

except Exception as e:
    raise RuntimeError(f"Erro ao carregar o modelo: {e}")

# Define a estrutura de dados de entrada usando Pydantic para validação
class InputData(BaseModel):
    Age: int
    Height: float
    Weight: float
    FCVC: int
    NCP: int
    CH2O: int 
    FAF: int
    TUE: int
    family_history: str
    FAVC: str
    SMOKE: str
    SCC: str
    CAEC: str
    CALC: str
    Gender: str
    MTRANS: str

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ADICIONE AQUI (log dos dados recebidos)
        json_data = request.get_json()
        app.logger.info(f"Dados recebidos para predição: {json_data}")

        # 1. Valida e converte os dados de entrada
        input_data = InputData(**json_data)
        app.logger.info(f"Dados validados: {input_data.dict()}")

        df = pd.DataFrame([input_data.dict()])

        # 2. Feature Engineering (exatamente como no treino)
        
        df['IMC'] = df['Weight'] / (df['Height']**2)
        df['HealthyMealRatio'] = df['FCVC'] / df['NCP'].replace(0, 1)  # Evita divisão por zero
        df['ActivityBalance'] = df['FAF'] - df['TUE']
        df['TransportType'] = df['MTRANS'].apply(lambda x: 'sedentary' if x in ['Automobile', 'Motorbike'] 
                                               else 'active' if x in ['Bike', 'Walking'] 
                                               else 'neutral')
        df['AgeGroup'] = (df['Age'] // 10).astype(int)

        # 3. Transformações categóricas
        bol_col = ['family_history', 'FAVC', 'SMOKE', 'SCC']
        df[bol_col] = df[bol_col].replace({'yes': 1, 'no': 0}).astype(int)
        df['Gender'] = df['Gender'].replace({'Male': 1, 'Female': 0}).astype(int)

        min_max_col = ['Age', 'Height', 'Weight', 'IMC', 'ActivityBalance', 'HealthyMealRatio', 'AgeGroup']
        df[min_max_col] = pipeline.original_scaler.transform(df[min_max_col]) 
        # 4. Lifestyle Score
        df['LifestyleScore'] = ((1 - df['SMOKE']) + df['SCC'] + 
                              (1 - df['FAVC']) + (1 - df['family_history']))

        # 5. One-Hot Encoding - método robusto
        categorical_cols = ['TransportType', 'CALC', 'CAEC']
        df = pd.get_dummies(df, columns=categorical_cols, dtype=int)
        
        # 6. Garante todas as colunas esperadas
        missing_cols = set(TRAIN_COLUMNS) - set(df.columns)
        for col in missing_cols:
            df[col] = 0
        
        df = df.select_dtypes(exclude=['object'])

            
        # 7. Ordena as colunas exatamente como no treino
        df = df[TRAIN_COLUMNS]

        # Verificação de colunas (apenas para debug)
        app.logger.info("\n[DEBUG] Verificação pré-predição:")
        app.logger.info(f"Colunas presentes: {df.columns.tolist()}")
        app.logger.info(f"Total de colunas: {len(df.columns)}")
        app.logger.info(f"Valores de exemplo: {df.iloc[0].to_dict()}")
        # 8. Predição
        prediction = pipeline.predict(df)
        return jsonify({'prediction': int(prediction[0])})

    except ValidationError as e:
        return make_response(jsonify({'error': 'Dados inválidos', 'details': e.errors()}), 400)
    except Exception as e:
        app.logger.error(f"Erro na predição: {str(e)}", exc_info=True)
        return make_response(jsonify({'error': 'Erro interno no servidor'}), 500)

@app.route('/')
def home():
    return "API de Predição de Obesidade está funcionando.", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)