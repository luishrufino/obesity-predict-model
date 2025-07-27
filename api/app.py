import logging
import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify, make_response
from pydantic import BaseModel, ValidationError

# ======================================================================
# --- SETUP DE LOGGING E IMPORTS ---
# Configura o logging para exibir mensagens no console (logs do Render)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Importa as classes customizadas (ESSENCIAL)
try:
    from shared.utils import FeatureEngineering, TrasformNumeric, MinMaxScalerFeatures, LifestyleScore, Model, DropNonNumeric, DropFeatures
    logging.info("Módulo 'shared.utils' importado com sucesso.")
except ImportError as e:
    logging.error(f"FALHA AO IMPORTAR 'shared.utils': {e}", exc_info=True)
    raise

# ======================================================================

app = Flask(__name__)

# ======================================================================
# --- CARREGAMENTO DO MODELO COM LOGGING ---
pipeline = None
try:
    logging.info("Iniciando carregamento do pipeline...")
    pipeline_path = 'obesity_model.pkl'
    
    if not os.path.exists(pipeline_path):
        logging.error(f"ARQUIVO DO MODELO NÃO ENCONTRADO NO CAMINHO: {os.path.abspath(pipeline_path)}")
        logging.info(f"Arquivos no diretório atual ({os.path.abspath('.') }): {os.listdir('.')}")
        raise FileNotFoundError(f"Arquivo do modelo não foi encontrado em '{pipeline_path}'")

    logging.info(f"Carregando modelo a partir de: {pipeline_path}")
    with open(pipeline_path, 'rb') as f:
        pipeline = pickle.load(f)
    logging.info(">>> Pipeline carregado com SUCESSO! <<<")

except Exception as e:
    logging.error("FALHA CRÍTICA AO CARREGAR O PIPELINE.", exc_info=True)
    raise e
# ======================================================================

# --- DEFINIÇÃO DA ESTRUTURA DE DADOS DE ENTRADA ---
class InputData(BaseModel):
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

# --- ROTAS DA API ---
@app.route('/')
def home():
    if pipeline is not None:
        return "API funcionando e modelo carregado. Use /predict com método POST.", 200
    else:
        return "API funcionando, mas o modelo não pôde ser carregado. Verifique os logs.", 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = InputData(**request.get_json())
        features_df = pd.DataFrame([input_data.dict()])

        transformed_df = pipeline[:-1].transform(features_df)
        prediction = pipeline.predict(features_df)

        calculated_features = {
            'HealthyMealRatio': round(transformed_df['HealthyMealRatio'].iloc[0].item(), 2),
            'ActivityBalance': transformed_df['ActivityBalance'].iloc[0].item(),
            'TransportType': transformed_df['TransportType'].iloc[0],
            'LifestyleScore': transformed_df['LifestyleScore'].iloc[0].item()
        }

        response_data = {
            "prediction": int(prediction[0]),
            "calculated_features": calculated_features
        }
        
        return jsonify({
            "status": "success",
            "data": response_data
        }), 200

    except ValidationError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        # Loga o erro do predict para depuração
        logging.error("Erro durante a predição.", exc_info=True)
        return jsonify({"error": "Ocorreu um erro interno durante a predição."}), 500

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({"error": "Endpoint não encontrado. Use /predict com método POST."}), 404)

# Bloco para execução local (não é usado pelo Gunicorn)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)