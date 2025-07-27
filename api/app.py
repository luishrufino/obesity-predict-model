from flask import Flask, request, jsonify, make_response
import pickle
import logging
import pandas as pd
from pydantic import BaseModel, ValidationError
from shared.utils import FeatureEngineering, TrasformNumeric, MinMaxScalerFeatures, LifestyleScore, ObesityMap, Model, DropNonNumeric, DropFeatures

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# ======================================================================

# Importa as classes customizadas (ESSENCIAL)
try:
    from shared.utils import FeatureEngineering, TrasformNumeric, MinMaxScalerFeatures, LifestyleScore, Model, DropNonNumeric, DropFeatures
    logging.info("Módulo 'shared.utils' importado com sucesso.")
except ImportError as e:
    logging.error(f"FALHA AO IMPORTAR 'shared.utils': {e}")
    raise


app = Flask(__name__)

# ======================================================================
# --- CARREGAMENTO DO MODELO COM LOGGING ---
pipeline = None
try:
    logging.info("Iniciando carregamento do pipeline...")
    
    # O caminho para o modelo dentro do container
    pipeline_path = 'obesity_model.pkl'
    
    # Verifica se o arquivo realmente existe no caminho esperado
    if not os.path.exists(pipeline_path):
        logging.error(f"ARQUIVO DO MODELO NÃO ENCONTRADO EM: {os.path.abspath(pipeline_path)}")
        # Lista os arquivos no diretório para depuração
        logging.info(f"Arquivos no diretório atual ({os.path.abspath('.') }): {os.listdir('.')}")
        raise FileNotFoundError(f"Arquivo do modelo não foi encontrado em '{pipeline_path}'")

    logging.info(f"Carregando modelo a partir de: {pipeline_path}")
    with open(pipeline_path, 'rb') as f:
        pipeline = pickle.load(f)
    logging.info(">>> Pipeline carregado com SUCESSO! <<<")

except Exception as e:
    # Este é o log mais importante. Ele vai capturar QUALQUER erro durante o carregamento.
    logging.error("FALHA CRÍTICA AO CARREGAR O PIPELINE. O worker não vai iniciar.", exc_info=True)
    # Re-lança a exceção para que o Gunicorn ainda falhe, mas nós teremos o log.
    raise e
# ======================================================================

# Tenta carregar o pipeline
try:
    pipeline_path = 'obesity_model.pkl' 
    with open(pipeline_path, 'rb') as f:
        pipeline = pickle.load(f)
except FileNotFoundError:
    raise Exception(f"Arquivo do modelo não encontrado em {pipeline_path}. Verifique o volume do Docker.")

# Define a estrutura de dados de entrada usando Pydantic
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

        # 4. Monta a resposta completa
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
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@app.route('/')
def home():
    return "API funcionando. Use /predict com método POST para obter previsões.", 200

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({"error": "Endpoint não encontrado. Use /predict com método POST."}), 404)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)