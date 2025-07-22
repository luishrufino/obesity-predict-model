from flask import Flask, request, jsonify, make_response
import pickle
import pandas as pd
from pydantic import BaseModel, ValidationError

app = Flask(__name__)

# Tenta carregar o pipeline
try:
    pipeline_path = '/model_data/pipeline.pkl'
    with open(pipeline_path, 'rb') as f:
        pipeline = pickle.load(f)
except FileNotFoundError:
    raise Exception(f"Arquivo do modelo não encontrado em {pipeline_path}. Verifique o volume do Docker.")



# Define a estrutura de dados de entrada usando Pydantic para validação
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
        prediction = pipeline.predict(features_df)
        return jsonify({
            "status": "success",
            "data": {"prediction": int(prediction[0])}
        }), 200
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return "API funcionando. Use /predict com método POST para obter previsões.", 200

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({"error": "Endpoint não encontrado. Use /predict com método POST."}), 404)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)