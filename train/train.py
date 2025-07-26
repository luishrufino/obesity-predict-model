import os
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report
from shared.utils import FeatureEngineering, TrasformNumeric, MinMaxScalerFeatures, LifestyleScore, ObesityMap, Model, DropNonNumeric, DropFeatures

url = 'https://raw.githubusercontent.com/luishrufino/obesity-predict-model/main/Obesity.csv'
obesity_df = pd.read_csv(url)

pipeline = Pipeline([
    ('drop_feature', DropFeatures()),
    ('feature_engineering', FeatureEngineering()),
    ('transform_numeric', TrasformNumeric()),
    ('min_max_scaler', MinMaxScalerFeatures()),
    ('dropnon_numeric', DropNonNumeric()),
    ('lifestyle_score', LifestyleScore()),
    ('model', Model())
])


X = obesity_df.drop(columns=['Obesity'])
y = ObesityMap().fit_transform(obesity_df['Obesity'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
f1 = classification_report(y_test, y_pred, output_dict=True)['macro avg']['f1-score']

print("\nDesempenho do modelo final (RandomForest):")
print(f"Acurácia: {acc:.4f}")
print(f"MAE: {mae:.4f}")
print(f"F1 Macro: {f1:.4f}")


model_path = '/model_data/pipeline.pkl'
os.makedirs(os.path.dirname(model_path), exist_ok=True)
with open(model_path, 'wb') as f:
    pickle.dump(pipeline, f)
print(f"Modelo salvo em {model_path}")