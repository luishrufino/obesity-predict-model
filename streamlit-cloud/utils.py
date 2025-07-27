from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import  MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin    
import pandas as pd

class DropFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, feature_to_drop=['Age']):
        self.feature_to_drop = feature_to_drop

    def fit(self, X, y=None): 
        return self

    def transform(self, X):
        X = X.copy()
        return X.drop(columns=self.feature_to_drop, errors='ignore')



class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()

        required_cols = ['Height', 'Weight', 'FCVC', 'NCP', 'FAF', 'TUE', 'MTRANS']

        if not all(col in X.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {set(required_cols) - set(X.columns)}")
        
        X['IMC'] = X['Weight'] / ((X['Height']**2))
        X['HealthyMealRatio'] = X['FCVC'] / X['NCP']
        X['ActivityBalance'] = X['FAF'] - X['TUE']
        X['TransportType'] = X['MTRANS'].apply(lambda x: 'sedentary' if x in ['Automobile', 'Motorbike'] else 'active' if x in ['Bike', 'Walking'] else 'neutral')

        return X

class TrasformNumeric(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.dummy_cols = [
            'TransportType_active', 'TransportType_neutral', 'TransportType_sedentary',
            'CALC_Always', 'CALC_Frequently', 'CALC_Sometimes', 'CALC_no',
            'CAEC_Always', 'CAEC_Frequently', 'CAEC_Sometimes', 'CAEC_no'
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        required_cols = ['family_history', 'FAVC', 'SMOKE', 'SCC', 'Gender', 'TransportType', 'CALC', 'CAEC']
        missing = set(required_cols) - set(X.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        bol_col = ['family_history', 'FAVC', 'SMOKE', 'SCC']
        X[bol_col] = X[bol_col].replace({'yes': 1, 'no': 0})

        X['Gender'] = X['Gender'].replace({'Male': 1, 'Female': 0})

        # Gera dummies
        X = pd.get_dummies(X, columns=['TransportType', 'CALC', 'CAEC'], dtype=int)

        # Garante que todas as dummies estejam presentes
        for col in self.dummy_cols:
            if col not in X.columns:
                X[col] = 0
        
        X = X.reindex(sorted(X.columns), axis=1)

        return X

class MinMaxScalerFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, min_max_col=None):
        if min_max_col is None:
            self.min_max_col = ['Height', 'Weight', 'IMC', 'ActivityBalance', 'HealthyMealRatio']
        else:
            self.min_max_col = min_max_col
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X[self.min_max_col])
        return self

    def transform(self, X):
        X = X.copy()
        X[self.min_max_col] = self.scaler.transform(X[self.min_max_col])
        return X

class LifestyleScore(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        required_cols = ['SMOKE', 'SCC', 'FAVC', 'family_history']

        if not all(col in X.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {set(required_cols) - set(X.columns)}")
        
        # Calculando a pontuação de estilo de vida
        X['LifestyleScore'] = (
            (1 - X['SMOKE']) +
            X['SCC'] +
            (1 - X['FAVC']) +
            (1 - X['family_history'])
        )

        return X

class DropNonNumeric(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        return X.select_dtypes(exclude=['object'])

class ObesityMap(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.obesity_dict = {
            'Insufficient_Weight': 0,
            'Normal_Weight': 1,
            'Overweight_Level_I': 2,
            'Overweight_Level_II': 3,
            'Obesity_Type_I': 4,
            'Obesity_Type_II': 5,
            'Obesity_Type_III': 6
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.Series):  # <- adapta para Series
            return X.map(self.obesity_dict)
        elif isinstance(X, pd.DataFrame):
            if 'Obesity' not in X.columns:
                raise ValueError("Column 'Obesity' is missing from the DataFrame.")
            return X['Obesity'].map(self.obesity_dict)
        else:
            raise TypeError("Input must be a pandas Series or DataFrame.")

class Model(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=20, n_estimators=50, random_state=42):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = RandomForestClassifier(
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def score(self, X, y):
        return self.model.score(X, y)