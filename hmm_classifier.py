import pandas as pd
from hmmlearn.hmm import GaussianHMM

class MarketRegimeHMM:
    def __init__(self, n_states: int = 2, n_iter: int = 100):
        self.n_states = n_states
        self.n_iter = n_iter
        self.model = GaussianHMM(n_components=self.n_states, covariance_type="full", n_iter=self.n_iter)
        
    def fit_predict(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        """
        Fit the HMM model on the feature columns and predict the hidden states.
        """
        df = df.copy()
        X = df[feature_cols].values
        
        self.model.fit(X)
        states = self.model.predict(X)
        
        df["regime_state"] = states
        # Map states: state 0 is assumed to be 'bull', state 1 to 'bear' (can be dynamic, but matching original logic)
        regime_map = {0: "bull", 1: "bear"}
        df["regime_label"] = df["regime_state"].map(regime_map)
        
        return df
