from abc import ABC, abstractmethod
import pandas as pd

#contract definer for rest of the normalizer/harmonizers

class BaseNormalizer(ABC):
    def __init__(self, method: str):
        self.method = method
        self.fitted = False
        self.report = {}

    
    @abstractmethod
    #learn parameters
    def fit(self, X: pd.DataFrame | None = None, metadata=None):
        pass

    @abstractmethod
    #applies parameters and transforms dataset
    def transform(self, **kwargs) -> pd.DataFrame:
        pass

    def fit_transform(self, X: pd.DataFrame | None = None, metadata=None, **kwargs) -> pd.DataFrame:
        self.fit(X, metadata)
        self.fitted = True
        out = self.transform(**kwargs)
        self.report["method"] = self.method
        self.report["output_shape"] = [int(out.shape[0]), int(out.shape[1])]
        return out
