
import unittest
import pandas as pd
from lifelines import CoxPHFitter
from adjusted_cox_function import fit_adjusted_cox_model

class TestAdjustedCoxModel(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            "tempo": [5, 10, 15, 20, 25],
            "evento": [1, 1, 0, 1, 0],
            "idade": [30, 40, 50, 60, 70],
            "sexo": ["M", "F", "F", "M", "M"],
            "peso": [70, 80, 75, 65, 90],
            "hospital": ["A", "B", "A", "B", "A"]
        })

    def test_model_output_type(self):
        model = fit_adjusted_cox_model(
            df=self.df,
            time_col="tempo",
            event_col="evento",
            cont_vars=["idade", "peso"],
            cat_vars=["sexo"]
        )
        self.assertIsInstance(model, CoxPHFitter)

    def test_model_with_strata(self):
        model = fit_adjusted_cox_model(
            df=self.df,
            time_col="tempo",
            event_col="evento",
            cont_vars=["idade"],
            cat_vars=["sexo"],
            strata=["hospital"]
        )
        self.assertIsInstance(model, CoxPHFitter)

    def test_model_with_cluster(self):
        model = fit_adjusted_cox_model(
            df=self.df,
            time_col="tempo",
            event_col="evento",
            cont_vars=["idade"],
            cat_vars=["sexo"],
            cluster_col="hospital"
        )
        self.assertIsInstance(model, CoxPHFitter)

    def test_model_raises_error_with_invalid_column(self):
        with self.assertRaises(KeyError):
            fit_adjusted_cox_model(
                df=self.df,
                time_col="tempo",
                event_col="evento",
                cont_vars=["inexistente"],
                cat_vars=["sexo"]
            )

    def test_model_with_weights(self):
        self.df["peso_amostral"] = [1.2, 0.8, 1.0, 1.1, 0.9]
        model = fit_adjusted_cox_model(
            df=self.df,
            time_col="tempo",
            event_col="evento",
            cont_vars=["idade"],
            cat_vars=["sexo"],
            weights_col="peso_amostral"
        )
        self.assertIsInstance(model, CoxPHFitter)

if __name__ == "__main__":
    unittest.main()
