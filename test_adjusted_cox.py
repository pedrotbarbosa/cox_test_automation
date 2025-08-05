import unittest
import pandas as pd
from adjusted_cox_function import fit_adjusted_cox_model

class TestAdjustedCoxModel(unittest.TestCase):
    def setUp(self):
        # Cria um DataFrame simulado com todas as colunas necessárias
        self.df_model = pd.DataFrame({
            'HospitalLengthStay_trunc': [5, 10, 15, 20],
            'OutcomeDeath': [1, 0, 1, 0],
            'Age': [65, 70, 80, 60],
            'SofaScore': [10, 8, 12, 7],
            'Saps3Points': [50, 45, 60, 40],
            'LengthHospitalStayPriorUnitAdmission': [2, 3, 1, 4]
        })

        self.duration_col = 'HospitalLengthStay_trunc'
        self.event_col = 'OutcomeDeath'
        self.cox_predictors = [
            'Age',
            'SofaScore',
            'Saps3Points',
            'LengthHospitalStayPriorUnitAdmission'
        ]

    def test_model_runs_successfully(self):
        try:
            results, model, df_result = fit_adjusted_cox_model(
                self.df_model,
                self.duration_col,
                self.event_col,
                self.cox_predictors
            )
            self.assertIsNotNone(results)
        except Exception as e:
            self.fail(f"Model raised an exception: {e}")

    def test_output_dataframe_has_columns(self):
        results, model, df_result = fit_adjusted_cox_model(
            self.df_model,
            self.duration_col,
            self.event_col,
            self.cox_predictors
        )
        self.assertTrue(isinstance(df_result, pd.DataFrame))
        for col in self.df_model.columns:
            self.assertIn(col, df_result.columns)

    def test_model_coefficients(self):
        results, model, _ = fit_adjusted_cox_model(
            self.df_model,
            self.duration_col,
            self.event_col,
            self.cox_predictors
        )
        for predictor in self.cox_predictors:
            self.assertIn(predictor, results.index)

if __name__ == '__main__':
    unittest.main()
