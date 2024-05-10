import unittest
import pandas as pd
from modeloVan import VanCalculator  # Adjust this import to your module's actual name

class TestFlujoCalculation(unittest.TestCase):
    def setUp(self):
        # Define constants for number of rows and scenarios
        num_rows = 15  # Assuming 15 rows for each test case
        scenarios = 3  # Number of scenarios for each type of data (below, at, above threshold)

        # Setup a sample dataframe with all necessary columns for the test
        data = {
            'DIAMETRO_MEDIDOR': [37] * num_rows + [39] * num_rows + [38] * num_rows,  # Diameters around the threshold
            'Curva proy': ['ULTRA'] * num_rows + ['ULTRA'] * num_rows + ['NOT_ULTRA'] * num_rows,  # Include different curve projections
            'Inversion': [1000] * num_rows + [2000] * num_rows + [3000] * num_rows,
            'Valor res': [500] * num_rows + [1000] * num_rows + [1500] * num_rows
        }

        # Generate dynamic columns for Ingresos, C_E, and impuestos for 1-15
        for i in range(1, 16):
            data[f'Ingresos_{i}'] = [100 * i] * scenarios * num_rows
            data[f'C_E_{i}'] = [10 * i] * scenarios * num_rows
            data[f'impuesto {i}'] = [1 * i] * scenarios * num_rows

        self.df = pd.DataFrame(data)

    def test_flujo_rollover_conditions(self):
        calculator = VanCalculator()
        calculator.dfs['BBDD - Error Actual'] = self.df
        calculator.calculate_flujo()

        # Fetch the modified DataFrame
        modified_df = calculator.dfs['BBDD - Error Actual']

        # Check if rollover conditions are correctly applied for Flujo 14 and Flujo 15
        for i in range(1, 16):
            expected_flujo = self.df[f'Ingresos_{i}'] + self.df[f'C_E_{i}'] + self.df[f'impuesto {i}']
            if i in [14, 15]:
                # Check for rows where conditions apply
                condition_mask = (self.df['DIAMETRO_MEDIDOR'] < 38) | (self.df['Curva proy'] != 'ULTRA')
                expected_flujo[condition_mask] += self.df['Inversion'] if i == 14 else self.df['Valor res']
            
            pd.testing.assert_series_equal(modified_df[f'Flujo {i}'], pd.Series(expected_flujo, name=f'Flujo {i}'), check_names=False)

if __name__ == '__main__':
    unittest.main()
