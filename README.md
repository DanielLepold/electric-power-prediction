# electric-power-prediction

## Description:

The electric-power-prediction console app is a Python application designed to
perform exploratory data analysis and regression calculations on datasets,
particularly focusing on estimating the electrical power (PE) of a
thermal power plant system based on environmental factors. The app utilizes
various regression techniques and provides insights into the relationships
between input variables and the target variable (PE).

### Command Line Arguments
The application accepts the following command line arguments:

- ```folder```: Output result folder path.
- ```train_path```: Train input file path.
- ```run_type```: Specify whether to run an Exploratory Data Analysis or
Regression Calculation.
  - To run an Exploratory Data Analysis, use "EDA" as the argument value.
  - To run a Regression calculation, use "REG" as the argument value.
- ```--test_path```: (Optional) Test input file path, required only for
regression calculation.

### Example Usage

```python main.py /path/to/output_folder /path/to/train_file REG --test_path /path/to/test_file```


