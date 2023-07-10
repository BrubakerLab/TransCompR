# Project Title
Translatable Components Regression Methodology (https://www.science.org/doi/10.1126/scisignal.aay3258)

## Installation
```
pip install TransCompR
```

## Usage
```
from transcompr import TransCompR
```
```
[transCompR_mdl, human_transComps, mouse_loadings, mouse_explained, mouse_scores] = TransCompR(human_data, human_classes, mouse_data)
```
### Input

- **human_data**: pandas `DataFrame` pandas DataFrame of n_genes x n_patients.

- **mouse_data**: pandas `DataFrame` pandas DataFrame of n_genes x n_mice.

- **human_classes**: `list`, numpy `array`, or pandas `DataFrame` human patients' phenotypic labels for regression.

### Output

transCompR_mdl, human_transComps, mouse_loadings, mouse_explained, mouse_scores

- **transCompR_mdl**: sklearn `LinearRegression model` Multivariate linear regression model of human patients' phenotypic labels regressed over human patients' score on mouse PCs

- **human_transComps**: numpy `array` human patients' score on mouse PCs.

- **mouse_loadings**: numpy `array` loadings of mouse genes on mouse PCs.

- **mouse_explained**: numpy `array` variance explained by mouse PCs.

- **mouse_scores**: numpy `array` mice's score on mouse PCs.
