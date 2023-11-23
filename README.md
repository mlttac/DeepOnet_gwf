# Developing a Cost-Effective Emulator for Groundwater Flow Modeling Using Deep Neural Operators

This repository contains the code and resources for the research paper "Developing a Cost-Effective Emulator for Groundwater Flow Modeling Using Deep Neural Operators," published in the Journal of Hydrology.

## Abstract
Our work introduces a groundbreaking emulator designed to forecast the impact of abstraction in aquifers efficiently and accurately. Utilizing a Deep Neural Operator (DeepONet) framework, our method learns mappings between infinite-dimensional function spaces, enabling precise predictions in groundwater flow modeling. The model's efficacy is demonstrated through various tests including forward time-dependent problems, inverse analysis, and nonlinear systems.

## Repository Structure
- `requirements.txt` - Lists all necessary packages to run the code.
- `/tests` - Includes test files (E1_ForwardProblem, E3_TimeDependentProblem).
- `/model` - Contains the model architecture.
- `/data_generation` - Scripts for data generation using MODFLOW.

## Getting Started
1. Clone this repository to your local machine.
2. Install the required packages with `pip install -r requirements.txt`.
3. Explore the folders:
   - `data_generation` for data preparation scripts.
   - `tests` for model testing.
   - `model` for the model architecture.

## Tests
- **E1_ForwardProblem**: Examines the distribution of hydraulic head in a confined aquifer with a constant-rate pumping well.
- **E3_TimeDependentProblem**: Focuses on predicting the hydraulic head response over time in response to variable pumping rates.

## Contributing
Contributions are welcome, especially for enhancing the tests or expanding the data generation capabilities. Please fork the repository, make your changes, and submit a pull request.

## Citation
If this work aids your research, please cite our paper:
```bibtex
@article{
  title={Developing a Cost-Effective Emulator for Groundwater Flow Modeling Using Deep Neural Operators},
  author={Maria Luisa Taccari and others},
  journal={Journal of Hydrology},
  year={2023},
  publisher={Elsevier}
}
