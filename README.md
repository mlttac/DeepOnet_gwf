# Developing a Cost-Effective Emulator for Groundwater Flow Modeling Using Deep Neural Operators

Authors: ML Taccari, H Wang, S Goswami, M De Florio, J Nuttall, X Chen, PK Jimack

This repository contains the code and resources for the research paper "Developing a Cost-Effective Emulator for Groundwater Flow Modeling Using Deep Neural Operators," published in the Journal of Hydrology.
  
## Abstract
Current groundwater models face significant challenges in their implementation due to heavy computational burdens. To overcome this, our work proposes a cost-effective emulator that efficiently and accurately forecasts the impact of abstraction in an aquifer. Our approach uses a deep neural operator (DeepONet) framework to learn operators that map between infinite-dimensional function spaces via deep neural networks. The goal is to infer the distribution of hydraulic heads in a confined aquifer in the presence of a pumping well. We successfully tested the DeepONet framework on multiple problems, including forward time-dependent problems, an inverse analysis, and a nonlinear system. Additionally, we propose a novel extension of the DeepONet-based architecture to generate accurate predictions for varied hydraulic conductivity fields and pumping well locations that are unseen during training. Our emulator's predictions match the target data with excellent performance, demonstrating that the proposed model can act as an efficient and fast tool to support a range of tasks that require repetitive forward numerical simulations or inverse simulations of groundwater flow problems. Overall, our work provides a promising avenue for developing cost-effective and accurate groundwater models.

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
