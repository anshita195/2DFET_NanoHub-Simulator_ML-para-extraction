## About The Project

This repository contains the code and results for a deep learning approach for
transistor parameter extraction. All code is implemented in Python; all data for
training is available in both the original Sentaurus csv output and compiled 
into useful NumPy arrays; and all neural networks are implemented in TensorFlow.


## Installation

To install the required dependencies, do:  

```bash
pip install -r requirements.txt
```

To install this package: 

```bash
pip install -e .
```
To create and activate a virtual environment:  

```bash
python -m pip install --upgrade pip
python -m pip install virtualenv
python -m virtualenv venv
venv\Scripts\activate      # On Windows
```

<!-- REPOSITORY LAYOUT -->
## Repository layout
Key files directories of this project are:

- [config.json](./config.json) -- A config file where key variables are defined.
- [data](./data)    -- Sentaurus simulation data from our preprint.
- [demo](./demo)    -- A training example using data from our preprint.
- [models](./models)  -- Sample pretrained models.
- [src](./src)     -- Core code for this project.


<!-- GETTING STARTED -->
## Getting Started

We provide a simple example for training and testing a neural network for 
parameter extraction of 2D trainsistors in the demo directory. 

See the [README file in the demo directory](./demo/README.md) for specific 
usage details.

<!-- LICENSE -->
## License

Distributed under the MIT License. See [LICENSE](./LICENSE).


