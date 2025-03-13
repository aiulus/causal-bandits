# Causal Bandits: When to Stop?

## Dependency Installation

Before running the project, ensure that all dependencies are installed. Follow the steps below to set up your environment.
### 1) Clone the Repository & Initialize Submodules
This project depends on external code from the RAPS repository. Run the following commands to properly set up the dependencies:
```plaintext
git clone --recurse-submodules https://github.com/aiulus/causal-bandits
cd causal-bandits
git submodule update --init --recursive
```
If you've already cloned the repo but forgot to include submodules, run:
```plaintext
git submodule update --init --recursive
```

### 2)Install Required Python Packages
After setting up the submodule, install the necessary Python dependencies:
```plaintext
pip install -r requirements.txt
```
Alternatively, if using conda:
```plaintext
conda create --name causal-bandits python=3.8
conda activate causal-bandits
pip install -r requirements.txt
```
## Example usages
### 1. Generate SCM with parallel topology, 2 non-reward nodes and standard Gaussian noises:
Navigate to src/utils
```plaintext
python SCM.py --graph_type parallel --n 2 --noise_types N(0,1)
``` 
saves the generated SCM to outputs/SCMs/.. as well as the causal graph generated as part of the SCM to outputs/graphs/.. as JSON files
### 2. Genereate a visualization for the same SCM:
```plaintext
python SCM.py --graph_type parallel --n 2 --noise_types N(0,1) --plot
```
saves the generated plot to outputs/plots as a .png file
