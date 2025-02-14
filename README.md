<div style="float:right; margin-left:20px; margin-top: -30px;">
    <img src="https://avatars.githubusercontent.com/u/81195336?s=200&v=4" alt="Optimal PSE logo" title="OptiMLPSE" height="150" align="right"/>
</div>
<br>
<br>

# EvalRetro
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A repository for evaluating single-step retrosynthesis algorithms - [Digital Discovery](https://doi.org/10.1039/D4DD00007B).
 
This code was tested for Linux (Ubuntu), Windows and Mac OS.

## Environment
Set up a new environment by running the following line in your terminal: 

``` 
conda env create -n evalretro --file environment.yml 
pip install rxnfp --no-deps
```
For MacOS, replace the environment.yml file with:
``` 
conda env create -n evalretro --file environment_mac.yml
pip install rxnfp --no-deps
```

## Testing your own algorithm
<details>
  <summary>📚 Discover more about testing your own single-step algorithm!</summary>
<br>

To test your own retrosynthetic prediction on a test dataset (e.g. [USPTO-50k](https://www.dropbox.com/sh/6ideflxcakrak10/AAAESdZq7Y0aNGWQmqCEMlcza/typed_schneider50k?dl=0&subfolder_nav_tracking=1)), follow the steps below: 
1. Place the file containing the predictions per molecular target in the ./data/"key" directory ("key" as defined in config file - step 2.) <br />
    > Please ensure the correct layout of your prediction file as shown in [File Structure](#File-Structure)
2. Enter the configuration details in the config under [new_config.json](./config/new_config.json) by replacing the example <br />
    > Please refer to [Configuration Structure](#Configuration-File) for the layout
3. To ensure that the file has the correct structure, run the following line of code: 
    ```
    conda activate evalretro
    python data_import.py --config_name new_config.json 
    ```
4. If no error is logged in step 3, the algorithm can be evaluated with: 
    ```
    python main.py --k_back 10 --k_forward 2 --invsmiles 20 --fwd_model 'gcn' --config_name 'new_config.json' --quick_eval True  
    ```
    Within the script, the following arguments can be adjusted: 
    - **k_back**: Evaluation includes _k_ retrosynthesis predictions per target
    - **k_forward**: Forward model includes _k_ target predictions per reactant set.
    - **fwd_model**: Type of forward reaction prediction model. Choose from [_gcn_, _lct_]
    - **config_name**: Name of the config file to be used
    - **quick_eval**: Boolean - prints the results (averages) for evaluation metrics directly to the terminal.
    - **data_path**: The path to the folder that contains your file, default = ./data
      
> For further help, look at the Jupyter notebook provided in [the examples directory](./examples/evaluate_algorithm.ipynb)

### File Structure
The file should follow **one** of the following two formats with the **first row entry per target molecule being the ground truth reaction** i.e. 1 ground-truth reaction + N predictions per target:

1. **Line-Separated** file: _N+1_ reactions per _molecular target_ are separated by an empty line (example: [TiedT](https://figshare.com/articles/journal_contribution/USPTO-50k/25325623?file=44795752))
2. **Index-Separated** file: _N+1_ reactions per _molecular target_ are separated by different indices (example: [G<sup>2</sup>Retro](https://figshare.com/articles/journal_contribution/USPTO-50k/25325623?file=44795767))

The headers within the file should contain the following columns: ["index", "target", "reactants"]

### Configuration File
The configuration for the benchmarked algorithm is shown in [the config directory](./config/raw_data.json). Specifying the configuration is important so that the data file is processed correctly by the code. 
The structure is in .json format and should contain: 
```
"key":{
    "file":"file_name.csv",       # The name of the prediction file in ./data/"key" directory
    "class":"LineSeparated",      # One of: ["LineSeparated", "IndexSeparated"]
    "skip":bool,                  # false if LineSeparated; true if IndexSeparated
    "delimiter":"comma",          # Delimiter of file. One of: ["comma", " "]
    "colnames": null,             # null - unless data file has different header to ["idx", "target", "reactants"]
    "preprocess":bool,            # false - in most cases
}
```
</details>

## Reproducibility
<details>
  <summary>🔍 Step-by-step guide on how to reproduce results presented in the paper.</summary>
<br>

1. Download all data files from dropbox and place inside ./data directory <br />
    > The datafiles related to all benchmarked algorithms can be found below: <br>
    > https://doi.org/10.6084/m9.figshare.25325623.v1
2. Run the following lines of code within your terminal:
   ```
   conda activate evalretro
   python data_import.py --config_name raw_data.json
   python main.py --k_back 10 --k_forward 2 --invsmiles 20 --fwd_model 'gcn' --config_name 'raw_data.json' --quick_eval False
   ```
3. Run `python plotting.py` to generate figures and tables
</details>

## Benchmarking Results
<details>
    <summary>📊 Results on USPTO-50k dataset</summary>

<br>

| Algorithms      | Rt-Accuracy (Top-10)* | Diversity | Validity | Duplicity | SCScore |
|-----------------|:---------------------:|:---------:|:--------:|:---------:|:-------:|
| _Semi-template_  |                       |           |          |           |         |
| MEGAN           |      0.78 \| 0.80     |    0.30   |   0.90   |    0.90   |   0.36  |
| GraphRetro      |      0.77 \| 0.80     |    0.19   |   0.84   |    0.47   |   0.35  |
| RetroXpert      |      0.46 \| 0.48     |    0.27   |    0.81  |    0.91   |   0.42  |
| G²Retro      |      0.69 \| 0.73     |    0.31   |     -    |    0.98   |   0.32  |
|  _Template-free_  |                       |           |          |           |         |
| Chemformer      |      0.86 \| 0.88     |    0.12   |   0.99   |    0.12   |   0.47  |
| Graph2Smiles    |      0.43 \| 0.45     |    0.23   |   0.64   |    0.90   |   0.46  |
| Retroformer     |      0.68 \| 0.71     |    0.24   |   0.92   |    0.83   |   0.43  |
| GTA             |      0.72 \| 0.75     |    0.24   |   0.94   |    0.76   |   0.47  |
| TiedTransformer |      0.69 \| 0.72     |    0.29   |   0.94   |    0.93   |   0.39  |
|  _Template-based_ |                       |           |          |           |         |
| GLN             |      0.84 \| 0.87     |    0.23   |    1.0   |    0.64   |   0.41  |
| LocalRetro      |      0.81 \| 0.85     |    0.30   |    1.0   |    0.95   |   0.40  |
| MHNReact        |          0.78         |    0.32   |    1.0   |    1.0    |   0.30  |

*As evaluted by _gcn_ (WLDN-5) | _lct_ (LocalTransform) forward models, respectively.

</details>


# Interpretability Study
<details>
  <summary>🚀 Click here to find out more details about interpretability of ML-based retrosynthesis models.</summary>
<br>

The code related to the interpretability study is found in [the interpretability folder](./interpret).

## Environment
The environment can be set-up running the following lines of code: 

```
conda create -n rxn_exp python=3.10
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg -c pyg
conda install scikit-learn -c conda-forge
conda install tqdm matplotlib pandas
pip install rdkit
```

## Data Files
Install both folders within ./data_interpretability using the following link and place them into the ./interpret directory: <br>
https://doi.org/10.6084/m9.figshare.25325644.v1

## Reproducibility
Pre-trained models are provided in the dropbox. However, models can be retrained by running: 
```
conda activate rxn_exp
cd interpret
python train.py --model_type DMPNN
```
The model_type can be chosen from: DMPNN, EGAT and GCN.

To test the trained models (i.e. EGAT and DMPNN) and create the plots as in the paper, run:  
```
conda activate rxn_exp
python inference.py
```
**Note**: The plots for the GNN models may slightly differ compared to the paper due to the stochastic nature of GNNExplainer.
![Example of interpretability case study](/examples/example_interpret.png)

</details>

# Acknowledgements
⭐ This repository is built on several open-source packages. We would like to thank the authors and acknowledge the following: 

- [RDKit](https://github.com/rdkit/rdkit): For reading SMILES strings and checking validity of molecules
- [SCScore](https://github.com/connorcoley/scscore): For calculating the SCScore metric.
- [rxnfp](https://github.com/rxn4chemistry/rxnfp): For calculating the Diversity metric.
- [WLDN-5](https://github.com/connorcoley/rexgen_direct) and [LocalTransform](https://github.com/kaist-amsg/LocalTransform): The reaction forward models used for calculating the Rt-Accuracy metric.
