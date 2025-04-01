# Infectious Disease Knowledge Graph(IDKG): Integrating Multi-Modal Data for a Comprehensive Knowledge Graph to Advance Infectious Disease Research
[![python](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=yellow)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-2.3.1-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![pytorch](https://img.shields.io/badge/Neo4j-5.26.0-3BA997.svg?style=flat&logo=neo4j)](https://neo4j.com)
[![MIT license](https://img.shields.io/badge/LICENSE-MIT-A8ACB9)](./LICENSE)

## Description
![](./IDKG.png)
IDKG is a comprehensive knowledge graph for infectious diseases that integrates multi-modal biomedical data. It has comprised nearly 50,000 nodes across 8 types and 1.2 million edges of 11 types. Leveraging deep learning techniques, IDKG offers application scenarios for drug development, with particular significance during pandemic outbreaks.

## Get start

Clone the repo.

```
git clone git@github.com:henryFan128/IDKG.git
```

Create and activate the enviroment.

```
conda env create -f environment.yml
conda activate IDKG
```

## Downstream tasks
### Drug repurposing 
In the `DrugRepurposing` directory, IDKG can be applied to repurpose drugs based on the data stored in graph databases, such as Neo4j.

```
# train the model
python train.py

# predict/inference
python predict.py
```
