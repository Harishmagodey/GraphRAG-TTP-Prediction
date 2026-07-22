# GraphRAG-TTP-Prediction

## Overview
This repository contains the implementation of an Explainable Temporal Cyber Threat Knowledge Graph (ET-CTKG) for cyber threat prediction. The system uses GraphRAG, MITRE ATT&CK knowledge graphs, temporal graph learning, and LLM-assisted reasoning to predict future adversarial Tactics, Techniques, and Procedures (TTPs).

## Features
- Cyber Threat Intelligence (CTI) data processing
- MITRE ATT&CK knowledge graph construction
- Entity and relation extraction
- Temporal Knowledge Graph generation
- GraphRAG-based threat prediction
- Explainable attack prediction
- Experimental evaluation and graph generation

## Repository Contents
- `Experiment_3_GraphRAG_TTP_Prediction.ipynb` – Main implementation notebook
- `APT_Knowledge_Graph.json` – APT knowledge graph
- `Attack_Knowledge_Graph.json` – MITRE ATT&CK knowledge graph
- `Extracted_data_1.json` – Extracted CTI data
- `Experiment3_Results.csv` – Experimental results
- `baseline_evaluation.py` – Baseline evaluation
- `graph_fig2.py` – Figure 2 generation
- `graph_fig3.py` – Figure 3 generation
- `graph_fig4.py` – Figure 4 generation

## Workflow
1. Collect Cyber Threat Intelligence (CTI) data.
2. Extract entities and relationships using LLMs.
3. Construct a Temporal Knowledge Graph.
4. Apply Temporal Graph Learning.
5. Predict future ATT&CK techniques.
6. Generate explainable security recommendations.

## Technologies
- Python
- GraphRAG
- Knowledge Graphs
- MITRE ATT&CK
- NetworkX
- Pandas
- Matplotlib

## Author
Harishma
