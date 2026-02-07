This project implements a late-integration multi-omics machine learning framework to study biological mechanisms surrounding HIV elite control. 
This framework integrates transcriptomics, proteomics, and DNA methylation data from heterogeneous public datasets using out-of-fold (OOF) predictions and decision-level integrationKey Features
Key features include: Multi-omics support, robust preprocessing, unimodal ML pipelines, Out-of-Fold (OOF) predictions, late integration, and a Interactive Streamlit UI.

Code reading order (if needed):

1. UI_stuff/app.py

2. harmonizing_stuff/classes_omic_harmonization/ AND harmonizing_stuff/data_harmonization/multiharmonize.py 

3. multiomics/training/train_transcriptomics.py
multiomics/training/train_proteomics.py
multiomics/training/train_methylation.py

4. multiomics/graphs/

