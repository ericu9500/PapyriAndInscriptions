# Instruct-Tuning Larger Pretrained Causal Language Models for the Reconstruction, Geographical Attribution, and Dating of Ancient Greek Documentary Texts

This repository contains all the necessary files and resources used in a study that is currently under review. The goal of the study is to fine-tune pretrained causal language models for reconstructing, geographically attributing, and dating ancient Greek documentary texts, including papyri and inscriptions.

## Overview

- **Training and Test Data**: 
  The scripts for downloading, formatting, and preprocessing papyri into training and test datasets are located in the `/train_data` directory. These scripts are numbered sequentially to allow easy replication or inspection of the data preparation process.

  For inscriptions, refer to the [iphi repository](https://github.com/sommerschield/iphi). Apply scripts `6–14` from this repository with the necessary adjustments to process inscription data. Processed datasets (before applying scripts `8–14`) are available on Hugging Face at [this link](https://huggingface.co/collections/Ericu950/papyri-and-inscriptions-66ed3af86b665725dcc28ca5).

- **Evaluation Notebooks**: 
  - **`evals/01_generate_test_results.ipynb`**: Use this notebook to generate test results using the fine-tuned models. Note that GPU access is required for efficient execution.
  - **`evals/02_compute_scores.ipynb`**: This notebook computes the evaluation scores based on pre-made evaluation files. Use this if you wish to inspect the evaluation metrics without regenerating test results.

- **Models**:  
  The fine-tuned models are available on Hugging Face: [Papyri and Inscriptions Collection](https://huggingface.co/collections/Ericu950/papyri-and-inscriptions-66ed3af86b665725dcc28ca5).

## Tools Used

- **Model Training**:  
  Model fine-tuning was performed using [Torchtune](https://github.com/pytorch/torchtune). The training configurations used are stored in the `yamls/` directory.

- **Model Merging**:  
  After training, the models were merged using [Mergekit](https://github.com/arcee-ai/mergekit). The settings for merging are also available in the `yamls/` directory.


## Acknowledgements

This research was supported by resources provided by the **National Academic Infrastructure for Supercomputing in Sweden (NAISS)**, partially funded by the **Swedish Research Council** through grant agreement no. 2022-06725.

The study builds on the foundational work of Assael, Y., Sommerschield, T., Shillingford, B., et al., in their publication: *Restoring and attributing ancient texts using deep neural networks*. **Nature 603**, 280–283 (2022). [https://doi.org/10.1038/s41586-022-04448-z](https://doi.org/10.1038/s41586-022-04448-z)
