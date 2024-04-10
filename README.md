# Repository Readme

This repository contains scripts for processing and analyzing gold Rhetorical Structure Theory (RST) trees from the RST Discourse Treebank (RST-DT) corpus. The functionality of each script is outlined below:

## 1. `tree_parser.py`

  - This script takes as input all the gold RST trees from the RST-DT corpus.
  - It transforms the Lisp code representations of the trees into a top-down constituency parsing format.
  - Then, it performs preliminary discourse parsing evaluation using the outputs of the DMRST discourse parser.

## 2. `rst_signaling_parse.py`

  - This script aligns the signaling corpus with the evaluation format created within the preceding sections.
  - It outputs statistics about what signals are most present within errors or successes identified during the discourse parsing evaluation.

## 3. `build_features_matrix.py`

  - This script is responsible for building a training dataset.
  - It uses signals as features for error or success prediction in the discourse parsing evaluation.

## 4. `error_prediction.py`

  - This script trains an XGBoost model to predict errors or successes based on the features built by `build_features_matrix.py`.



## Citations:

If you use this repository or the data generated by it in your research, please cite the following paper:

```
@inproceedings{pastor2024signals,
  title={Signals as Features: Predicting Error/Success in Rhetorical Structure Parsing},
  author={Pastor, Martial and Oostdijk, Nelleke},
  booktitle={Proceedings of the 5th Workshop on Computational Approaches to Discourse (CODI 2024)},
  pages={139--148},
  year={2024}
}
```
