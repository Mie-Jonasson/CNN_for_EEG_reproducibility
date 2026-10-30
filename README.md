# CNN_for_EEG_reproducibility

## Instruction

Challenge 1: CNNs for EEG decoding and visualization 

Group: Seminars!! (student-organized) 

Paper: Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J., Glasstetter, M., Eggensperger, K., Tangermann, M., ... & Ball, T. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. Human brain mapping, 38(11), 5391-5420. 

Read the paper: https://onlinelibrary.wiley.com/doi/10.1002/hbm.23730 

Reproduce their evaluation results using the repo: https://github.com/NeuroTechX/moabb/tree/develop [Linjeskift til tekstombrydning]Follow this tutorial (check here: https://braindecode.org/stable/auto_examples/model_building/plot_regression.html#sphx-glr-auto-examples-model-building-plot-regression-py): https://braindecode.org/stable/auto_examples/model_building/plot_bcic_iv_2a_moabb_trial.html#sphx-glr-auto-examples-model-building-plot-bcic-iv-2a-moabb-trial-py 

Make sure to switch to Deep4Net classifier  

Reproduce the DeepConvNet results in Table 2 

![](image.png)

Notes from TA 2025: 

✅  Which are the specific "evaluation results" from the paper to reproduce? The ones in table 2 

✅ In the first tutorial, pip installs work and the code runs. However, the outputs (e.g. the training curves) are not what's shared in the tutorial. 

✅ MOABB installs and getting started example works

## Report

https://www.overleaf.com/project/6ac8dcb8f771144562b4448a

## Reproduce Table 2 (Deep ConvNet)

`reproduce_table2_deepconvnet.py` trains a cropped Deep4Net per dataset, band and subject with the paper's two-phase early stopping (Sec. A.4) and evaluates once on the test set. Data loading is in `data_loading.py`, preprocessing in `preprocessing.py` (explained in `Preprocessing.ipynb`). See the script's docstring for which settings come from the paper and which are assumptions.

```bash
# smoke test (1 subject, few epochs)
python reproduce_table2_deepconvnet.py train --dataset BCIC_IV_2a --low-cut 4 --subjects 1 --max-epochs 4 --max-increase-epochs 2 --verbose

# all 46 runs (2 datasets x 2 bands x subjects) on the HPC as a Slurm array
sbatch hpc/table2_array.sbatch

# Table 2 style summary of all finished runs (writes results/summary.csv)
python reproduce_table2_deepconvnet.py summarize
```

Target Deep ConvNet mean accuracies from Table 2 (FBCSP + difference):

| Dataset | 0–f_end Hz | 4–f_end Hz |
|---|---|---|
| BCIC IV 2a | 68.0 + 2.9 = **70.9%** | 67.8 + 2.3 = **70.1%** |
| HGD | 91.2 + 1.3 = **92.5%** | 90.9 + 0.5 = **91.4%** |
| Combined (mean over all 23 subjects) | 82.1 + 1.9 = **84.0%** | 81.9 + 1.2 = **83.1%** |

## Plan

- [x] Run the tutorial notebook
- [x] Initial DeepConvNet Table 2 script (`reproduce_table2_deepconvnet.py`)
- [x] What are all the links? why do we need the MOABB repo directly? What is the relevance of the "Convolutional neural network regression model on fake data"?
- [x] What is the combined dataset? - script for joining the two datasets: moabb/moabb/datasets/compound_dataset/base.py
- [x] Figure out dataset loading using MOABB (BCIC IV 2a, HGD) -> both can be loaded through "braindecode.datasets.BNCI2014_001" and "braindecode.datasets.HGD" (code exists in the tutorial)
- [x] Figure out frequency filtering on datasets from 0 or from 4 (code exists in the tutorial)
- [x] Figure out the training procedure
- [x] Extend script to HGD + Combined rows of Table 2
- [ ] Train models and produce decoding accuracy (HPC)
- [ ] Write the report
