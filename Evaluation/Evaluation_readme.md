# Evaluation Experiments

This directory contains the outputs and evaluation results of the experiments performed during model training and testing.

The main experiment folders are located in the `evaluation/out_eval` directory. Each folder corresponds to a specific training configuration.

## Main Experiments

### `Out_eval_6_experiment`

Training configuration:

- Batch size: `16`
- Learning rate: `4e-4`
- Total training steps: `17,000`
- Warm-up steps: `2,500`
- Decay steps: `8,000`

This experiment evaluates the model using a batch size of 16 and a relatively high learning rate.

---

### `Out_eval_7_experiment`

Training configuration:

- Batch size: `16`
- Learning rate: `1e-4`
- Total training steps: `17,000`
- Warm-up steps: `2,500`
- Decay steps: `8,000`

This experiment uses the same batch size and number of training steps as `Out_eval_6_experiment`, but with a lower learning rate.

---

### `Out_eval_8_2_experiment`

Training configuration:

- Batch size: `32`
- Learning rate: `4e-4`
- Total training steps: `8,500`
- Warm-up steps: `1,250`
- Decay steps: `4,000`

This experiment uses a larger batch size and approximately half the number of training steps compared with the experiments using a batch size of 16.

---

### `Out_eval_9_experiment`

Training configuration:

- Batch size: `32`
- Learning rate: `1e-4`
- Total training steps: `8,500`
- Warm-up steps: `1,250`
- Decay steps: `4,000`

This experiment uses a batch size of 32 and a lower learning rate.

---

### `out_eval_10`

Training configuration:

- Batch size: `16`
- Learning rate: `4e-4`
- Total training steps: `17,000`
- Warm-up steps: `2,500`
- Decay steps: `8,000`
- Number of iterations: `1`

This experiment has the same main training configuration as `Out_eval_6_experiment`, but it was executed with a single iteration.

## Additional Evaluation Folders

### `Out_eval_9_2`

Training configuration:

- Batch size: `32`
- Learning rate: `1e-4`
- Total training steps: `8,500`
- Warm-up steps: `2,500`
- Decay steps: `8,000`

This folder contains an additional evaluation using the same batch size and learning rate as `Out_eval_9_experiment`, but with different warm-up and decay schedules.

---

### `Out_eval_8`

Training configuration:

- Batch size: `32`
- Learning rate: `4e-4`
- Total training steps: `8,500`
- Warm-up steps: `2,500`
- Decay steps: `8,000`

This folder contains an additional evaluation using the same batch size and learning rate as `Out_eval_8_2_experiment`, but with different warm-up and decay schedules.


## Experimental Purpose

The experiments were created to compare the effects of:

- Different batch sizes: `16` versus `32`
- Different learning rates: `4e-4` versus `1e-4`
- Repeated or single training iterations

The remaining folders in `evaluation/out_eval` were preliminary or exploratory runs. They were used to test the model and verify the training and evaluation procedures. These folders should not be considered part of the main set of controlled experiments.
