# Backdoor Detection

`detect.py` is the script that runs the backdoor detection algorithm.
It takes as input a path to model and a dataset: "mnist", "cifar10", or "gtsrb".
It will then run the backdoor detection algorithm on the model and dataset.

## Usage

```bash
python detect.py <path to model>  <mnist|cifar10|gtsrb>
```

## Example

```bash
python detect.py ../mnist/badnets_baseline.h5 mnist
```

## Output

The output of the script is a list of MAD values for each of the labels in the dataset.
