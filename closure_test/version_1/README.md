Information about the various iterations of this project.

Versions:

## Version 1_1:

`version_1_1` is run with the `SimultaneousFitLoss` class active but importantly sets the weight in front of the `\text{BSA}` observable to $0$ (which means we set the weight in front of the cross-section observable to $1$).

### DNN Hyperparameters:
Replicas: 20
Epochs (without EarlyStop): 750
Input nodes: 4
Output nodes: 2
Hidden layers: 1
Nodes per layer: 10
Activation per layer: [ReLU]
Final activation: Linear

### Data:
Training points: $231$
Validation points: $57$
Testing points: $72$
Test/Train split: $20/80$
Train/Validation split: $80/20$
Sum: $231 + 57 + 72 = 360$
Observables: $d^{4}\sigma$

### Comments: