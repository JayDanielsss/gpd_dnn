Information about the various iterations of this project.

Versions:

## Version 3_1:

`version_3_1` is run with the `SimultaneousFitLoss` class active with weights $w_{\text{BSA}} = 0.5$ and $w_{d^{4}\sigma} = 0.5$.

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
Observables: $\text{BSA}$, $d^{4}\sigma$

### Comments:
1. Confirmed same issue: $\text{BSA}$ fit well, but $\d^{4}\sigma$ is systematically offset during the fitting.