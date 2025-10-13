# EUBRL: Epistemic Uncertainty Directed Bayesian Reinforcement Learning
An implementation of EUBRL, a Bayesian RL algorithm that leverages epistemic guidance to achieve principled exploration.

## Installation
``` bash
python install.py
```

## Usage
Examples are given for running EUBRL. Any other algorithms in the same folder can be run in a similar manner.
### Chain
``` bash
bash scripts/Chain/EUBRL.sh
```

### Loop
``` bash
bash scripts/Loop/EUBRL.sh
```

### DeepSea
#### Stochastic
``` bash
bash scripts/DeepSea-Stochastic/EUBRL.sh
```

#### Deterministic
``` bash
bash scripts/DeepSea-Deterministic/EUBRL.sh
```

### LazyChain
#### Stochastic
``` bash
bash scripts/LazyChain-Stochastic/EUBRL.sh
```

#### Deterministic
``` bash
bash scripts/LazyChain-Deterministic/EUBRL.sh
```

## Acknowledgment
Our code is based on open-source repository [BayesRL](https://github.com/dustinvtran/bayesrl).

## License
This project is licensed under the [GNU General Public License v3.0 (GPLv3)](LICENSE).
