# Health Status for Hard Drive Failure Detection

Repository with the code corresponding to the implementation of the experiments in my master thesis: Health Status for Hard Drive Failure Detection.
It uses samples of SMART (Self-Monitoring, Analysis and Reporting Tool) attributes of hard drives to try and predict disk failures before they actually happen.

This library corresponds to the code required to perform the preprocessing of the data, train the models and interpret of the output of the machine learning models to evaluate its performance.
It is extensible and should allow for the implementation of additional methods, may it be additional models, other feature selection algortihms, etc.

Any additional method implemented can be easily compared to five different models already available: backpropagation neural network, RNN, LSTM, classification tree and regression tree.
Each model is available in two flavors: binary and multiclass.
The binary ones are the simple implementation in which each sample is classified in either a "Healthy" or "Failing" state.

The multiclass models use the concept of health status in which, during training, the samples are divided in more than two classes depending on their proximity to the critical failure point for the failing disks.
Therefore, there is a bigger granularity of the samples and classes than when using the ad-hoc approach.
In this case, a voting algorithm is needed to translate the output of the networks into the binary healthy/failing classification of the disks.

In order to extend the current library, start by looking at `utils.py` and `neuralNetworks.py`.
The former implements the actual training steps while the latter shows how the networks themselves are created.

The configuration of the system is done using TOML files in which every step of the training and evaluation process is setup such as the size of the failing window, the type of model, its number of hidden nodes and so on.

For more details of the theory behind the code and discussion of some results using this library, read the complete [thesis](https://github.com/Miguel0312/master-thesis-final).

## How to Use

Simply execute the command:

```
python3 main.py experiment1.toml [experiment2.toml experiment3.toml ...]
```

The experiments described in files `experiment1.toml`, `experiment2.toml`, etc. will be executed sequentially in the given order.
