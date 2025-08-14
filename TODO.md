## General:

- [ ] Implement other feature selection algorithms
- [ ] Implement other Health Score algorithms
- [x] Compute Time In Advance
- [ ] Save the evaluation interval when serializing
- [x] Check what happens when instead of 0 and 1 we use 0.1 and 0.9 for the binary models
- [x] Apply the algorithm with multiple health status to a network with only 1 output -> transform this into a regression problem (generalization of Health Status in Recurrent Networks) -> For now, not very good results -> check branch network-regression, need to debug the RNN case. Increasing Loss + NaN loss for many classes -> Need to find hyperparameters that are not necessarily the same
- [ ] Implement the voting algorithm from Health Status in Recurrent Networks which takes a max over multiple samples instead of adding them
- [ ] Maybe implement feature extraction (methods listed on ) 
- [ ] Test the regression tree with the continuous health status
- [ ] Rerun the tests for the multilevel models

## Binary BPNN:

- [x] Find standard parameters
- [x] Find number of epochs
- [x] Vary number of failing samples
- [x] Vary Change Rate Interval
- [x] Vary Feature Count
- [ ] Vary Feature Selection Algorithm
- [x] Vary Good Bad ratio
- [x] Vary Hidden Nodes
- [x] Vary Learning Rate
- [x] Vary Decay Interval
- [x] Vary Vote Count
- [x] Vary Vote Threshold
- [x] Exclude Change Rate
- [ ] Change the activation function


## Multilevel BPNN:

- [x] Find standard parameters
- [x] Find number of epochs
- [x] Vary number of failing samples
- [x] Vary Change Rate Interval
- [x] Vary Feature Count
- [ ] Vary Feature Selection Algorithm
- [ ] Vary Health Status Algorithm
- [x] Vary Health Status Count
- [x] Vary Good Bad ratio
- [x] Vary Hidden Nodes
- [x] Vary Learning Rate
- [x] Vary Decay Interval
- [x] Vary Vote Count
- [x] Vary Vote Threshold
- [x] Exclude Change Rate
- [ ] In the report, mention that the voting system requires the non-linear function of the last layer to be removed

## Binary RNN:

- [x] Find standard parameters
- [x] Find number of epochs
- [x] Vary number of failing samples
- [x] Vary Change Rate Interval
- [x] Vary Feature Count
- [ ] Vary Feature Selection Algorithm
- [x] Vary Good Bad ratio
- [x] Vary Hidden Nodes
- [x] Vary Learning Rate
- [ ] ~~Vary Decay Interval~~ (No interesting results)
- [x] Vary Vote Count
- [x] Vary Vote Threshold
- [x] Vary Lookback
- [x] Exclude Change Rate


## Multilevel RNN:

- [x] Find standard parameters
- [x] Find number of epochs
- [x] Vary number of failing samples
- [x] Vary Change Rate Interval
- [x] Vary Feature Count
- [ ] Vary Feature Selection Algorithm
- [ ] Vary Health Status Algorithm
- [x] Vary Health Status Count
- [x] Vary Good Bad ratio
- [x] Vary Hidden Nodes
- [x] Vary Learning Rate
- [ ] ~~Vary Decay Interval~~ (No interesting result)
- [x] Vary Vote Count
- [x] Vary Vote Threshold
- [x] Vary Lookback
- [x] Exclude Change Rate
- [ ] Check if the model is able to learn more classes with more nodes

## Binary LSTM:

- [x] Find standard parameters
- [x] Find number of epochs
- [x] Vary number of failing samples
- [x] Vary Change Rate Interval
- [x] Vary Feature Count
- [ ] Vary Feature Selection Algorithm
- [x] Vary Good Bad ratio
- [x] Vary Hidden Nodes
- [x] Vary Learning Rate
- [x] Vary Decay Interval
- [x] Vary Vote Count
- [x] Vary Vote Threshold
- [x] Vary Lookback
- [x] Exclude Change Rate


## Multilevel LSTM:

- [x] Find standard parameters
- [ ] Find number of epochs
- [x] Vary number of failing samples
- [x] Vary Change Rate Interval
- [x] Vary Feature Count
- [ ] Vary Feature Selection Algorithm
- [ ] Vary Health Status Algorithm
- [x] Vary Health Status Count
- [x] Vary Good Bad ratio
- [x] Vary Hidden Nodes
- [x] Vary Learning Rate
- [x] Vary Decay Interval
- [x] Vary Vote Count
- [x] Vary Vote Threshold
- [x] Vary Lookback
- [x] Exclude Change Rate
- [ ] Check if the model is able to learn more classes with more nodes

## CT

- [x] Find standard parameters
- [x] Vary number of failing samples
- [x] Vary Change Rate Interval
- [x] Vary Feature Count
- [ ] Vary Health Status Algorithm
- [ ] Vary Feature Selection Algorithm
- [x] Vary Good Bad ratio
- [x] Vary Max Depth
- [x] Vary minimum samples on a leaf
- [x] Vary Criterion
- [x] Vary Vote Count
- [x] Vary Vote Threshold
- [x] Exclude Change Rate
- [x] Vary Health Status Count

## RT

- [x] Find standard parameters
- [x] Vary number of failing samples
- [x] Vary Change Rate Interval
- [x] Vary Feature Count
- [ ] Vary Health Status Algorithm
- [ ] Vary Feature Selection Algorithm
- [x] Vary Good Bad ratio
- [x] Vary Max Depth
- [x] Vary minimum samples on a leaf
- [ ] Vary Vote Count
- [ ] Vary Vote Threshold
- [ ] Exclude Change Rate
- [x] Vary Health Status Count