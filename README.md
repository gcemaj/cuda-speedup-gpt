# Unsupervised Speedup Prediction of GPU paralleization using CFG and Transformers

It is not always worth it to pararellize a application using GPUs, 
this model attempts to predict the cost/benefit of doing so without having to run expensive tests on large amounts of data.

## Context Free Grammar
A CFG is created that can generate CUDA programs with the following features

- 1D, 2D, 3D problem sets
- shared memory utilization
- thread syncing
- atomic operations
- call `__device__` functions

The CFG is then used to generate 5000 different programs, each have a corresponding serialized version and paralleized version of the random program.
Each of these is then comiled and run with several different inputs (matrix size, block sizes and grid sizes) and the performance is measured as well as the 
correctness of the outputs by comparing the serialized versions to the parallel ones.

Programs that generate coda that is not equivalent at runtime are discarded.

## Modeling

Once the dataset is generated a pre trained gpt-neo model trained on source code is used as a function embedding featurizer.
This is then fed into a small feed forward neural network.

The following results are reported over a smaller 500 sample dataset with a 25-75 train validation split, no hypertuning is done on the model

```
training score= 0.99 R^2
test score= 0.98 R^2
```