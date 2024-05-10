# Instructions to run evaluations

Possible datasets:

- digts
- titanic.csv
- loan.csv
- parkinsons.csv

All plots generated are saved in the figures/ directory

## Randon Forest

```
python random_forest_runner.py [dataset] [criterion] [min_size_split] [min_gain] [max_depth]
```

## Neural Network

```
python nn_runner.py [dataset] [lambda] [alpha] [num_hidden_layers] [neurons per hidden layer 0] [neurons per hidden layer 1] ...
```

To get the cost J plot

```
python nn_plot.py [dataset] [lambda] [alpha] [num_hidden_layers] [neurons per hidden layer 0] [neurons per hidden layer 1] ...
```

## Ensemble

```
python nn_ensemble.py [dataset]
```
