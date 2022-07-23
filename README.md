### The Ripple-Spreading Algorithm for the Multi-Objective Shortest Path Problem

##### Reference: Deb K, Pratap A, Agarwal S, et al. A fast and elitist multiobjective genetic algorithm: NSGA-II[J]. IEEE Transactions on Evolutionary Computation, 2002, 6(2): 182-197. 

##### Reference: Ahn C W, Ramakrishna R S. A genetic algorithm for shortest path routing problem and the sizing of populations[J]. IEEE Transactions on Evolutionary Computation, 2002, 6(6): 566-579.

The multi-objective aims to find a set of paths with minimized costs. 

| Variables   | Meaning                                                      |
| ----------- | ------------------------------------------------------------ |
| network     | Dictionary, {node 1: {node 2: [weight 1, weight 2, ...], ...}, ...} |
| source      | The source node                                              |
| destination | The destination node                                         |
| gen         | The maximum number of generations (iterations)               |
| pop         | Population size                                              |
| p_mutation  | The probability of mutation                                  |
| p_crossover | The probability of crossover                                 |
| neighbor    | List, [[the neighbor nodes of node 1], [the neighbor nodes of node 2], ...] |
| nw          | The number of objectives                                     |
| population  | List, all individuals ({'pareto rank': the Pareto rank (integer), 'chromosome': the chromosome (path), 'objective': the objective value on each objective (list)}) |

#### Example

![image](https://github.com/Xavier-MaYiMing/The-NSGA-II-for-the-multi-objective-shortest-path-problem/blob/main/MOSPP.png)

```python
if __name__ == '__main__':
    test_network = {
        0: {1: [62, 50], 2: [44, 90], 3: [67, 10]},
        1: {0: [62, 50], 2: [33, 25], 4: [52, 90]},
        2: {0: [44, 90], 1: [33, 25], 3: [32, 10], 4: [52, 40]},
        3: {0: [67, 10], 2: [32, 10], 4: [54, 100]},
        4: {1: [52, 90], 2: [52, 40], 3: [54, 100]},
    }
    source_node = 0
    destination_node = 4
    print(main(test_network, source_node, destination_node))
```

##### Output:

```python
[
    {'path': [0, 3, 4], 'objective': [121, 110]}, 
    {'path': [0, 2, 4], 'objective': [96, 130]}, 
    {'path': [0, 3, 2, 4], 'objective': [151, 60]},
]
```

