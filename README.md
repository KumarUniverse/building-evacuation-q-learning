# building-evacuation-q-learning
An algorithm that uses Q-learning to find the optimal path to evacuate a building.

## [Tutorial](http://mnemstudio.org/path-finding-q-learning-tutorial.htm)

## Building Diagram
![building_diagram](https://github.com/KumarUniverse/building-evacuation-q-learning/blob/main/pictures/building-diagram.png)

## Building Directed Graph
![building_directed_graph](https://github.com/KumarUniverse/building-evacuation-q-learning/blob/main/pictures/room-graph.png)

## Results
```
Completed 100 Q-learning simulations with an average move count of 5.25
Completed 100 evacuations with an average move count of 1.57
Q table:
[
    [0, 0, 0, 0, 69.0, 0],
    [0, 0, 0, 38.99, 0, 95.76],
    [0, 0, 0, 42.62, 0, 0],
    [0, 58.82, 18.71, 0, 68.97, 0],
    [42.8, 0, 0, 42.92, 0, 99.66],
    [0, 0, 0, 0, 0, 0],
]

Completed 500 Q-learning simulations with an average move count of 5.008
Completed 500 evacuations with an average move count of 1.59

Completed 1000 Q-learning simulations with an average move count of 5.03
Completed 1000 evacuations with an average move count of 1.45
```
