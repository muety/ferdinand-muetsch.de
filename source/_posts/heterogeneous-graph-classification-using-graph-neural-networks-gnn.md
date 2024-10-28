---
title: Heterogeneous Graph Classification using Graph Neural Networks (GNN)
date: 2024-10-28 08:51:13
tags: [ai, machine-learning]
---

# GNNs 101
Neural networks (NNs), in various flavors, have become the de-facto standard in pretty much every subfield of machine learning nowadays. They being used on structured data like tables and time-series, raster data like images and sequential data like natural language sentences. More recently, graph neural networks have emerged in addition, in order to apply established deep learning techniques to graph-structured data as well. Graphs are a super flexible and universally applicable data model, that can be used anywhere from biology / chemistry to medicine, social networks analysis, recommender systems and static code analysis to energy grids autonomous driving. In my research, I apply GNNs for **clustering and generating traffic scenarios**, but more on that later... I might do a separate blog posts only on the fundamentals of how GNNs work, but for now, let's only stick to the very basics.

Graphs consist of **nodes and edges**, where edges interconnect between nodes in a directed or undirected fashion. Nodes will usually be accompanied by a **feature vector** composed of various attributes about the node (e.g. the position and velocity of a car or pedestrian on the road). Optionally, egdes can have attributes as well. _Heterogeneous_ graphs are a special type of graphs, namely such that consist of different _types_ of nodes, likely having different sorts of attributes. Besides, _spatio-temporal_ graphs are such that feature a temporal domain in some sense, usually either encoded _temporal aggregation_ / _temporal unrolling_ or sequentially in form of a _dynamic graph_ [4]. 

Most graph neural networks are based on the **message passing** scheme, where feature information is being propagated between interconnected nodes along their edges in an iterative fashion. The core part of GNNs are the **graph convolution** layers, the job of which is to aggregate and combine node features in a sensible way, that is, to generate **node embeddings**. The inner workings these layers can be broken down into three separate components: a **message function**, an **aggregation operator** and an **update function**. These convolution layers are primarily used for feature extraction. Depending on the type of learning tasks (there are **node-level**-, **link-level**- and **graph-level** tasks), those layers are followed by some sort of aggregation- or _pooling_ layer. Finally, in the _readout_ stage, classical neural network layers (usually simple linear layers) are used to map from the embeddings to the final prediction (e.g. a class). 

<!-- placeholder: graph pipeline image -->

# Traffic Scenarios as Graphs
## Modeling Considerations
In my research, I concern myself a lot with traffic scenarios. A scenario can be viewed as a sequence of scenes, which, in turn, can be modeled considering different types of information. Often times, a 6-layer model [1] is considered to distinguish between information about:

1. the **road network**,
2. **traffic infrastructure** (signs, trees, ...),
3. **temporary modifications** (constructions, ...),
4. **dynamic objects** (cars, pedestrians, cyclists, ...),
5. the **environment** (weather, time of day, ...) and
6. **digital information** (traffic light state, V2X information, ...). 

Moreover, another taxonomy distringuishes scenario descriptions into (1) **functional**, (2) **logical** and (3) **concrete**, based on the level of detail [2]. 

When coming up with a suitable traffic scenario model, there are a couple of design decisions to be made, including:
1. What information to consider (see above)
2. What data representation to choose
    * Graph-based
    * Vector-based (maps only)
    * Rasterized (2D image)
    * Object lists (agent states only)
    * Logical (e.g. OpenDRIVE, OpenSCENARIO)
    * Ontology-based ([3])
3. Scene- or agent-centric representation
4. How to represent time dimension

A commonly seen pattern is also to use separate encodings for maps and agent state, e.g. VectorNet [5], LaneGCN [6] or some custom CNN (cf. [7]) as an upstream map encoder, followed by a sequence model [7] or transformer [8] for the dynamic state.

## Heterogeneous Traffic Scenario Graph Model
For my research, I decided to opt for a representation in which traffic scenarios are encoded entirely as a single graph, including map topology / road geometry, traffic agents and the temporal dimension. It's heavily inspired by the graph models presented in [9] and [10], but varies in some aspects. I'm using a graph that is _heterogeneous_ (map- and agent / obstacle nodes), directed and spatio-temporal (using **temporal unrolling** [4]), yet _static_ (single graph). 

<img src="images/traffic_scenario_graph_02.svg" width="100%" style="margin-top: 30px; margin-bottom: 15px; padding: 10px;">

_Fig. 2: Traffic scenario graph representation schema_

Another option that I considered initially was to use the _Semantic Scene Graph_ proposed by [11], then stack multiple scenes to extend it to scenarios and eventually feed these through a sequence model to obtain a global (latent) representation. **`Obstacle`** nodes are used to encode both static obstacles like road infrastructure and dynamic obstacles, primarily traffic participants. **`RoadSegment`** nodes represent the map, broken down into _lanelets_, that is, short segments of driving lanes, walkways, intersections, etc. Nodes are connected by edges of four different types:
* **`ObstacleToObstacle:`** Semantic relations between two actors, e.g. "car A follows car B" or "car A gives way to cyclist B".
* **`RoadToRoad:`** Map topology and -hierarchy, e.g. "lane 1 is right of lane 2".
* **`ObstacleToRoad:`** Association between actors and their current positions on the map, e.g. "car A is driving on lane B".
* **`ObstacleToObstacleTemporal:`** Modeling the temporal dimension of dynamic actors

Obstacles include attributes about their individual properties (type, dimensions, ...) and their current state (position, velocity, orientation, ...). Road segments are represented by their centerline and according widths, sampled at discrete steps. Inspired by [12] and [13], all **positions are relative** to the mean positions of all agents at `t=0`. Also, for every scenario, the **map is cropped** to the scenario's maximum extent, plus a buffer of 100 meters.

<img src="images/traffic_scenario_graph_01.svg" width="100%" style="margin-top: 30px; margin-bottom: 15px; padding: 10px;">

_Fig. 3: Traffic scenario graph data model_

<!-- 
- UnScenE: Toward Unsupervised Scenario Extraction for Automated Driving Systems from Urban Naturalistic Road Traffic Data
-->

# References
* [1] 6-Layer Model for a Structured Description and Categorization of Urban Traffic and Environment
* [2] A Survey on Safety-Critical Driving Scenario Generation - A Methodological Perspective
* [3] One Ontology to Rule Them All: Corner Case Scenarios for Autonomous Driving
* [4] Graph Neural Networks: Foundations, Frontiers, and Applications
* [5] VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation
* [6] Learning Lane Graph Representations for Motion Forecasting
* [7] Trajectron++: Dynamically-Feasible Trajectory Forecasting With Heterogeneous Data
* [8] HDGT: Heterogeneous Driving Graph Transformer for Multi-Agent Trajectory Prediction via Scene Encoding
* [9] Heterogeneous Graph-based Trajectory Prediction using local Map Context and a Social Interaction Graph
* [10] Geometric Deep Learning for Autonomous Driving: Unlocking the Power of Graph Neural Networks With CommonRoad-Geometric
* [11] Towards Traffic Scene Description: The Semantic Scene Graph
* [12] GoRela: Go Relative for Viewpoint-Invariant Motion Forecasting
* [13] AgentFormer: Agent-Aware Transformers for Socio-Temporal Multi-Agent Forecasting