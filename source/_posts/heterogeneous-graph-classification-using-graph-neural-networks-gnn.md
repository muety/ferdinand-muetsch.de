---
title: Heterogeneous Graph Classification using Graph Neural Networks (GNN)
date: 2024-10-28 08:51:13
tags: [ai, machine-learning]
---

# GNNs 101
Neural networks (NNs), in various flavors, have become the de-facto standard in pretty much every subfield of machine learning nowadays. They being used on structured data like tables and time-series, raster data like images and sequential data like natural language sentences. More recently, graph neural networks have emerged in addition, in order to apply established deep learning techniques to graph-structured data as well. Graphs are a super flexible and universally applicable data model, that can be used anywhere from biology / chemistry to medicine, social networks analysis, recommender systems and static code analysis to energy grids autonomous driving. In my research, I apply GNNs for **clustering and generating traffic scenarios**, but more on that later... I might do a separate blog posts only on the fundamentals of how GNNs work, but for now, let's only stick to the very basics.

Graphs consist of **nodes and edges**, where edges interconnect between nodes in a directed or undirected fashion. Nodes will usually be accompanied by a **feature vector** composed of various attributes about the node (e.g. the position and velocity of a car or pedestrian on the road). Optionally, egdes can have attributes as well. _Heterogeneous_ graphs are a special type of graphs, namely such that consist of different _types_ of nodes, likely having different sorts of attributes. Besides, _spatio-temporal_ graphs are such that feature a temporal domain in some sense, usually either encoded _temporal aggregation_ / _temporal unrolling_ or sequentially in form of a _dynamic graph_ [4]. 

Most graph neural networks are based on the **message passing** scheme, where feature information is being propagated between interconnected nodes along their edges in an iterative fashion. The core part of GNNs are the **graph convolution** layers, the job of which is to aggregate and combine node features in a sensible way, that is, to generate **node embeddings**. The inner workings these layers can be broken down into three separate components: a **message function**, an **aggregation operator** and an **update function**. These convolution layers are primarily used for feature extraction. Depending on the type of learning tasks (there are **node-level**-, **link-level**- and **graph-level** tasks), those layers are followed by some sort of aggregation- or _pooling_ layer. Finally, in the _readout_ stage, classical neural network layers (usually simple linear layers) are used to map from the embeddings to the final prediction (e.g. a class). 

<img src="images/gnn_pipeline.svg" width="100%" style="margin-top: 30px; margin-bottom: 15px; padding: 10px; background: #fff">
<small>Fig. 1: GNN Learning Pipeline &copy; 2024, Karlsruhe Institute of Technology</small>

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
<small>Fig. 2: Traffic scenario graph representation schema ()</small>

Another option that I considered initially was to use the _Semantic Scene Graph_ proposed by [11], then stack multiple scenes to extend it to scenarios and eventually feed these through a sequence model to obtain a global (latent) representation. **`Obstacle`** nodes are used to encode both static obstacles like road infrastructure and dynamic obstacles, primarily traffic participants. **`RoadSegment`** nodes represent the map, broken down into _lanelets_, that is, short segments of driving lanes, walkways, intersections, etc. Nodes are connected by edges of four different types:
* **`ObstacleToObstacle:`** Semantic relations between two actors, e.g. "car A follows car B" or "car A gives way to cyclist B".
* **`RoadToRoad:`** Map topology and -hierarchy, e.g. "lane 1 is right of lane 2".
* **`ObstacleToRoad:`** Association between actors and their current positions on the map, e.g. "car A is driving on lane B".
* **`ObstacleToObstacleTemporal:`** Modeling the temporal dimension of dynamic actors

Obstacles include attributes about their individual properties (type, dimensions, ...) and their current state (position, velocity, orientation, ...). Road segments are represented by their centerline and according widths, sampled at discrete steps (`k=100`). Inspired by [12] and [13], all **positions are relative** to the mean positions of all agents at `t=0`. Also, for every scenario, the **map is cropped** to the scenario's maximum extent, plus a buffer of 100 meters.

<img src="images/traffic_scenario_graph_01.svg" width="100%" style="margin-top: 30px; margin-bottom: 15px; padding: 10px;">
<small>Fig. 3: Traffic scenario graph data model</small>

In addition, **edge attributes** are employed as well. Specifically, every edge encodes relative information about the interconnected node, e.g. the _relative_ velocities between obstacle or an obstacle's displacement _relative_ to a lane's centerline. Only the temporal edges do not contain any feature information. Instead, the chronological ordering of obstacle states is modeled implicitly by the presence or abstence of temporal edges between node pairs. Note that while fig. 3 lists `timestep` as an attribute of the `Obstacle` class (implemented as actual Python classes), the attribute is dropped later when constructing actual node feature vectors. **No explicit timestep** information ends up in the final graph, however, optional **positional encoding** is added (see below).

Every node's and every edge's features are eventually flattened into fixed-size feature vectors of various dimensionality. For example, an obstacle node's feature vector is of size 30, including two components for x- and y position, longitudinal and lateral velocity, a one-hot-encoded type attribute, etc. Road segment feature vectors are larger (210 components in total), primarily because they are sampled equidistant at 100 `(x, y)` points (note: other than shown in fig. 3, only a lane's entry- and exit widths are kept for the final feature vector).

### Scaling and Positional Encoding
Since the attributes are from vastly different domains and defined in different value ranges, min-max-scaling (using Scikit-Learn's [`MinMaxScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)) is applied as a preprocessing step to every feature vector, resulting in attributed normalized to `[0, 1]`.

Moreover, optional positional encoding can be added to the obstacle state feature vectors to encode their temporal ordering. I decided to apply standard [sinusoidal](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html?highlight=positional#torch_geometric.nn.encoding.PositionalEncoding) encoding, where `x` is the respective scene's timestamp normalized to `[0, T]`. Following the original paper, the encoding is simply added to the original feature via summation.

# Scenario Graph Classification
## Problem Description
As a simple graph-level learning task, I decided to build a network to _classify_ traffic scenarios using the **[nuPlan dataset](https://www.nuscenes.org/nuplan)**. Goal is to be able to feed a scenario encoded the above graph format into the classification network and predict the category (out of 51) it most likely belongs to. The fact that these classes are (a) distributed very unevenly (see below) and (b) can be inferred trivially in a purely analytical way for the most parts makes the learning problem a lot less spectacular. Also, there is no actually sensible real-world use case for this type of classifier. However, I still wanted to build a fully-fledged GNN-based classifier to gain some more practice and to put my above data model to the test.

From a machine learning perspective, the problem essentially boils down to **represenation learning** (obtaining descriptive latent graph embeddings), followed by simple **multi-class classification**.

### nuPlan Dataset
NuPlan consists of more than 100,000 scenarios (> 500 hours of driving) of varying duration recorded in four different locations, featuring traffic agent- and ego trajectories as well as - for some parts - raw camera- and lidar sensor readings. The dataset is primarily meant to be used for trajectory prediction and traffic simulation in the context of the [nuPlan competition](https://nuplan-devkit.readthedocs.io/en/latest/competition.html). However, it also comes with simple labels per scenario. Every scenario is categorized into one of **51 different classes** using an auto-labelling process based on simple heuristics. Particular classes are, for example:

* `high_magnitude_speed`
* `low_magnitude_speed`
* `on_stopline_traffic_light`
* `accelerating_at_crosswalk`
* `changing_lane`
* `traversing_narrow_lane`
* ...

... and many more. As can be seen, every class corresponds roughly to a simple traffic maneuvor, which is what I aim to predict. 

<img src="images/nuplan_scenario_types.svg" width="100%" style="margin-top: 30px; margin-bottom: 15px; padding: 10px; background: #fff">
<small>Fig. 4: nuPlan Scenario Type Distribution ("mini" split)</small>

Unfortunately, classes are distributed very unevenly. The large majority of scenarios are of type `stationary` or `unknown`, so a classification model is likely to overfit to simply predict one of these classes and still achieve decent performance.

Maps come as GeoPackage (`.gpkg`) files containing various different vector- and raster layers, each with their own properties and semantics. 

<img src="images/nuplan_map.png" width="100%" style="margin-top: 30px; margin-bottom: 15px; padding: 10px;">
<small>Fig. 5: Selected layers of nuPlan Boston map visualized in QGIS (&copy; OpenStreetMap)</small>

## Network Architecture
The classification model's architecture that I developed is very simple and straightforward and essentially involves a GNN-based feature extraction stage followed by a classification head. 

<img src="images/gnn_classification_pipeline.svg" width="100%" style="margin-top: 30px; margin-bottom: 15px; padding: 10px; background: #fff">
<small>Fig. 6: GNN-based graph classification model architecture</small>

First, the heterogeneous graph (obstacle- and road segment nodes interconnected by four different types of attributed edges) is fed into a series of three consecutive message passing layers, effectively yielding a receptive field of 3 hops. Hidden embedding dimensions are chosen as 32, 64 and 128 for obstacles and 64, 128 and 256 for road segments. Batch normalization is applied before every convolution layer. Edge features are considered as part of inter-obstacle message passing, but are not being updated themselves. A modified variant of PyG's [`SAGEConv`](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.SAGEConv.html#torch_geometric.nn.conv.SAGEConv) layer is used, which additionally supports edge features. It does so by passing them through a single linear layer to project them to the according source node's feature dimensionality and subsequently adding them onto to that node feature vector.

```python
class EdgeSAGEConv(SAGEConv):
    '''
    Extension of SAGEConv to include edge attributes by adding (sum) them to every source neighbor node's message.
    Inspired by https://github.com/pyg-team/pytorch_geometric/discussions/2495#discussioncomment-687410.
    '''
    def __init__(self, *args, edge_in, root_node_dim=None, **kwargs):  # root node is "left" node of relation
        super().__init__(*args, **kwargs)
        if root_node_dim is None:
            root_node_dim = self.in_channels[0] if isinstance(self.in_channels, tuple) else self.in_channels
        self.edge_lin = Linear(edge_in, root_node_dim or self.out_channels)

    def forward(self, x, edge_index, edge_attr):
        self._edge_attr = edge_attr
        out = super().forward(x, edge_index)
        self._edge_attr = None
        return out

    def message(self, x_j):
        return (x_j + self.edge_lin(self._edge_attr))  # TODO: concatenate instead of add via summing
```

As a second step, the transformed obstacle node embeddings are fed through three parallel pooling layers (min-, max- and mean pooling) and subsequent concatenation to obtain a graph embedding. Edge features are not used any further after the initial message passing layers.

Lastly, this graph embedding is passed through a single linear layer to obtain the logits that correspond to the predicated scenario type class probabilities.

## Implementation Details
The [nuplan-devkit](https://github.com/motional/nuplan-devkit) is used to read map information and actual scenarios from nuPlan into Python classes to benefit from an object-oriented programming style through the code base. A `to_graph()` method is implemented to map the scenarios to [NetworkX `DiGraph`](https://networkx.org/documentation/stable/reference/classes/digraph.html)s as an intermediate representation (to perform additional graph analyses, etc.) before, eventually, another converter yields [PyG `HeteroData`](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData) objects from them, including vectorized feature representations. 

The entire ML-model is implemented using the excellent [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) framework and all data handling in this context is implemented in a [`Dataset`](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Dataset.html#torch_geometric.data.Dataset) object with additional support for in-memory caching.

Listing 1 shows an excerpt from my implementation of the scenario dataset, depicting its constructor signature and relevant method stubs. In addition, listing 2 shows an excerpt from the actual model definition, specifically the parts where convolution-, pooling- and classification layers are defined.

```python
class NuplanDataset(Dataset):
    def __init__(
            self,
            root: Optional[str] = None,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None,
            objective: str = Literal['scenario_classification'],
            max_n: int = -1,
            min_obstacles: int = 0,
            in_memory: bool = False,
            positional_encoding: Optional[Literal['sinusoidal', 'rotary']] = 'sinusoidal',
            minmax_scaling: bool = True,
            force_reload: bool = False,
            device: torch.device = torch.device(f'cuda:{torch.cuda.device_count()-1}' if torch.cuda.is_available() else 'cpu'),
    ):
        # ...

    def process(self):
        # ...

    def get(self, idx):
        # ...
```

<small>Listing 1: Nuplan scenario graph dataset implementation</small>

```python
class DeeptestSimpleGNN(torch.nn.Module):
    def __init__(self):
        # ...

        self.convs1.append(HeteroConv({
            ('obstacle', 'obstacle_to_obstacle_temporal', 'obstacle'): SAGEConv(obstacle_in, OBSTACLE_OUT_1, normalize=True),
            ('obstacle', 'obstacle_to_obstacle', 'obstacle'): EdgeSAGEConv(obstacle_in, OBSTACLE_OUT_1, edge_in=FEATURES_OBSTACLE_TO_OBSTACLE, normalize=True),
            ('obstacle', 'obstacle_to_road', 'road_segment'): EdgeSAGEConv((obstacle_in, road_segment_in), ROAD_OUT_1, edge_in=FEATURES_OBSTACLE_TO_ROAD, normalize=True),
            ('road_segment', 'road_to_road', 'road_segment'): EdgeSAGEConv(road_segment_in, ROAD_OUT_1, edge_in=FEATURES_ROAD_TO_ROAD, normalize=True),
        }))

        # ...

        self.pool1_mean = MeanAggregation()
        self.pool1_max = MaxAggregation()
        self.pool1_min = MinAggregation()

        self.lin1 = Linear(OBSTACLE_OUT_3 * 3, n_classes)

    # ...

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
        batch_dict: Dict[str, Tensor],
        edge_attr_dict: Optional[Dict[str, Tensor]] = None
    ) -> Tensor:
        # ...

        for conv in self.convs1:
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict if self.include_edge_attrs else {})
            x_dict = {k: x.relu() for k, x in x_dict.items()}

        # ...

        x = x_dict['obstacle']

        pooled1 = torch.cat([
            self.pool1_mean(x, batch), # (batch_size, OBSTACLE_OUT_3)
            self.pool1_max(x, batch),  # (batch_size, OBSTACLE_OUT_3)
            self.pool1_min(x, batch),  # (batch_size, OBSTACLE_OUT_3)
        ], dim=1)

        x = self.lin1(x)
        return x
    
    # ...
```

<small>Listing 2: Scenario classification model definition</small>

<!-- 
- UnScenE: Toward Unsupervised Scenario Extraction for Automated Driving Systems from Urban Naturalistic Road Traffic Data
-->

# Experiments
<!-- todo -->
<!-- nuplan mini, train-test split, ... -->

## Ablation Study
<!-- todo -->

# Conclusion
<!-- mention uneven class distribution -> better sampling, considering entire dataset -->

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