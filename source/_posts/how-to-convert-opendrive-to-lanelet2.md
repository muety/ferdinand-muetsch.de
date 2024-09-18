---
title: How to convert OpenDRIVE to Lanelet2
date: 2024-04-19 23:44:12
tags:
---

[OpenDRIVE](https://www.asam.net/standards/detail/opendrive/) and [Lanelet2](https://github.com/fzi-forschungszentrum-informatik/Lanelet2) are both representation formats for maps, particularly for road networks and commonly used in autonomous driving. Despite being designed for pretty much the same purpose, both formats differ quite heavily from a conceptual perspective, especially in how they model road networks internally. However, one can be converted to the other.

![](images/opendrive-cover.webp)
([Source](https://carla.readthedocs.io/en/latest/tuto_content_authoring_maps/))

# Primer on OpenDRIVE and Lanelet2
OpenDRIVE is much more standardized and structured, less ambiguous, but also more complex conceptually. Lanelet2, on the other hand, is very open and flexible, but thus also leaves room for interpretation in many cases.

In OpenDRIVE, the entire road network is defined relative to a reference line, along which individual lanes and other road elements extend at given offsets. OpenDRIVE files (`.xodr`) themselves are XML files following a standardized schema.  You can find an example [here](https://github.com/carla-simulator/carla/blob/dev/PythonAPI/util/opendrive/TownBig.xodr).

Lanelet2, on the other hand, essentially represents the world in form of simple geometric structures, including points, linestrings, polygons and - first and foremost - so called _lanelets_. Lanelets are essentially polygons that follow a few special constraints and that represent individual, elementary pieces of lanes (and thus of entire roads). Each of these [primitives](https://github.com/fzi-forschungszentrum-informatik/Lanelet2/blob/master/lanelet2_core/doc/LaneletPrimitives.md), can (and usually will) have a set of _tags_ attached to them, which describe their role and "meaning". Lanelet2 maps are stored as OpenStreetMap files (`.osm`), that is, XML files consisting of nodes and relations between these nodes. [Here](https://github.com/fzi-forschungszentrum-informatik/Lanelet2/blob/master/lanelet2_maps/res/mapping_example.osm) is an example.

Lanelet2 maps can be viewed and edited using [JOSM](https://josm.openstreetmap.de/). For viewing OpenDRIVE maps, you can use [odrviewer.io](https://odrviewer.io/). For creating and editing, you can use the (non-free) [RoadRunner](https://carla.readthedocs.io/en/latest/tuto_content_authoring_maps/#create-a-road-network-using-roadrunner) software, [TrueVision Designer](https://www.truevision.ai/designer) or the [Driving Scenario Creator add-on](https://github.com/johschmitz/blender-driving-scenario-creator) for Blender.

# Converting OpenDRIVE to Lanelet2
As Lanelet2 is the structurally "simpler" format, it can be created from OpenDRIVE in a fairly straightforward way. Converting OpenDRIVE to Lanelet, on the other hand, is much harder. There have been [attempts](https://carla.readthedocs.io/en/latest/tuto_G_openstreetmap/#convert-openstreetmap-format-to-opendrive-format) to come up with conversion tools, but - to my knowledge - none of them works great, yet. We're focusing on the `.xodr` -> `.osm` conversion in this article.

Luckily, a research group at [TUM](https://www.tum.de/) has come up with a very convenient tool for this, called [CommonRoad Scenario Designer](https://commonroad.in.tum.de/tools/scenario-designer). They even published a [paper](https://mediatum.ub.tum.de/1449005) on _Automatic Conversion of Road Networks from OpenDRIVE to Lanelets_.

## Prerequisites
You will need to have a recent version of **Python** (preferably Python 3.11) installed. Also, of course, you will need the `.xodr` file to be converted.

## Installation
Install the scenario designer using pip:

```bash
$ pip install commonroad-scenario-designer
```

## Conversion
Run the conversion:

```bash
$ crdesigner --input_file my_map_01.xodr --output_file my_map_01.osm odrlanelet2
```

That's it! There's literally nothing more to it.

![](images/lanelet_map.webp)
(Lanelet map visualized in [QGIS](https://qgis.org))

# Projections and coordinate systems

The scenario designer will, by default, not perform any coordinate conversions. Output will be in the same coordinate reference system (CRS) as your XODR input. Often times, you will see [`EPSG:4258`](https://epsg.io/4258) or [`EPSG:4326`](https://epsg.io/4326) to be used, that is, latitude / longitude pairs - or what is commonly referred to as GPS coordinates. However, when loading your Lanelet2 file programatically (see example below) using their tool suite, things look much differently. You will find coordinates in a vastly different range.

The reason is that the Lanelet2 library converts to Cartesian, that is x- and y, coordinates upon loading the map. Cartesian ("projected") coordinates are much easier to work with, because you can apply Euclidean arithemtic, etc. (simply get the distance between two points like you learned in high school), which does not work as is when dealing with lat / lon angles. More specifically, Lanelet2, by default, uses something they call a [Spherical Mercator projection](https://github.com/fzi-forschungszentrum-informatik/Lanelet2/blob/master/lanelet2_io/include/lanelet2_io/Projection.h#L54). To my understanding, this is some custom-made thing, does not correspond to an official CRS / projection and thus will cause you a lot of problems when working with in other contexts.

Thus, I recommend to instruct Lanelet2 to use [UTM coordinates](https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system) (see [`UtmProjector`](https://github.com/fzi-forschungszentrum-informatik/Lanelet2/blob/master/lanelet2_projection/src/UTM.cpp#L8)) instead. UTM is is a widely adopted framework, even though special in a way that it splits the world into so called _zones_ and then has CRS' that are only defined for certain parts of the globe. It is, nevertheless, very useful and my recommendation if you want to work with your Lanelets' coordinates somewhere else than only inside the Lanelet2 map itself.

Here's a code example to load and reproject the map in Python:

```python
import lanelet2
from lanelet2.projection import UtmProjector
from lanelet2.io imprt Origin

origin = Origin(42.344889, -71.044961)  # some arbitrary point somewhere close to (within a few km) to the center of your map
utm_projector = UtmProjector(origin, useOffset=False, throwInPaddingArea=False)
lanelet_map = lanelet2.io.load('my_map_01.osm', utm_projector)

print(f'Loaded {len(list(lanelet_map.laneletLayer))} lanelets ...')
```

# Closing remarks
**Update Sep, 18th 2024:** The popular GIS framework [GDAL](https://gdal.org) will also feature an OpenDRIVE driver in version [3.10](https://github.com/OSGeo/gdal/pull/9504), the implementation of which is described in detail in [this work](https://elib.dlr.de/110123/). This will allow to easily convert from OpenDRIVE to most other support geo-data formats, such as GeoJSON, GeoPackage or Shapefile ([...](http://switchfromshapefile.org/)). Since there is also an [OSM driver](https://gdal.org/en/latest/drivers/vector/osm.html) and Lanelet2 is just a specific "dialect" (probably not the correct term) of OSM, getting to lanelets will probably not be very hard from that point on.


**Happy coding!**