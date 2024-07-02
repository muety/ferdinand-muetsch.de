---
title: norospy - Using ROS without ROS
date: 2024-07-02 22:15:23
tags:
---

Wait, what? Okay, one thing at a time. First of all, what is ROS even?

# What is ROS?

[ROS](https://wiki.ros.org) is short for _Robot Operating System_. However, the name is somewhat misleading in my understanding. ROS is pretty comprehensive, but I wouldn't consider it an operating system. Rather, it's a rich suite of libraries and tools specifically suited for building robotics applications and thus sort of the de-facto standard software platform in robotics and related fields - such as autonomous driving.

![](images/ros_logo.svg)

Essentially, it mainly features a **communication middlware layer**, a **binary message serialization** format and a lot of **useful algorithms** and utilities for common robotics use cases (coordinate transformations, navigation, image processing, sensor- and actuator drivers, etc.).

For the messaging part, it's pretty much similar to technologies like the MQTT and AMQP protocol. ROS primarily provides **asynchronous** communication, where **providers** publish messages to **topics** and **subscribers** read them according to their needs. Not only TCP/IP-based communication is supported, but also local, inter-process communication (IPC). The latter is especially useful in robotics use cases, for which it's not uncommon to have real-time requirements and send especially large payloads (like point lidar point clouds or uncompressed, raw images).

Additionally, ROS comes with its own binary, schema-based message format used for payload serialization. You can think of it like the ROS-specific variant of technologies like [Protobuf](https://protobuf.dev/), [Avro](https://avro.apache.org/), [Flatbuffers](https://flatbuffers.dev/), [mcap](https://mcap.dev), [MessagePack](https://msgpack.org/), any many more.

# ROS Version Compatibility
When talking about ROS versions, first thing to mention is that every new reversion (aka. distribution) comes with its own super cool and artistic [logo](https://wiki.ros.org/Distributions)!

But apart from that, it is important to know that there are [two major-version distributions](https://roboticsbackend.com/ros1-vs-ros2-practical-overview/) of ROS. The first ROS 2 distro was released back in 2017 already, but ROS 1 is still widely in use. Its last release - [Noetic](https://wiki.ros.org/noetic) - will reach its end of life in May 2025, though. The big problem - as you would expect with major version releases - there is no backwards compatible. Consequently, dev teams will have to migrate their entire code base to switch from v1.x to v2.x, which is a non-trivial endeavor and the main reason for many robots, cars, etc. still running on the outdated ROS 1.

When developing for ROS 1, a big hurdle is the fact that is necessarily requires Ubuntu 20.04 (about to be deprecated mid 2025) and outdated library versions. Setting up a ROS 1 distribution on a modern OS (like Ubuntu 24.04, Fedora 40, ...) is close to impossible. At the same time, most developers will likely not be willing to intentionally downgrade to a deprecated stack, though.

# norospy - Docker Image and Python Client
![](images/norospy_01.svg)

As a workaround, I developed [norospy](https://gitlab.kit.edu/kit/aifb/ATKS/public/norospy). It involves a Docker setup and ROS-free Python client for streaming data from ROS 1 without requiring an actual ROS distribution to be installed. This way, you can encapsulate all ROS-related inside a Docker container, while not being forced to run your actual application code on an ancient (pre-22.04) OS. It also includes functionality to bridge from [CARLA](https://carla.org) via ROS / Foxglove to Python.

Using the provided Dockerfile, you can run an entire ROS distribution - primarily including a [roscore](https://wiki.ros.org/roscore) and most common tools, modules and libraries - inside Docker without installing any ROS-related software in your actual host system.

The second part is a small Python client library that I wrote on top of the excellent tooling provided as open source by [Foxglove](https://foxglove.dev). It addresses the problem that you'd need to have a full ROS distro installed on your system in order to use the [rospy](https://wiki.ros.org/rospy) Python package. That is, for any sort of interaction with ROS - e.g. subscribing to topics, deserializing binary messages, performing coordinate conversions, etc. - you'll need the ROS C++ stack and thus forces you into Ubuntu <= 20.04.

To work around this, an instance of [ROS Foxglove bridge](https://docs.foxglove.dev/docs/connecting-to-data/ros-foxglove-bridge/) runs inside the container and exposes ROS topics via Websockets, following Foxglove's [ws-protocol](https://github.com/foxglove/ws-protocol) specification. From thereon, you can interact with other ROS nodes via plain websockets in a platform-independent way. Well, almost, but not quite. For reading and writing the binary serialized messages, you'd still need ROS. Luckily, Fovglove open-sourced an entirely independent, pure Python [library](https://pypi.org/project/mcap-ros1-support/) ROS 1-compatible (de-)serialization, which norospy builds upon (btw. if you're interested in reading and writing [rosbags](https://wiki.ros.org/rosbag) purely with Python, there is the excellent [rosbags library](https://pypi.org/project/rosbags/) provided by [MARV](https://gitlab.com/ternaris/marv-robotics)).

## How to use? 
To interact with the ROS ecosystem via norospy without having to install ROS itself, you'll simply have to build the according Docker image, spawn up a container and connect to the websocket bridge via norospy's `ROSFoxgloveClient`.

**Build image**
```bash
docker build -t localhost/foxglove-ros-bridge .
```

**Create container**
```bash
docker run -t -d \
  --net=host \
  --name foxglove-ros-bridge \
  -e ROS_IP=127.0.0.1 \
  -e ROS_HOSTNAME=localhost \
  localhost/foxglove-ros-bridge
```

**Connect and subscribe**
```python
import signal
from norospy import ROSFoxgloveClient


# Callback for receiving images (used later)
def on_image(msg, ts):
    with open(f'/tmp/{ts}.jpg') as f:
        f.write(msg.data)


# Create a new client and start it
client = ROSFoxgloveClient('ws://localhost:8765')
client.run_background()

# Use case 1: Subscribe to data (e.g. images from CARLA, in this example)
client.subscribe('/carla/ego_vehicle/rgb_front/image/compressed', 'sensor_msgs/CompressedImage', on_image)

try:
    signal.pause()
finally:
    client.close()
```

The above code snippet is a simple example for a Python client that subscribes to a topic and persist the received RGB images in a callback.

For further details, best consult the provided [documentation](https://gitlab.kit.edu/kit/aifb/ATKS/public/norospy/-/blob/main/README.md).