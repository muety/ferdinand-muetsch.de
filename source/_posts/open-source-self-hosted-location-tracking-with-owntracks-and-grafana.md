---
title: 'Open-source, self-hosted location tracking with OwnTracks and Grafana'
date: 2021-06-25 08:49:45
tags: [selfhosted]
---

# Introduction
If you are using any kinds of Google services on your smartphone, chances are high that Google continuously tracks your location and updates your [Timeline](https://www.google.com/maps/timeline).

![Google Location Timeline Screenshot](https://apps.muetsch.io/images/o:auto/rs,s:640?image=https://muetsch.io/images/owntracks4.jpg)
([Source](https://techwelkin.com/wp-content/uploads/2017/04/google-timeline-location-history.jpg))

This is a pretty cool feature and for me it is super interesting to see in retrospect which places I have visited in the past. However, it is also a bit concerning to see how precisely Google tracks your daily life. That is why I turned the location tracking off and decided to go for a rather privacy-focused approach instead.

# Your Own Setup
To build your own, self-hosted location tracking system, you are going to need a bit of technical expertise, a few different open-source software components and a small server to host them.

## Requirements / Components
* Android- or iOS smartphone
* Web server (Caddy, nginx, Apache 2, ...)
* PHP >= 7.x
* MySQL or MariaDB
* Grafana

## OwnTracks
![OwnTracks Logo](https://apps.muetsch.io/images/o:auto?image=https://muetsch.io/images/owntracks1.png)

The first step to tracking your location is to record it anywhere you go. As most people almost always carry their smartphones with them and as most smartphones have GNSS sensors, the choice is quite obvious. The only thing missing is an appropriate app. This is where [OwnTracks](https://owntracks.org/) comes to play. It is an open-source mobile app that does exactly what we need - record your location and send it to a custom configured MQTT broker or HTTP endpoint. It is available for both Android (written in Kotlin) and iOS (written in Objective-C) and comes with super detailed, developer-focused [documentation](https://owntracks.org/booklet/).

![OwnTracks Architecture Diagram](https://owntracks.org/booklet/guide/images/owntracks-iotconf-http-arch.png)
([Source](https://owntracks.org/booklet/guide/whathow/#how-owntracks-works))

It comes with different tracking [modes](https://owntracks.org/booklet/features/location/), which essentially specify how often to send your location. Weirdly enough, these modes behave differently on Android an iOS.

|                          | **Tracking on iOS**                                               | **Tracking on Android**                       |
|--------------------------|---------------------------------------------------------------|-------------------------------------------|
| **Move mode**                | After `x` meters (default: 100) or `t` seconds (default: 300) | Every 30 seconds                          |
| **Significant changes mode** | After >= 500 meters and >= 5 minutes                     | After >= `x` meters and >= `t` seconds |
| **Manual mode**              | On user request                                               | On user request                           |
| **Quiet mode**               | Never                                                         | Never                                     |

In fact, things are a bit more complex than this, as you will have to distinguish between how often the app **requests** your location from the device's sensors and how often it **sends** it. Also, there are different options regarding the desired precision (e.g. GPS is more accurate, but uses more power). Moreover, on iOS, there are _region monitoring_ and _iBeacon_ monitoring modes in addition. 

I would suggest to read through the docs and decide which modes suits you best, depending on your smartphone's OS. I, however, decided to go for the _significant changes_ mode on Android with the following custom config variables.

```json
{
    "locatorDisplacement": 25,
    "locatorInterval": 60,
    "locatorPriority": 2
}
```

This basically tells the app to _request my location with block-level accuracy (100 meters) every time I move by more than 25 meters, but at most every 60 seconds_. These settings work quite well for me and I did not recognize any significant impact on battery consumption. 

![](https://apps.muetsch.io/images/o:auto/rs,s:320?image=https://muetsch.io/images/owntracks2.png)

## Server-side script
After setting up the client side, a server-side component to receive the OwnTracks app's requests is still missing. OwnTracks ships with its [Recorder](https://github.com/owntracks/recorder), which is a small and simple web application written in C. However, I did not like it a lot, as it does not look particularly beautiful and is very limited regarding its functionality. I rather wanted to visualize my data in Grafana. But to get it there, it first needs to be persisted to a database. 

---

**UPDATE:** I found [`owntracks/frontend`](https://github.com/owntracks/frontend) in the meantime, which seems to be a lot more advanced than the recorder's web UI. You may want to use this as an alternative to Grafana, which will make the setup a lot easier.

---

A heartbeat request's payload looks like this.

```json
{
  "_type": "location",
  "acc": 13,
  "alt": 163,
  "batt": 91,
  "bs": 1,
  "conn": "w",
  "created_at": 1624607444,
  "lat": 48.9995682,
  "lon": 8.3940805,
  "t": "u",
  "tid": "l3",
  "tst": 1624607139,
  "vac": 3,
  "vel": 0
}
```

I quickly wrote [a little PHP script](https://gist.github.com/muety/3dcbb22916a4812cf3ed40ff17f1d9e2) to take OwnTracks location heartbeats and write them to a MySQL database. It can live at any web server with PHP support (I am using [Caddy2](https://caddyserver.com) with `php-fpm`). Assuming it is deployed under `https://my.server.tld/track.php` then that is the URL you need to configure as an HTTP target endpoint in the OwnTracks app. Optionally, you can configure HTTP Basic auth inside your web server's config. OwnTrack's client app has support for that built in.

## Grafana dashboard
The last missing part is to actually visualize your location data. I am a big fan of Grafana, as you can easily build beautiful visualizations with low effort. Grafana integrates with MySQL as a data source, so it can read the location data previously ingested by the above script. Using the [grafana-map-panel](https://github.com/panodata/grafana-map-panel) plugin, you can visualize geo data in dashboard. In addition, I added another two graphs for plotting my velocity and my phone's battery level over time. This is what it looks like in the end. 

![Grafana Dashboard Screenshot](https://apps.muetsch.io/images/o:auto/rs,s:640?image=https://muetsch.io/images/owntracks3.png)

The geo data panel is generated from this underlying SQL query:

```sql
SELECT
  tst AS "time",
  lat,
  lon,
  vel
FROM recordings
WHERE
  $__unixEpochFilter(tst) AND
  user = '$user' AND
  device = '$device'
ORDER BY tst
```

# Conclusion
This is it! You have your own, self-hosted, Google-free location timeline now. Have fun!