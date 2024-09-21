---
title: Sending Prometheus Alerts to Telegram with Telepush
date: 2022-03-11 20:13:37
tags: [sysadmin, linux]
---

<img src="images/prometheus_telepush.webp" width="256px">

# Prometheus Alerting
[Prometheus](https://www.prometheus.io/) is among the most popular tools for server monitoring. It aggregates metrics from potentially countless different sources (provided through so called [exporters](https://www.prometheus.io/docs/instrumenting/exporters/)), stores them in its own time series database and allows them to be queried, e.g. by visualization tools like [Grafana](https://grafana.com). One central aspect of monitoring, besides visualization of data, is **alerting**. A very common use case for Prometheus is to watch a server's runtime performance metrics, e.g. its CPU utilization or hard drive usage, and be notified when things go unusual, for instance, when the hard drive is close to running out of free space. Or you might want to use the [`blackbox_exporter`](https://github.com/prometheus/blackbox_exporter) to watch an HTTP or TCP endpoint for being available and get immediate alerts as it goes down. For this, Prometheus integrates with [Alertmanager](https://www.prometheus.io/docs/alerting/latest/overview/), their own little daemon process whose sole purpose is to send out alert notifications triggered by Prometheus queries.

Alertmanager supports for different receivers, among the most popular of which arguably is [`email`](https://www.prometheus.io/docs/alerting/latest/configuration/#email_config). But there are many more, including the generic [`webhook`](https://www.prometheus.io/docs/alerting/latest/configuration/#webhook_config) one, which is quite powerful.

# Receiving alerts in Telegram
E-Mail is well and good, but also feels a bit "heavy-weight" and outdated. You might rather want to receive your server monitoring alerts through your favorite messenger, for instance, Telegram. In the following, you'll learn how to do so with minimal effort by using [**Telepush**](https://telepush.dev). As an example, we are going to set up Prometheus to watch a locally running web service and send a Telegram messages once it's unavailable.

## Prerequisites
I assume you have an up and running setup of Prometheus, Alertmanager and Blackbox Exporter (only required for this example) already.

## Obtain a token from [Telepush](https://telepush.dev)
You will receive notifications through Telepush's bot. From a technical perspective, an alert takes this way:

```
Prometheus --> Alertmanager --> Telepush API --> Telegram
```

That is, Alertmanager sends all alerts to Telepush's web API, which then forwards it to your Telegram chat with the bot. To instruct Telepush where to route which messages, you initially have to get in contact with Telepush's Telegram bot and obtain a token.

Open a new chat with [TelepushBot](https://t.me/MiddlemanBot) (_Telepush was formerly called MiddlemanBot_) and simply send `/start`. You'll receive a **token**. Keep it for the next step.

## Configure Alertmanager
Let's continue with the interesting part and configure Alertmanager to send alerts via Telepush. In my case, the Alertmanager's config file lives at `/opt/alertmanager/alertmanager.yml`.

```yaml
# alertmanager.yml

global:

# catch-all route to receive and handle all incoming alerts
route:
  group_by: ['alertname']
  group_wait: 10s       # wait up to 10s for more alerts to group them
  receiver: 'telepush'  # see below

# telepush configuration here
receivers:
- name: 'telepush'
  webhook_configs:
  - url: 'https://telepush.dev/api/inlets/alertmanager/<YOUR_TOKEN>'    # add your Telepush token here
    http_config:
```

Then, apply the configuration through a reload:

```bash
curl -X POST http://localhost:9093/-/reload
```

**🎉 You're done!** That's already it.

No more is needed to hook up Alertmanager with Telepush. For the sake of completeness, though, let's continue with the config requires to connect Prometheus and Alertmanager in the first place and to set up blackbox monitoring.

## Excursus: Configure Prometheus
Below is the configuration to instruct Prometheus where to find the Alertmanager and a job to monitor a webpage at `http://localhost:4000` for being up and returning a `2xx` HTTP response code.

```yaml
# prometheus.yml

# Alertmanager configuration
alerting:
  alertmanagers:
  - static_configs:
    - targets:
       - "localhost:9093"                   # address of your alertmanager service

# Load custom rules rules once and periodically evaluate them according to the global 'evaluation_interval'.
rule_files:
  - "custom_alerts.yml"

# Scrape configs
scrape_configs:
  - job_name: 'blackbox-demo'               # choose a name
    metrics_path: /probe
    scrape_timeout: 10s
    params:
      module: [https_2xx]                   # check for http success response
    static_configs:
      - targets:
        - 'http://localhost:4000'           # add your target url to watch
    relabel_configs:
      - source_labels: [__address__]
        target_label: __param_target
      - source_labels: [__param_target]
        target_label: instance
      - target_label: __address__
        replacement: '127.0.0.1:9101'       # address of your blackbox exporter
```

Additionally, we need to configure the notification template for alerts triggered by Blackbox Exporter.

```yaml
# custom_alerts.yml

- name: BlackboxAlerts
  rules:
  - alert: EndpointDown
    expr: probe_success == 0
    for: 1m
    labels:
      severity: "critical"
    annotations:
      summary: "Endpoint {{ $labels.instance }} down"
```

## Excursus: Configure Blackbox Exporter
Another short snippet is required in the blackbox exporter's config file to make available the `http_2xx` module used above.

```yaml
# blackbox.yml

modules:
  http_2xx:
    prober: http
```

# Testing it
To set the set up, we simply reload Prometheus (just like reloading the Alertmanager above) and wait for a minute.

![](images/telepush1.webp)