---
title: How to integrate Caddy with fail2ban
date: 2023-05-14 20:58:57
tags: [sysadmin, linux]
---

[Fail2ban](https://github.com/fail2ban/fail2ban) is an awesome security tool for servers. It allows to define rules (such as maximum number of failed SSH login attempts per time period) to temporarily IP-ban attackers or at least clients with suspicious behavior. I use it to lock out anyone who is repeatedly failing to establish an SSH session. In addition, I recently decided I also wanted to use it on a webserver level to block any potential attacker, e.g. anyone who is received a lot of `401` HTTP responses within short time. 

The way fail2ban works is by continuously analyzing log files with regexes and taking action (usually creating a blocking `iptables` entry for the respective client IP) when certain criteria are met (too many matches for same IP within certain time span). It comes with a ton of pre-defined filter rules for all kinds of different application logs, e.g. for sshd, nginx, httpd, MySQL, MongoDB any others. I'm using [Caddy](https://caddyserver.com/) as a webserver, though, and there are no filters for Caddy included. So I had to come up with my own.

Before Caddy [removed support for Common Log Format (CLF)](https://github.com/caddyserver/caddy/issues/4148), you could easily just use the nginx filter rules for Caddy as well. But today, Caddy writes its request logs as JSON by default. They look like so:

```json
{"level":"info","ts":1684090060.4355888,"logger":"http.log.access.log0","msg":"handled request","request":{"remote_ip":"164.68.116.134","remote_port":"50449","proto":"HTTP/2.0","method":"POST","host":"wakapi.dev","uri":"/api/users/current/heartbeats.bulk","headers":{"Authorization":[],"User-Agent":["wakatime/v1.73.1 (windows-10.0.19045.2965-unknown) go1.20.4 vscode/1.79.0-insider vscode-wakatime/24.0.10"],"Timezone":["Europe/Berlin"],"X-Machine-Name":["DESKTOP-N15LMBD"],"Accept":["application/json"],"Content-Type":["application/json"],"Content-Length":["403"],"Accept-Encoding":["gzip"]},"tls":{"resumed":false,"version":772,"cipher_suite":4865,"proto":"h2","server_name":"wakapi.dev"}},"user_id":"","duration":0.004969017,"size":27,"status":201,"resp_headers":{"Server":["Caddy"],"Alt-Svc":["h3=\":443\"; ma=2592000"],"Link":[],"Strict-Transport-Security":["max-age=2592000; includeSubDomains"],"Content-Type":["application/json"],"Date":["Sun, 14 May 2023 18:47:40 GMT"],"Content-Length":["27"],"Content-Security-Policy":["default-src 'self' 'unsafe-inline' 'unsafe-eval'; img-src 'self' https: data:; form-action 'self' *.stripe.com; block-all-mixed-content;"]}}
{"level":"info","ts":1684090060.9077053,"logger":"http.log.access.log0","msg":"handled request","request":{"remote_ip":"164.68.116.134","remote_port":"50468","proto":"HTTP/2.0","method":"GET","host":"wakapi.dev","uri":"/api/users/current/statusbar/today","headers":{"Timezone":["Europe/Berlin"],"X-Machine-Name":["DESKTOP-N15LMBD"],"Authorization":[],"Accept-Encoding":["gzip"],"Accept":["application/json"],"Content-Type":["application/json"],"User-Agent":["wakatime/v1.73.1 (windows-10.0.19045.2965-unknown) go1.20.4 vscode/1.79.0-insider vscode-wakatime/24.0.10"]},"tls":{"resumed":false,"version":772,"cipher_suite":4865,"proto":"h2","server_name":"wakapi.dev"}},"user_id":"","duration":0.00654806,"size":547,"status":200,"resp_headers":{"Content-Security-Policy":["default-src 'self' 'unsafe-inline' 'unsafe-eval'; img-src 'self' https: data:; form-action 'self' *.stripe.com; block-all-mixed-content;"],"Link":[],"Strict-Transport-Security":["max-age=2592000; includeSubDomains"],"Content-Type":["application/json"],"Content-Encoding":["gzip"],"Vary":["Accept-Encoding"],"Server":["Caddy"],"Alt-Svc":["h3=\":443\"; ma=2592000"],"Date":["Sun, 14 May 2023 18:47:40 GMT"]}}
{"level":"info","ts":1684090061.1078796,"logger":"http.log.access.log0","msg":"handled request","request":{"remote_ip":"164.68.116.134","remote_port":"50479","proto":"HTTP/2.0","method":"POST","host":"wakapi.dev","uri":"/api/users/current/heartbeats.bulk","headers":{"Accept-Encoding":["gzip"],"Content-Type":["application/json"],"Authorization":[],"User-Agent":["wakatime/v1.73.1 (windows-10.0.19045.2965-unknown) go1.20.4 vscode/1.79.0-insider vscode-wakatime/24.0.10"],"Timezone":["Europe/Berlin"],"X-Machine-Name":["DESKTOP-N15LMBD"],"Accept":["application/json"],"Content-Length":["407"]},"tls":{"resumed":false,"version":772,"cipher_suite":4865,"proto":"h2","server_name":"wakapi.dev"}},"user_id":"","duration":0.005720788,"size":27,"status":201,"resp_headers":{"Content-Length":["27"],"Content-Security-Policy":["default-src 'self' 'unsafe-inline' 'unsafe-eval'; img-src 'self' https: data:; form-action 'self' *.stripe.com; block-all-mixed-content;"],"Content-Type":["application/json"],"Server":["Caddy"],"Alt-Svc":["h3=\":443\"; ma=2592000"],"Link":[],"Strict-Transport-Security":["max-age=2592000; includeSubDomains"],"Date":["Sun, 14 May 2023 18:47:41 GMT"]}}
```

([Single entry formatted](https://pastr.de/p/yqvropjspbed0pzn8upj27g8))

I came up with the following filter rule, which includes a (probably not optimal) regex for extracting the remote IP address and date from these JSON lines.

```
# /etc/fail2ban/filter.d/caddy-status.conf

[Definition]
failregex = ^.*"remote_ip":"<HOST>",.*?"status":(?:401|403|500),.*$
ignoreregex =
datepattern = LongEpoch
```

This will watch for requests with `401`, `403` or `500` response codes. You may add others in addition.

The according section in `jail.local` looks like this:

```
# /etc/fail2ban/jail.local
# ...

[caddy-status]
enabled     = true
port        = http,https
filter      = caddy-status
logpath     = /var/log/caddy/*.access.log
maxretry    = 10
```

Besides the `iptables-multiport` ban action, I'm using the `abuseipdb` action for additionally reporting all blocked IPs to [AbuseIPDB](https://www.abuseipdb.com/).
