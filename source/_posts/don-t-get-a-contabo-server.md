---
title: Don't get a Contabo server
date: 2024-04-25 09:50:19
tags: [sysadmin, linux]
---

I'm usually not a person to rant on companies. But in this case, my experience with [Contabo](https://contabo.com) was so bad I felt like wanting to sort of warn people from taking a bad decision and present better alternatives instead.

# Contabo offerings

Contabo is one of many hosting providers that typically offer classic webhosting, virtual private servers (VPS), virtual dedicated servers (VDS), managed block storage, etc. - essentially infrastructure for hobbyists or companies who don't want to run an own data center or go to AWS (because [they'll probably not need it](https://www.trickster.dev/post/you-probably-dont-need-aws-and-are-better-off-without-it/)). Contabo has super cheap deals, for example, you can get a 4 CPUs, 6 GB RAM, 400 GB SSD VPS for just 4.50 € / month. The prices are actually quite appealing. Plus, they run their virtual servers on [KVM](https://linux-kvm.org/page/Main_Page) - compared to other cheap hosting providers, who often times use [OS-level virtualization](https://en.wikipedia.org/wiki/OS-level_virtualization) (e.g. with [OpenVZ](https://openvz.org/)) that comes with a lot of inconvenient limitations.

However, this comes at a price. The downside to these cheap offerings if only a **[95 % annual uptime guarantee](https://contabo.com/en/legal/terms-and-conditions/)**. This means that your server may be down for a whole **18 days per year**, or, **1 hour 12 mins per day** on average (see [uptime / SLA calculator](https://uptime.is/)). That's quite bad, if you think about it - especially considering that other providers (e.g. [netcup](https://www.netcup.de/)) give you over 99 % SLAs.

![](images/contabo_1.png)

# My experiences

I've had a couple of personal services, as well as [Wakapi.dev](https://wakapi.dev), [Anchr.io](https://anchr.io) and a few others running on a Contabo VPS for years. The server went down more or less regularly - on average, probably like once a month for one or two hours. While it was annoying, it didn't hurt me a lot, because I didn't have _super important_ stuff running on it. What really bothered me, though, was their communication: usually they didn't provide any information on the incident or at least a notice that something was wrong and that they would be working on fixing it.

## The last straw breaking the camel's back

I thought about switching providers every now and then, but always got to the conclusion that it wouldn't be worth the efforts. Last week, though, the overstretched my patience. On Saturday at around 10 pm I received a Telegram message from my [Uptime Kuma](https://github.com/louislam/uptime-kuma) monitoring, telling me that the server was down again. At this point, I was already used to it. However, this downtime was meant to last for days.

After my VPS was still offline the next morning, I started to become nervous. I checked their X page ([@ContaboCom](https://twitter.com/ContaboCom)) and it realized I wasn't the only one facing issues. Apparently, not only the Nuremberg location was affected, but also data centers in Singapore, the US and other Contabo data centers in Germany. But it wasn't a total outage: my friend's VPS, located in the same DC, was still up. We `traceroute`d the packets and concluded that there must be some networking / routing problem. Their [status page](https://contabo-status.com) was still pretending that everything was fine and there were "no interrupations". Thanks. In the meanwhile, people on X started ranting.

[![](images/contabo_2.png)](https://twitter.com/Dotsy_Delivery/status/1781805104233861517)

Again, they didn't communicate any problems and didn't respond to tickets. If you follow their X page, those issues seem to have been around for quite a while already and seem to still continue. 

[![](images/contabo_3.png)](https://twitter.com/jmfinlay/status/1783242088072597684)

The longer my server was inaccissble, the more nervous I became. A few semi-important customer services were running on the VPS as well, which would have needed to be up again at latest Monday morning. So I started restoring the most important things from backups and moved temporarily moved them to another server at a different provider.

Eventually, it wasn't until Monday noon when my VPS came back online - again, without any notice from their support team since the incident had started. Our suspicion regarding the networking issue turned out correct: the server itself had been running for the whole time. It just didn't have internet connection.

![](images/contabo_4.png)

For me, this was the last straw that broke the camel's back. I decided to switch providers and cancel my Contabo subscription as soon as possible. Thanks to [this great article](https://webwork.blog/servermigration-mit-dd-uber-ssh/) (use Firefox- or Chrome translation features to translate into English) on cloning an entire disk via `dd` over SSH, migration was actually super easy and hassle-free.

# Alternatives

Again, I don't intend to _rant_ on Contabo. Considering my past experiences, however, I just think there are much better alternatives.

* Friends of mine are super happy with [netcup](https://www.netcup.de/vserver/vps.php) VPS' and managed hosting and even run while business infrastructures there. They are similarly cheap, but guarantee much higher uptime (also, they offer ARM64 servers, btw.). netcup is also where I decided to switch to.
* [PHP-Friends](https://php-friends.de/) looks promising as well, even though I didn't use their services, yet. They write very interesting technical blog posts on X, also ([@phpfriends](https://twitter.com/phpfriends)).
* At [TLDHost](https://www.tldhost.de/mietserver/vserver-kvm) I only purchased a simple webhosting, no VPS, but their support was exceptionally great when I used it. The owner itself would respond to my tickets and implement small custom technical changes for me.
* [Hetzner](https://www.hetzner.com) is another very popular choice and there are many more ...