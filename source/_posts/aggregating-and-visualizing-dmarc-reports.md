---
title: Aggregating and Visualizing DMARC Reports
date: 2021-05-07 21:23:08
tags: [sysadmin, linux]
---

![](https://apps.muetsch.io/images/o:auto/rs,s:640?image=https://muetsch.io/images/dmarc-cover.jpg)

# DMARC, DKIM and SPF
If you are using a custom domain for your e-mail with providers like [mailbox.org](https://mailbox.org) or even host your own mail server, it is likely that you came across these terms at some point. All three are technologies for improving e-mail security and authentication. Their whole purpose is to ensure that a mail was actually sent by the person who appears to have sent it, i.e. the person mentioned in the mail's _From_ header. I don't want to go too much into detail here and only explain each of them very briefly. 

## SPF
[SPF](https://shibumi.dev/posts/spf-dkim-dmarc/) – the _Sender Policy Framework_ – is quite easy to understand. As the owner of a domain, say _example.org_, you precisely specify in the DNS records of your domain what mail servers, identified by their IPs, are allowed to send mail for that domain. For instance, if you set a `TXT` record like `"v=spf1 ip4:164.68.116.134 a -all"`, you are telling any receiving mail server in the world to only accept mail from `*.example.org` if the sending SMTP server has that very certain IP and drop all other incoming messages. Of course, you have to rely on the receiving server to fulfill its responsibility of actually performing that DNS lookup and verification.

Syntax of SPF records still goes a bit beyond the above example and allow for _include_s, which is basically a way of delegating the specification of the actual SPF record to a different domain. This is especially helpful when using external mail providers, as you, as a customer, obviously cannot know every single IP of their outgoing SMTP hosts.

## DKIM
The concept of [DKIM](https://en.wikipedia.org/wiki/DomainKeys_Identified_Mail) is similarly straightforward. Its idea is that a sending mail server, e.g. yours or your mail hoster's one, adds a cryptographic signature to all outgoing mail. Given the sender's public key, a receiving server can easily verify that signature then. To find out the public key belonging to the sending servers of _example.org_, that public key is stored as another `TXT` record in _example.org_'s DNS zone. That's it. 

## DMARC
In contrast to the previous two methods, DMARC is not actually used for authentication, but rather for reporting. It is a way to notify a domain owner about what is going wrong out there in the internet with regard to e-mail. DMARC essentially tells (again, via DNS) receiving mail servers, including GMail, Outlook, GMX, etc., who to send reports about failed SPF and DKIM verifications to. Depending on the actual implementation, not only failure notifications are sent, but also regular, summarizing reports, even if all goes well. Moreover, DMARC also specifies an XML-based file format for those reports. You, as a domain owner, can then read and understand those reports and take action – whatever that might be. 

An example for a full DMARC record can be found [here](images/dmarc_example.xml). I only want to emphasis on one part of the `<record>` section here.

```xml
<row>
    <source_ip>80.241.xx.xxx</source_ip>
    <count>1</count>
    <policy_evaluated>
        <disposition>none</disposition>
        <dkim>pass</dkim>
        <spf>pass</spf>
    </policy_evaluated>
</row>
<identifiers>
    <header_from>wakapi.dev</header_from>
</identifiers>
```

The report (from Google, in this case) notifies me, owner of _wakapi.dev_, that one mail from the above IP was received within a certain time span and that both SPF and DKIM checks were alright. Seems like my DNS records are correct and nobody tried to scam in my name. Perfect.

# Configuring DMARC
Setting up SPF and DKIM for your domain is highly recommended, as the chances of your mails being considered spam by a receiver are significantly lower then. Afterwards, you still need to set up DMARC. It is probably best to just google how to do that, there are many great posts out there (like [this (German-language) one](https://www.ionos.de/digitalguide/e-mail/e-mail-sicherheit/dmarc-erklaert/)).

In essence, you will end up with a DNS record similar to this one: 
```
v=DMARC1; p=reject; adkim=r; aspf=r; rua=mailto:dmarc@muetsch.io; ruf=mailto:dmarc@muetsch.io
```

In the example, any mail server who actively implements DMARC reports, is instructed to send them to _dmarc@muetsch.io_. I would recommend to have a separate address (different from your main address) for DMARC records, to you can easily set up rules like _"move everything addresses to dmarc@mydomain.tld into some IMAP folder and mark it as read"_. 

# Aggregation and visualization
As can be seen from the example above, DMARC records are a bit unpleasant to read and you probably do not want to go through every report (couple per day) manually. Luckily, a quick GitHub search led me to a project that helps with this, specifically it does:

1. Read DMARC reports from your inbox via IMAP
2. Parse them
3. Persist them into a database
4. Visualize them on a website

![](https://apps.muetsch.io/images/o:auto/rs,s:640?image=https://muetsch.io/images/dmarc1.png)
(Source: https://www.techsneeze.com)

To be precise, the tool is two separate scripts, the parser and the web dashboard.

## Database setup
First, you will need a MySQL or MariaDB database, which the parser can write to and the dashboard can read from. If you do not already have a running MySQL instance anyway, you can easily set one up, e.g. using Docker.

```bash
docker run -d -p 3306:3306 -e MYSQL_RANDOM_ROOT_PASSWORD=yes -e MYSQL_DATABASE=dmarc -e MYSQL_USER=dmarc -e MYSQL_PASSWORD=sshhh --name mariadb-dmarc mariadb
```

This command already creates a database and user for you. If you don't want to spawn a whole new database instance or don't want to use Docker, you will have to create the database and user manually.

```bash
$ mysql -u root -p
```

```sql
$ USE dmarc;
$ CREATE USER 'dmarc'@'%' IDENTIFIED WITH mysql_native_password BY 'sshhh';
$ GRANT ALL PRIVILEGES ON 'dmarc'.* TO 'dmarc'@'%';
$ FLUSH PRIVILEGES;
$ quit;
```

## Parser setup
The parser can be found at [techsneeze/dmarcts-report-parser](https://github.com/techsneeze/dmarcts-report-parser) and is written as a single-file Perl script, alongside a configuration file. The repo's README explains the setup process quite well. 

In the config file, you need to set your above database connection (usually something like `imap.yourprovider.com`, port 993, SSL) and credentials as well as your IMAP credentials to log in to your mail account. With `$imapreadfolder` you tell the parser which IMAP folder to search for DMARC mails. This should preferably not be your inbox' root folder to not distract the parser and not risk losing any mail. I, personally, set up a sub folder _dmarc_ in my inbox (IMAP path is `INBOX/dmarc` then) and created a rule to automatically put all DMARC mails in there. 

Since the parser is a one-shot script and not a long-running process, you will probably want to invoke it on a regular basis, which you can set up a cron job for, using `crontab -e`

```
@hourly 	cd ~/dev/dmarcts-report-parser && ./dmarcts-report-parser.pl -i
```

## Viewer setup
The viewer / web dashboard is located at [techsneeze/dmarcts-report-viewer](https://github.com/techsneeze/dmarcts-report-viewer) and implemented as a simple PHP application. That means, you will need a web server like Apache2, nginx or Caddy (I'd recommend the latter) alongside a PHP installation (e.g, using PHP-FPM). Explaining how to set these things up is beyond the scope of this article, but you can just google it. After putting your database credentials once again (this time for reading), you are good to go. 

Hit `http://localhost/dmarcts-report-viewer.php` (or whatever your domain name is) in your browser and you will be presented a (not that nice-looking, but) very convenient and practical web UI to get an overview over all DMARC reports, grouped by domains, recipient providers and date. Much cooler than reading XML files one by one!

Thanks a lot for the great work by [@techsneeze](https://github.com/techsneeze).