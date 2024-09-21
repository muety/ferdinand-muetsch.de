---
title: How development history keeps repeating itself
date: 2022-11-04 11:18:08
tags: [software-engineering]
---

![Tools in a workshop](images/dev_history1.webp)

I want to share a couple of thoughts that repeatedly come to my mind the more experienced I become in (web-) development. If you're observing the ecosystem thoughtfully, you can kind of see history repeat itself every couple of years - and the wheel being reinvented over and over again. Here is my personal opinion about innovation in the web world, about technologies like PHP and SOAP versus modern JavaScript frameworks like Next.js and paradigms like Serverless, about developer productivity, and more. 

# Innovation on the web
Innovation in the world of web development, especially in the JavaScript ecosystem, has already become a meme among developers. People joke about new frameworks and libraries popping up every day. The moment you decide to pick up the latest and greatest frontend framework, it is most likely already outdated again, because the next big thing has already climbed up to the front page of HackerNews. 

This is not necessarily a bad thing! Innovation is always great to have and many of the web frameworks and libraries out there are actually of very good quality (code, but especially also documentation) in my opinion. The hard part about this, however, is to not get distracted as a developer. It can be quite overwhelming and in the end you'll find yourself suffering from [JavaScript fatigue](https://hackernoon.com/how-it-feels-to-learn-javascript-in-2016-d3a717dd577f).

My take here is: **keep your eyes open for what's new and shiny, but decide for one technology**, commit yourself to it, become and expert with it and only switch to something else if you really see a need to.

## New technologies against boredom
But why is that even? Why are there such vast amounts of technologies, all of which serve very similar purposes in the end?

I cannot give a definitive answer to this question. But I reckon one reason is around the fact that the **entry barrier** to writing and publishing your own web framework is quite low. The web ecosystem is already so big that you'll find resources and support on virtually any topic – and platforms like Reddit, HackerNews or ProductHunt make it easy to get people aware of your project.

Another reason could be this: in a [recent podcast episode](https://www.programmier.bar/podcast/cto-special-17-stephan-schmidt-amazing-cto) of _programmier.bar_ I heard the statement being made that developers need **new technologies to stay interested** and keep up their excitement. The interviewee claims that there is a trend of (tech) companies increasingly pushing for the development of _shallow features_, as opposed to _deep features_. That is, features, which are fairly straightforward to implement and where a developer doesn't have to put a lot of thoughts and brain power into. To prevent themselves from getting bored, developers jump into new technologies / frameworks / languages or start building their own ones. 

# Reinventing the wheel
All of these novel technologies (can be languages, frameworks, libraries, or also concepts / paradigms) aim to solve some problem. But most of those problems have already been solved before. Inspired by [this Syntax.fm episode](https://syntax.fm/show/393/hasty-treat-spicy-takeout-php-is-good-and-we-re-just-re-creating-it), and by a provocative tweet about Serverless being just the same as good old PHP scripts I saw recently (can't find the link anymore :-/), I want to share some of my thoughts on that same topic. 

## PHP vs. X
Let's compare a few allegedly "ageing" technologies with more recent approaches, starting with PHP scripts.

Please note: some of this is intentionally written in a slightly provocative way. Please note also: I'm neither a PHP developer, nor do I have a lot of experience with the below frameworks.

### Server-side rendering (SSR)
I feel like this is one of the hottest topics right now. Recent frameworks with SSR support include [Next.js](https://nextjs.org/), [Nuxt.js](https://nuxtjs.org/), [SvelteKit](https://kit.svelte.dev/), [Remix](https://remix.run) and others. But didn't PHP do server rendering already 20 years ago? Yes, but... PHP did _only_ server-side rendering and nothing else. The strength of tools like Nuxt is to give developers the best of both worlds. They combine the benefits of SSR with those of single-page apps (SPA), which is extremely useful for very dynamic websites. A popular concept in this context is [partial hydration](https://markus.oberlehner.net/blog/partial-hydration-concepts-lazy-and-active/). So, if this was a competition, I'd definitely give this point to the side of the "modern approaches". 

### Mixing template and code
Some developers appreciate [JSX](https://reactjs.org/docs/introducing-jsx.html), because a component's HTML markup and its JavaScript business logic code are very close together, with one being an intrinsic part of the other. One could argue, though, that PHP did just that already since its very beginnings, so not too much new here.

### File-based routing
Some frameworks advertise their mechanism of [file-based routing](https://nextjs.org/docs/routing/introduction), i.e. not having to declare HTTP routes / paths in code, but have them inferred implicitly from the app's directory structure. However, this is not at all new, either. PHP used to do just that - you place a `.php` into some folder and can call it from the browser by its file path. 

### Serverless
The idea of Serverless is (a) to have developers not having to worry about deploying, running and scaling their app in production (or even doing server administration), (b) to reduce boilerplate code for authentication, authorization, security and the web server itself and (c) to have very small, self-contained, single-purpose functions instead of giant coupled code bases. Serverless functions are triggered through an event (usually an HTTP call), then do their job, and are shut down again afterwards - they are entirely stateless. Sounds familiar? This is exactly what PHP scripts are. They are invoked per request and only live as long as the request itself. What is the novelty here? Indeed, Serverless is very, very close to what PHP scripts have been for years already. However, to be fair, Serverless (e.g. AWS Lambda) is - due to the way it is designed - probably much more scalable than PHP scripts running inside a single web server / FPM.

## RPC technologies
Another field where I found a lot of parallels between new, "fancy" technologies and old, "legacy" approaches is remote-procedure calls (RPC). RPC is an alternative approach to API design and comparable with resource-based paradigms such as REST or "query"-based approaches like GraphQL. 

### gRPC, tRPC, ...
The most popular framework today in this space is [gRPC](https://grpc.io/) by Google, but there are also [Cap'n'Proto](https://capnproto.org/), [tRPC](https://trpc.io/), [JSON-RPC](https://go.dev/blog/json-rpc) (less of a framework, rather a convention) and others. And then there used to be [**SOAP**](https://en.wikipedia.org/wiki/SOAP) for a long time. It's [considered outdated and deprecated](https://www.redhat.com/architect/apis-soap-rest-graphql-grpc) today. But in essence, it was doing the exact same thing - just in a slightly different way. Of course, gRPC has a few advantages over SOAP, especially more efficient message representation, thanks to binary encoding, and communication, thanks to HTTP/2 – on the other hand, though, SOAP has discoverability, thanks to to WSDL. The point I'm trying to make is that the fundamental concepts are almost exactly the same.

# Old ≠ bad
The central point I want to make with this article is: technology is not necessarily bad, just because it's old. Not even PHP (especially with [PHP 8.0](https://www.php.net/releases/8.0/en.php)). Many seemingly novel concepts have already been there before at some point, and not few of them do still have their reason to exist. When a new technology emerges, it is often times just an evolution, rather than a revolution - a slight improvement or the rejuvenation of a previous technology (cf. gRPC vs. SOAP, Serverless vs. PHP scripts, ...). Of course, this is important. Technology must be adapted to modern needs. But also, at the same time, developers should be careful about not ending up reinventing the wheel once again.

# Key takeaways
From this discussion, my key takaways are:

* **Technology is not bad, just because it's old.** Don't be fooled by the novelty bias.
* **Don't ever _rant_ about a technology** - especially not just because everyone else on Twitter does. Even [dreaded](https://survey.stackoverflow.co/2022/#technology-most-loved-dreaded-and-wanted) things like PHP or VBA have their right to exist. You can anyways only judge about a technology, if you have actually worked with it extensively. 
* **Focus on building features**, not tools - unless you really have a need to.