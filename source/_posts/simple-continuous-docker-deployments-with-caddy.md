---
title: Simple continuous Docker deployments with Caddy
date: 2021-11-28 12:52:11
tags: [sysadmin, linux]
---

# CI and CD

![](https://apps.muetsch.io/images/o:auto/rs,s:640?image=https://muetsch.io/images/cicd1.png)
([Source](https://harness.io/blog/continuous-delivery-tools/))

Continuous deployment (CD) refers to the process of frequently delivering software updates through automated deployments and usually goes hand in hand with continuous integration (CI). The input to a CI pipeline (e.g. running on [GitLab CI](https://docs.gitlab.com/ee/ci/), [GitHub Actions](https://github.com/features/actions), [Jenkins](https://www.jenkins.io/), [TeamCity](https://www.jetbrains.com/teamcity/) or [GoCD](https://www.gocd.org/)) is raw source code, while the output usually is a well-tested, self-contained build artifact, that contains your application. This could be a binary executable (common with compiled languages like C, C++, Go), a JAR or WAR file (in the case of JVM languages), a DEB or RPM package, a Python [wheel](https://www.python.org/dev/peps/pep-0427/), and, among all these other options, also a Docker image. Let us focus on the latter, as this is one of the most common and most flexible approaches nowadays. So, after having been built and stored in some Docker library (e.g. publicly on DockerHub or in a private container registry like [GitLab](https://docs.gitlab.com/ee/user/packages/), [Artifactory](https://jfrog.com/artifactory/) or [Nexus](https://www.sonatype.com/products/repository-pro)), the Docker image must still somehow find its way to the server, where the application shall be deployed - at least in the case of web applications. This is where CD comes to play. 

# Docker + Caddy (+ GitLab CI)
After a new Docker image was built, you somehow need to tell the host machine's Docker daemon to pull and run that new image. A common practice is to use tools like [Watchtower](https://containrrr.dev/watchtower/), which can be configured to either regularly poll a registry for new image tags or react to webhooks. However, I didn't want to run another extra software tool solely for the purpose of updating my Docker container. As a consequence, I came up with a minimalist, custom-built solution involving nothing but [Caddy](https://caddyserver.com/) (with the [caddy-exec](https://github.com/abiosoft/caddy-exec) plugin) – which runs on my host machine as a web server and reverse proxy anyway – and a shell script. 

Conceptually, Caddy spawns a webhook endpoint, that is called as part of the last step of my CI pipeline and then runs a few simple bash commands to update my container. Here is how to set everything up.

1. Get a Caddy release containing the `caddy-exec` plugin from their [downloads page](https://caddyserver.com/download?package=github.com%2Fabiosoft%2Fcaddy-exec). I assume you have worked with Caddy before, know how to run and use it and are familiar with `Caddyfile`. 
1. Allow the `caddy` user to access the Docker daemon by adding it to the `docker` group:
    ```bash
    $ sudo usermod -a G docker caddy
    ```
    You might need to reboot or re-login afterwards, in order for the changes to take effect.

    If you're concerned about security, you might want to come up with a more elaborate solution here, e.g. involving [authorization](https://docs.docker.com/engine/extend/plugins_authorization/) or so.
1. Optional: log in to private container registry and make credentials accessible by `caddy` user. If you're concerned about security, you might want come up with a cleaner solution here. 
    ```bash
    $ docker login registry.gitlab.com
    $ chown -R $USER:caddy $HOME/.docker/
    $ chmod 770 $HOME/docker
    $ chmod 640 $HOME/.docker/config.json
    ```
1. Create a new bash script, e.g. `update_app.sh`, that contains the logic for updating your container.
    ```bash
    #!/bin/bash

    # Replace your-username with your actual user or leave out `--config` when using public registry
    DC="/home/your-username/.docker"
    
    # Pull image `registry.gitlab.com/your-group/your-project` from non-public GitLab container registry
    docker --config $DC pull registry.gitlab.com/your-group/your-project

    # Stop old container
    docker --config $DC stop your-app-1 && docker --config $DC rm your-app-1

    # Run new container (e.g. some web app listeing on port 8080)
    docker --config $DC run -d -p 127.0.0.1:8080:8080 --name your-app-1 registry.gitlab.com/your-group/your-project:latest
    ```

1. Make the file executable by `caddy` user:
    ```bash
    $ sudo chown caddy:caddy update_app.sh
    $ sudo chmod+x update_app.sh
    ```
1. Edit your Caddyfile to look like so:
    ```
    your-app.example.org {
        # Assuming you also use Caddy as a reverse proxy to your app, not required
        reverse_proxy http://localhost:8080

        # Request matcher, based on query param and token header field
        @your-app_push {
            query project=your-app

            # A "pre-shared key" to "authenticate" against this endpoint
            # Replace this by some random string
            header X-Token o47iUJnhq3vZZIMT
        }

        route /push-hook* {
            exec @your-app_push {
                command bash
                args /home/your-username/update_app.sh
                timeout 120s
            }
        }
    }
    
    ```
1. Restart (or reload) Caddy for changes to take effect. Now, you should be able to call the endpoint to kick off your update script:
    ```bash
    $ curl -X GET -H "X-TOKEN: o47iUJnhq3vZZIMT" https://your-app.example.org/push-hook?project=your-app
    ```
1. Include this webhook request to your pipeline. Here, I'm using GitLab CI, you would need to adapt this logic to the syntax of whatever CI server you use. This example is a minimal pipeline definition for building a Docker image (using [Kaniko](https://blog.alexellis.io/quick-look-at-google-kaniko/)), pushing it to GitLab container registry and eventually calling the above webhook endpoint. 
    ```yaml
    # .gitlab-ci.yml

    stages:
        - publish
        - notify

    workflow:
        rules:
            # Run pipeline only for tags
            - if: $CI_COMMIT_TAG

    build-publish-docker:
        stage: publish
        image:
            name: gcr.io/kaniko-project/executor:debug
            entrypoint: [ "" ]
        script:
            - echo "{\"auths\":{\"$CI_REGISTRY\":{\"username\":\"$CI_REGISTRY_USER\",\"password\":\"$CI_REGISTRY_PASSWORD\"}}}" > /kaniko/.docker/config.json
            - /kaniko/executor --context $CI_PROJECT_DIR --cache=true
            --dockerfile $CI_PROJECT_DIR/Dockerfile --destination $CI_REGISTRY_IMAGE:$CI_COMMIT_TAG --destination $CI_REGISTRY_IMAGE:latest

    notify-webhook:
        stage: notify
        allow_failure: true

        # DEPLOY_TOKEN must be defined as a CI variable under Settings -> CI/CD -> Variables 
        script:
            - 'curl -H "X-Token: $DEPLOY_TOKEN" https://your-app.example.org/push-hook?project=$CI_PROJECT_NAME'
    ```
