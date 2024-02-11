## Security


## Typical SDLC Process

In most cases, every engineering project starts with a spec. Then, engineers begin implementing the software in code, building, and deploying the service.

The problem is that security audits are often forgotten until the very end of the project, just before deploying the new tool in production. 

What if you discover a security flaw at this point? In most cases, it will be very expensive to fix it. Thus, usually no security verifications are done before deployment.

We will see how to enforce practical threat modeling and service hardening into the microservices SDLC to ensure security is not forgotten.

Here we asdume your release is a full-fledged MLOps project where many microservices run and communicate on top of Kubeflow or Kubernetes.


## The Shifting-Left Principle

The solution to the problem we described above is to move the security considerations into the early stages of the SDLC process which is known as the _shifting-left principle_. 

By shifting left, the steps we must take to ensure the security of the applications and substrate execution environment are incorporated into the development process.

The following steps provide a simple checklist to follow:

1. Have the security team evaluate the design document and the architecture diagram to identify security flaws as you write it.

2. Refine the design document considering the outcomes of the previous step.

3. Continuously run tools such as linters during code development to provide actionable feedback to developers to quickly remediate security risks.

4. Scan the dependencies and software builds for security flaws before deployment.

5. Scan the environment where you will deploy your services for security flaws.

6. Continuously monitor a running deployment to ensure that everything runs smoothly with no security incidents.

7. Evaluate any significant changes to the application or infrastructure after the service deployment.

8. Follow the previous steps if you need to re-build or re-deploy anything in the system.

9. Automate most of the steps in this process.

10. Repeat this process until the service is retired.

What could go wrong? We can use a threat modeling framework like STRIDE to answer this question. Let us make sure that we know a few security key concepts first. 


## Key Security Concepts

To understand security, we need to know the fundamental concepts:

- **Authenticity:** When a non-genuine user poses as a legitimate user.

- **Integrity:** When an attacker tampers with data or a system configuration.

- **Non-repudiation:** The act of denying something happened such as deleting the system logs.

- **Confidentiality:** When sensitive data is exposed.

- **Availability:** When the system uptime is affected.

- **Authorization:** When the users or services in the system do not respect access levels and permissions.


## STRIDE

STRIDE is a mnemonic that assists in remembering the primary ways a system can be affected by an attacker:

- “S” for Spoofing: This is when a user or service poses as someone or something for malicious intent which affects authenticity.

- “T” for Tampering: This is when an attacker modifies the system for malicious intent such as changing config files which affects integrity.

- “R” for Repudiation: This is when a user or an application clears their footprint, denying doing something suspicious which affects non-repudiation.

- “I” for Information Disclosure: This  occurs when confidential information is revealed to unauthorized persons or parties which affects confidentiality.

- “D” for Denial of Service: This event affects the availability of a system which affects availability.

- “E” for Elevation of Privilege: This  occurs when a user or service takes a role that they are not authorized to take which affects authorization.



## The Docker Attack Surface

Docker is perhaps the most popular container engine whis is a set of Platform as a Service (PaaS) products that use OS-level virtualization to deliver software in packages called containers.

Docker runs on top of a host Operating System (OS) and allows multiple containers to be run; each container is isolated from another and bundles its own software, libraries, and configuration files; each container can communicate through well-defined channels.

- Docker daemon: The Docker daemon is the engine that creates and manages containers which runs in the background waiting for instructions. The Docker engine consists of the Docker daemon and the Docker client. 

- Docker client: The Docker client is the interface that interacts with the Docker daemon, passing the commands given by the user through the Command Line Interface CLI.

### Docker Threat Fundamentals

There are three boundaries within the Docker system:

- The Docker Client
- The Docker host
- The Docker Registry

First, we review the workflow of how we interact with each of these components.

### Docker Threat Modeling with Stride

STRIDE framework always asks: what could go wrong? 


#### Docker Client

For the Docker client, we are concerned with:

- Client compromise: an attacker may access the administrator’s desktop that runs the Docker binaries.

- Client authorization abuse: an attacker may abuse the client’s authorization to make privileged changes on the host machine.

- Dockerfile: the Dockerfile may be misconfigured.

We can apply STRIDE to the Docker client:

- Spoofing: Messages from the Docker client to the Docker daemon or registry could be intercepted by a malicious party and replayed. Transport Layer Security (TLS) can help establish secure communication between the different components.

- Tampering: An attacker could alter the Dockerfile used to build an image by taking control of the Docker client.

- Repudiation: This can also affect the Dockerfile used to build an image.

- Information Disclosure: An attacker that controls the client can discover sensitive information such as secrets that are hardcoded into configuration files (bad practice).

- Denial of Service: Excessive traffic from the client could overwhelm the daemon and cause it to crash, leading to a client compromise.

- Elevation of Privilege: Running a container as root affects what processes a container can run.

#### Docker Host

For the Docker host, we are concerned with:

- Isolation tampering: Prevent container escape attacks.

- The --privilege flag: Prevent containers from doing more than they are supposed to do.

- Insecure defaults: Set insecure defaults for how Docker operates.

- Misconfiguration: Setting configuration options that create attack surfaces for Docker components.

We can apply STRIDE to the Docker host:

- Spoofing: When a sidecar container has access to the primary container, it may violate the isolation principle and spoof the namespace used by the primary container.

- Tampering: An attacker who has client access can misconfigure the Docker daemon.

- Repudiation: An attacker can change the configuration of the Docker daemon.

- Information Disclosure: A misconfigured file system may expose sensitive information.

- Denial of Service: Unbounded memory usage from a running container can happen due to misconfiguration issues.

- Elevation of Privilege: Daemon privilege may be escalated if a username is not defined and the container runs as root which gives access to any other container on the same host, leading to isolation tampering.

#### Docker Registry

For the Docker registry, we are concerned with:

- Image security: The image you pull should not have security holes.

- Open Source Software (OSS) security: The OSS installed inside the image should be vetted.

- Docker registry security: Prevent unauthorized access to the registry.

We can apply STRIDE to the Docker registry:

- Spoofing: An image may be spoofed with malicious images or libraries as part of the container layers.

- Tampering: An attacker may insert a malicious image into the registry and have organizations use it. Another issue is when an attacker introduces OSS libraries on repos such as PyPI and the user adds them to their images.

- Repudiation: Unsigned malicious images may be committed without an attacker.

- Information Disclosure: Sensitive data such as secrets may be hardcoded into the Dockerfile, passed in the container image, and exposed at runtime.

- Denial of Service: The registry may not restrict the amount of content accepted by a user, causing issues to the registry’s security.

- Elevation of Privilege: Container image permissions may be elevated using the — privilege flag, leading to a severe security breach.


## The Kubernetes Attack Surface

Kubernetes is an open-source container orchestration system for automating software deployment, scaling, and management. 

### Kubernetes Components

At the highest level, Kubernetes consists of two major components: the control plane and the data plane.




## References

[What You Love to Ignore in Your Data Science Projects](https://towardsdatascience.com/what-you-love-to-ignore-in-your-data-science-projects-208754eea8e8)

[The Docker Attack Surface](https://medium.com/geekculture/the-docker-attack-surface-5184a36a23ca)

[The Kubernetes Attack Surface](https://medium.com/geekculture/the-kubernetes-attack-surface-7d1f854f9d9)

[Your Docker Setup is Like a Swiss Cheese; Here’s How to Fix it!](https://medium.com/geekculture/your-docker-setup-is-like-a-swiss-cheese-heres-how-to-fix-it-cd1f49f40256)


