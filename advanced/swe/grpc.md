# gRPC

## What is gRPC

gRPC is a modern, high-performance framework that is an evolution of the remote procedure call (RPC) protocol [1]. 

At the application level, gRPC streamlines messaging between clients and back-end services. 

A typical gRPC client app will expose a local, in-process function or stub where the local function invokes another function on a remote machine. 

What appears to be a local call essentially becomes a transparent out-of-process call to a remote service. 

The RPC plumbing abstracts the point-to-point networking communication, serialization, and execution between computers.

## How does gRPC work

gRPC uses Protocol Buffers to interchange messages between client and server [3]. 

Once the server receives the client request, it executes the method and sends the client response back with a status code and optional metadata. 

gRPC allows clients to specify wait time to allow the server to respond before the RPC call is terminated.

## Benefits of gRPC

gRPC uses HTTP/2 for its transport protocol which features many advanced capabilities:

- gRPC is lightweight and highly performant. 

- gRPC provides a binary framing protocol for data transport unlike HTTP 1.1 which is text-based.

- Multiplexing support for sending multiple parallel requests over the same connection while HTTP 1.1 limits processing to one request/response message at a time.

- Bidirectional full-duplex communication for sending both client requests and server responses simultaneously.

- Built-in streaming enabling requests and responses to asynchronously stream large data sets.

- Header compression that reduces network usage.

- Loose coupling between client and server makes it easy to make changes. 

- gRPC allows integration of API’s programmed in different languages. 

## gRPC vs REST

- **Payload Format:** REST uses JSON for exchanging messages between client and server whereas gRPC uses Protocol Buffers which can be compressed better than JSON, so gRPC transmits data over networks more efficiently.

-  **Transfer Protocols:** REST uses the HTTP 1.1 protocol which is text-based whereas gRPC uses the HTTP/2 binary protocol that supports header compression for efficient network usage.

- **Streaming vs Request-Response:** REST supports the Request-Response model of HTTP1.1 whereas gRPC uses bi-directional streaming capabilities of HTTP/2 where the client and server send a sequence of messages to each other using a read-write stream.

## Protocol Buffers

gRPC embraces an open-source technology called **Protocol Buffers** which provide a highly efficient and platform-neutral serialization format for serializing structured messages that services send to each other [1]. 

Using a cross-platform Interface Definition Language (IDL), developers define a service contract for each microservice. 

The contract is implemented as a text-based `.proto` file which describes the methods, inputs, and outputs for each service.

The same contract file can be used for gRPC clients and services built on different development platforms.


Using the proto file, the Protobuf compiler (protoc) generates both client and service code for your target platform which includes:

- Strongly typed objects that are shared by the client and service which represent the service operations and data elements for a message.

- A strongly typed base class with the required network plumbing that the remote gRPC service can inherit and extend.

- A client stub that contains the required plumbing to invoke the remote gRPC service.

At run time, each message is serialized as a standard Protobuf representation and exchanged between the client and remote service. 

Unlike JSON or XML, Protobuf messages are serialized as compiled binary bytes.


## gRPC usage

Favor gRPC for the following scenarios [1]:

- Synchronous backend microservice-to-microservice communication where an immediate response is required to continue processing.

- Polyglot environments that need to support mixed programming platforms.

- Low latency and high throughput communication where performance is critical.

- Point-to-point real-time communication: gRPC can push messages in real time without polling and has excellent support for bi-directional streaming.

- Network constrained environments: binary gRPC messages are always smaller than an equivalent text-based JSON message.


## gRPC support in .NET

gRPC is integrated into .NET Core 3.0 SDK and later.



## API vs Webhook

A quick intro to event-based and request-based data transfers [5]. 

An API is the interface of a piece of software that connects it to other pieces of software. It receives requests in a pre-specified format, processes them internally, and returns the requested information (such as JSON or XML format). 

An API is **request-driven** which means it is triggered when a call is made.

A webhook is **event-driven** which means the webhook broadcasts new information when something changes in the system. In fact, webhook is also referred to as a **reverse API**. 

We can view an API as a **pull system** and a webhook as a **push system**.



## References

[1] [gRPC](https://docs.microsoft.com/en-us/dotnet/architecture/cloud-native/grpc)

[2] [How To Build An AutoML API](https://towardsdatascience.com/how-to-build-an-automl-api-8f2dd5f687d1)

[3] [How to use gRPC API to Serve a Deep Learning Model using TF Serving](https://towardsdatascience.com/serving-deep-learning-model-in-production-using-fast-and-efficient-grpc-6dfe94bf9234)

[4] [TF Serving - Auto Wrap your TF or Keras model and Deploy it with a production-grade GRPC Interface](https://medium.com/data-science-engineering/using-tensorflow-serving-grpc-38a722451064)

[5] [Webhook vs API — Which One Do You Need?](https://towardsdatascience.com/webhook-vs-api-which-one-do-you-need-8c430f8ea71b)

