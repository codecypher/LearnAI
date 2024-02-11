# PyTorch DataLoader

Here are some code snippets using PyTorch DataLoader

## Custom image folders

```py
    # Create the dataset
    train_ds = dset.ImageFolder(root=args.data_root/train, transform=transform)

    # Create the dataloader
    train_dl = torch.utils.data.DataLoader(train_ds,


    # Create the dataset
    trest_ds = dset.ImageFolder(root=args.data_root/test, transform=transform)

    # Create the dataloader
    test_dl = torch.utils.data.DataLoader(test_ds,
```

### dcgan_celeba.py

```py
def load_data(args, show_images=False):
    """
    Load the datasets
    """
    # dset.CelebA(args.data_root, split='all', download=True)

    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Create the dataset
    train_ds = dset.ImageFolder(root=args.data_root, transform=transform)

    # Create the dataloader
    train_dl = torch.utils.data.DataLoader(train_ds,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=args.workers)

    args.experiment.log_dataset_info(train_ds)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")

    if show_images:
        # Plot some training images
        real_batch = next(iter(train_dl))
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(
            vutils.make_grid(real_batch[0].to(device)[:64],
                             padding=2,
                             normalize=True).cpu(),
                             (1, 2, 0)))

    return train_dl, device
```


### gan_mnist.py

```py
def load_mnist_data(args):
    """
    Load MNIST datasets

    We create a dataset object and a data loader that batches and shuffles
    post-transformation images for us.
    """
    # Image processing
    # transform = transforms.Compose([
    #                 transforms.ToTensor(),
    #                 transforms.Normalize(mean=(0.5, 0.5, 0.5),   # 3 for RGB channels
    #                                      std=(0.5, 0.5, 0.5))])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5],  # 1 for greyscale channels
                             std=[0.5])])

    # DataLoader is the PyTorch module to combine the image
    # and its corresponding label in a package.

    # We can define a simple transformation that converts images to tensors
    # then applies a standard normalization procedure for easier training.
    transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5), (0.5))
                            ])

    # Create a dataset object and a data loader that batches
    # and shuffles post-transformation images for us.
    trainset = datasets.MNIST(root=args.root,
                              download=True,
                              train=True,
                              transform=transform)

    train_loader = DataLoader(dataset=trainset,
                              batch_size=args.batch_size,
                              shuffle=True)
    return train_loader

```

### pytorch_cifar.py


```py
def load_data():
    """
    Load and normalize CIFAR10 dataset.

    We are creating data loaders which allows us to load data in batches
    such as when you have large data set it will not fit into memory for training.

    You can try different batch sizes by doubling  (128, 256, 512)
    until your GPU/Memory fits it and processes it faster.
    When it starts to slow down you can decrease the batch size by one step.

    shuffle=True gives randomization to the data
    """
    batch_size = 4
    num_workers = 4    # number of sub-processes to use for data loading (parallelization)
    pin_memory = True  # dataloader copies Tensors to pinned memory before returning them

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_ds = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_ds = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    return train_loader, test_loader, classes
```


### pytorch_mnist.py

```py

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_data = datasets.MNIST("../data",
                                train=True,
                                download=True,
                                transform=transform)
    test_data = datasets.MNIST("../data",
                               train=False,
                               transform=transform)

    trainloader = torch.utils.data.DataLoader(train_data, **train_kwargs)
    testloader = torch.utils.data.DataLoader(test_data, **test_kwargs)
```


### wgan_mnist.py

```py
def load_data(args):
    """
    Load and normalize the MNIST dataset.

    We are creating data loaders which allows us to load data in batches
    when we have large data set and it will not fit into memory for training.

    We can try different batch sizes by doubling (128, 256, 512)
    until the GPU/Memory fits it and processes it faster.

    When it starts to slow down you can decrease the batch size by one step.

    References:
      https://blog.paperspace.com/dataloaders-abstractions-pytorch/
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = {'num_workers': args.num_workers, 'pin_memory': args.pin_memory} if device == 'cuda' else {}

    # Configure dataloader
    # os.makedirs("../data", exist_ok=True)

    transform = transforms.Compose(
        [transforms.Resize(args.img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.5],[0.5])]
        #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_dataset = datasets.MNIST(
        root=args.root,         # root directory of dataset
        train=True,             # create dataset from training set
        download=True,          # download dataset from internet to root
        transform=transform
    )

    test_ds = datasets.MNIST(
        root=args.root,
        train=False,
        download=True,
        transform=transform
    )

    train_size = len(train_dataset) - args.val_size

    # train / validation split
    train_ds, val_ds = random_split(train_dataset,
                                    [train_size, args.val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,                   # Gives randomization to the data
        **kwargs
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        **kwargs
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size_test,
        shuffle=False,
        **kwargs
    )

    print("train: {}  val: {}  test: {}".format(
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset),
    ))

    return train_loader, val_loader, test_loader
```


## References

[PyTorch Lightning: DataModules, Callbacks, TPU, and Loggers](https://krypticmouse.hashnode.dev/pytorch-lightning-datamodules-callbacks-tpu-and-loggers)

[Training PyTorch on Cloud TPUs](https://ultrons.medium.com/training-pytorch-on-cloud-tpus-be0649e4efbc)





