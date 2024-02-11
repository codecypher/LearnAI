# PyTorch LSTMs for Time Series Forecasting

## Curating Data to pass it to an LSTM model

We will be working with the `date` and `close` fields from the above table which means we will build an LSTM model that predicts given the close value of previous n days, what would the close value be on the current day.

The number of steps which we look in the past is commonly known as the number of lags or _lags_.

We will make it an input from the user so that they can experiment with it.

There are a series of steps that we need to follow to wrangle the data in a format that can be loaded in the PyTorch model:

- Extract the columns of interest from the dataframe: (Close and Date columns).

- Shift the dataframe down by number of lags (times) and remove the first number of lags rows.

  This will give us input-output pairs in a columnar format that could then be converted into a list of tuples for input and a list of floats for the output value of the stock price.

  The logic for this is defined in the function below:

```py
    def curate_data(pth, price_col, date_col, n_steps):
        """Read the dataset and based on n_steps/lags to consider in the time series, create input output pairs
        Args:
            pth ([str]): [Path to the csv file]
            price_col ([str]): [The name of column in the dataframe that holds the closing price for the stock]
            date_col ([str]): [The nameo oc column in the dataframe which holds dates values]
            n_steps ([int]): [Number of steps/ lags based on which prediction is made]
        """
        df = pd.read_csv(pth)

        # Create lags for the price column
        for idx in range(n_steps):
            df[f"lag_{idx + 1}"] = df[price_col].shift(periods = (idx + 1))

        # Create a dataframe which has only the lags and the date
        new_df = df[[date_col, price_col] + [f"lag_{x + 1}" for x in range(n_steps)]]
        new_df = new_df.iloc[n_steps:-1, :]

        # Get a list of dates for which these inputs and outputs are
        dates = list(new_df[date_col])

        # Create input and output pairs out of this new_df
        ips = []
        ops = []
        for entry in new_df.itertuples():
            ip = entry[-n_steps:][::-1]
            op = entry[-(n_steps + 1)]
            ips.append(ip)
            ops.append(op)

        return (ips, ops, dates)
```

- To feed data into a model in PyTorch, we need to create a dataloader which generates batches of inputs and outputs.

- The pivotal function is the `getitem` function which feeds inputs and outputs. We wrap them into torch tensors from the lists because pytorch models only accepts data that has datatype of pytorch tensor. Since the prices are decimal numbers, we wrap the data into float tensors.

- To address overfitting, we perform a train-validation split with 70% of the data in train set and 30% data in validation set.

```py
    class StockTickerDataset(Dataset):
        """This class is the dataset class which is used to load data for training the LSTM
        to forecast timeseries data
        """

        def __init__(self, inputs, outputs):
            """Initialize the class with instance variables
            Args:
                inputs ([list]): [A list of tuples representing input parameters]
                outputs ([list]): [A list of floats for the stock price]
            """
            self.inputs = inputs
            self.outputs = outputs

        def __len__(self):
            """Returns the total number of samples in the dataset
            """
            return len(self.outputs)

        def __getitem__(self, idx):
            """Given an index, it retrieves the input and output corresponding to that index and returns the same
            Args:
                idx ([int]): [An integer representing a position in the samples]
            """
            x = torch.FloatTensor(self.inputs[idx])
            y = torch.FloatTensor([self.outputs[idx]])

            return (x, y)
```

- One more crucial aspect that is very important from a deep learning perspective is to avoid _covariate shift_.

  One way to accomplish this is _standardization_ of the input data which converts all the input parameters to have a mean of 0 and standard deviation of 1. Internally in the different layers of the network, _batch normalization_ is used to avoid covariate shift problems.

  The following function therefore standardizes the data based on input i.e. train distribution.

```py
    def standardize_data(X, SS = None, train = False):
        """Given a list of input features, standardizes them to bring them onto a homogenous scale
        Args:
            X ([dataframe]): [A dataframe of all the input values]
            SS ([object], optional): [A StandardScaler object that holds mean and std of a standardized dataset]. Defaults to None.
            train (bool, optional): [If False, means validation set to be loaded and SS needs to be passed to scale it]. Defaults to False.
        """
        if train:
            SS = StandardScaler()   
            new_X = SS.fit_transform(X)
            return (new_X, SS)
        else:
            new_X = SS.transform(X)
            return (new_X, None)
```

- After we split the data into train and validation datasets and standardize them, we will create pytorch dataloaders to load the data and feed it to the PyTorch model.

```py
    def getDL(x, y, params):
        """Given the inputs, labels and dataloader parameters, returns a pytorch dataloader
        Args:
            x ([list]): [inputs list]
            y ([list]): [target variable list]
            params ([dict]): [Parameters pertaining to dataloader eg. batch size]
        """
        training_set = stockTickerDataset(x, y)
        training_generator = torch.utils.data.DataLoader(training_set, **params)
        return training_generator
```

Now that we have all the data ready in the necessary format, we shall switch gears to define an LSTM model for predicting stock prices.

## Defining the LSTM model architecture

Long-Short-Term-Memory (LSTM) is a Recurrent Neural Network that is used for modelling problems involving sequences. There are several input-output use-cases possible for sequence problems such as one to one (word translation), one to many (image caption generation), many to many (laguage translation).

In our case we have many to one which means that given a list of inputs, we are predicting one output.

For sequences of reasonably short lengths (less than 15–20 units per record), LSTMs do a wonderful job of decoding the correlations and capturing them to build a robust model but because of **vanishing gradient problems**, they cannot capture long term correlations.

For now, let us focus on creating an LSTM PyTorch model.

As we can see below, our model consists of an LSTM layer and two fully connected linear layers.
The LSTM layer needs a three dimensional input of the form (seq_len, batch_size, input_size).

```py
    import torch
    from torch import nn

    class ForecasterModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, n_lyrs = 1, do = .05, device = "cpu"):
            """Initialize the network architecture
            Args:
                input_dim ([int]): [Number of time lags to look at for current prediction]
                hidden_dim ([int]): [The dimension of RNN output]
                n_lyrs (int, optional): [Number of stacked RNN layers]. Defaults to 1.
                do (float, optional): [Dropout for regularization]. Defaults to .05.
            """
            super(forecasterModel, self).__init__()

            self.ip_dim = input_dim
            self.hidden_dim = hidden_dim
            self.n_layers = n_lyrs
            self.dropout = do
            self.device = device

            self.rnn = nn.LSTM(input_size = input_dim, hidden_size = hidden_dim, num_layers = n_lyrs, dropout = do)
            self.fc1 = nn.Linear(in_features = hidden_dim, out_features = int(hidden_dim / 2))
            self.act1 = nn.ReLU(inplace = True)
            self.bn1 = nn.BatchNorm1d(num_features = int(hidden_dim / 2))

            self.estimator = nn.Linear(in_features = int(hidden_dim / 2), out_features = 1)


        def init_hiddenState(self, bs):
            """Initialize the hidden state of RNN to all zeros
            Args:
                bs ([int]): [Batch size during training]
            """
            return torch.zeros(self.n_layers, bs, self.hidden_dim)

        def forward(self, input):
            """Define the forward propogation logic here
            Args:
                input ([Tensor]): [A 3-dimensional float tensor containing parameters]
            """
            bs = input.shape[1]
            hidden_state = self.init_hiddenState(bs).to(self.device)
            cell_state = hidden_state

            out, _ = self.rnn(input, (hidden_state, cell_state))

            out = out.contiguous().view(-1, self.hidden_dim)
            out = self.act1(self.bn1(self.fc1(out)))
            out = self.estimator(out)

            return out

        def predict(self, input):
            """Makes prediction for the set of inputs provided and returns the same
            Args:
                input ([torch.Tensor]): [A tensor of inputs]
            """
            with torch.no_grad():
                predictions = self.forward(input)

            return predictions
```

Batch size is a training hyperparameter which can be set at the time of training, seq_len and input size are two parameters that we need to be aware of from our dataloader.

Since we are going to look at `n_lags` historical stock prices, our `seq_len` is 3 and we are only looking at one feature which is stock price so our `input_features` is only 1.

Also our hidden and cell states are defined to be all zeros, but they could also be defined as random numbers at the start of training.

The `forward` method contains the logic for the forward propagation through the network. We pass the input through an LSTM layer then through a fully connected layer. This is followed by a batch normalization layer to prevent internal covariate shift and a non-linear ReLU activation layer. Finally we pass it through the output estimator layer which gives us the predicted stock price.

The `predict` method just implements the forward pass but by switching off the gradient tracking functionality since we only want the prediction and do not want to do any back-propagation.

This is just a basic architecture that can be tweaked and modified per the needs of the problem. Feel free to add or subtract fully connected layers, change the hidden dimensions and the number of hidden layers inside the LSTM layer and regularise using more dropout layers as you deem fit.

## Training the model

We have now reached the crucial part of training the model. We will have to define a lot of hyperparameters before we start the training and here we use `streamlit` to allow the user to interactively define them.

Some of these are the batch size, learning rate, the optimizer, number of hidden layers, neurons in the hidden layer, etc.

Once these are selected, you can hit the submit button to see the model train in action.

## Making predictions on unseen data

Once we have a trained model, we can perform predictions on previously unseen data that we set aside as validation data at the beginning of training.

We will do that and plot it to analyze the performance of the model.


## References

[PyTorch LSTMs for time series forecasting of Indian Stocks](https://medium.com/analytics-vidhya/pytorch-lstms-for-time-series-forecasting-of-indian-stocks-8a49157da8b9)


[Training Time Series Forecasting Models in PyTorch](https://towardsdatascience.com/training-time-series-forecasting-models-in-pytorch-81ef9a66bd3a)

