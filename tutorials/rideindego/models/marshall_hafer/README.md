## The Marshall Model

Marshall is our first model to predict the prices.


## What are we doing?

In our `marshall` model, we build the full code to predict the `duration` of a trip. The basic ideas are:

- easy to reproduce,
- easy to tune hyperparameters,
- can be deployed to the cloud,
- less entangles in different steps so that we can run each steps independently.

To illustrate the repetitions in this project, we have replicated the definitions of the same functions in different modules.

### Top Down

The whole pipeline involves many different components:

- ETL: get the raw data and save it; This is related to what problem we are solving; It is not quite related to the model we are using.
- Preprocessing: prepare the data for our model, this is closed related to what model we are developing.
- Model: the actual core model.
- Prediction: take in new data and return the predicted the results. This is closed related to the final usage of the model.
- Performance: some more benchmark of the model.


We ran into the following problems:

- Config management: sharing and applying configs in different steps.
  - File path management.
- Parameter management: hyperparameters are not systematically managed in training and testing.
- Artifact management: save, reload, and preserve artifacts.
  - Prepare folders for the artifcts;
  - Sync the data to and from the cloud.
- Pipeline mamangement: connecting the steps.
  - Run through a list of actions on the full dataset, in order. For example, when preprocessing the data, we might have an ordered list of functions;
  - Run through the data records in order, perform actions, and save.


## How about Hafer ML

The Hafer ML framework provides atomic elements on the above topics and also the glue to stick everything together.


