# GPO

## Overview
This the `global_port_optimisation` model itself. 

The table below describes the function each file within this directory.

| File                  | Description |
| --------------------- | ----------- |
| configs.py            | Definition of universal model configurations.       |
| constraints.py        | Set of constraints required for global transport model        |
| driver.py             | This function takes scenario information defined by client and creates an instance of the optimisation class. It also produces output data and some figures.        |
| model.py              | This is the GPO class, where we initialise the model with the client's desired parameters and perform a model run. This is the main class that the client should interact with.        |
| optimiser.py          | Class designed for optimising the ammonia network to various shipping destinations        |
| params.py             | Definition of universal model parameters.        |
| select_locations.py   | Selects locations for use in optimisation. Top locations above a certain level of production are selected.        |
| toolbox.py            | A load of helper functions.        |


## TODO

- [ ] Merge `select_locations.py` into the `toolbox.py` script
- [ ] Create script with postprocessing functions to wrangle data
- [ ] Create script with plotting functions to plot outputs
- [ ] Create a streamlit app to visualise results