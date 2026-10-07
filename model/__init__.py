"""green-lory: the cost of green ammonia from wind and PV at every 1-degree cell of the world.

Modules, in the order a run uses them:

    config.py       paths and physical constants
    technology.py   the technology YAML: costs, annuities, per-cell finance and build overrides
    weather.py      hourly PV and wind profiles per cell
    plant.py        one plant at one cell: the PyPSA network, its constraints, the solve
    accounting.py   costs after the solve: water, land, cost shares, grid use
    land.py         land budgets and the capacity rule (how much ammonia a cell can make)
    land_table.py   building the land tables from MODIS, WDPA and GEBCO
    surface.py      running the plant over many cells

Run definitions are data: runs.yaml gives every setting of every run, by id.
"""
