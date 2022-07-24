# Installation

1. Clone this repo

```
git clone https://github.com/ivv101/Chandra-ACIS-clusters-app

cd Chandra-ACIS-clusters-app
```

1. [optional] Create environment

```
mamba create -n Chandra-ACIS-clusters-app-env
mamba activate Chandra-ACIS-clusters-app-env
```

2. Install dependencies

```
mamba env update --file environment.yml
```

3. Run Bokeh server

```
bokeh serve app/app1.py --show
```




<!-- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ivv101/tst/main?urlpath=/proxy/5006/bokeh-app)


This repository demonstrates how to run a Bokeh server from within Binder. To do so, we did the following things:

1. Created a `bokeh-app` directory in the repo with a `main.py` file in it. This is the application that will be served. We've added the
   [Bokeh weather example](https://github.com/bokeh/bokeh/tree/master/examples/app/weather) as a demo. 
2. Installed `bokeh` for the viz and `jupyter_server_proxy` which we'll use to direct people to the port on which Bokeh runs. See `environment.yml`.
3. Added a custom server extension (`bokehserverextension.py`) that will be run to direct people to the Bokeh app (which is run on a port)
4. Used `postBuild` to enable the `jupyter_server_proxy` extension, then set up and enable our custom server extension for Bokeh. 
5. Created a Binder link that uses `urlpath` to point users to the port on which the Bokeh server will run:

   ```
   https://mybinder.org/v2/gh/ivv101/tst/main?urlpath=/proxy/5006/bokeh-app
   ```
   
When people click on the Binder link, they should be directed to the running Bokeh app.
 -->