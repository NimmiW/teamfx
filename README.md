# Python Flask app on Azure App Service Web

This is a minimal sample app that demonstrates how to run a Python Flask application on Azure App Service Web.

This repository can directly be deployed to Azure App Service.

For more information, please see the [Python on App Service Quickstart docs](https://docs.microsoft.com/en-us/azure/app-service-web/app-service-web-get-started-python).

# Build and Run the project

` D:\coursework\L4S2\GroupProject\repo\python-docs-hello-world> pip install -r requirements.txt`

Then run,

`D:\coursework\L4S2\GroupProject\repo\python-docs-hello-world> python main.py`

# Contributing

- Erandi: backtesting module
- Dilmi: optimzation module
- Sanka: predictions modules
- Nimmi: anomalies module

# Folder Structure
The main folder structure is as follows.

```
python-docs-hello-world/
    /yourapp
        /main.py
        /anomalies
        /backtesting
        /predictions
        /optimization
        /static/
             /anomalies
             /backtesting
             /predictions
             /optimization
        /templates/
             /anomalies
             /backtesting
             /predictions
             /optimization
        /requirements.txt
```

Module functions should be wriiten on individual module folders in the root directory.

Images should be placed on static folder.

Html templates should be placed on templates folder.

Please observe the behavior of btr module which is a tes module to study the work flow.

All the dependencies should be listed in reuirements.txt at the root directory.

All the routes should be listed at main.py at the relevant section.