
# Repo will activelly updated 

## File Description 
- Allez au Diable.py , file for EDA. Will be refactored to .ipynb
- DataPreps.py, data preprocessing
- HyperparametersFinder.py, finding optimal params for a given model
- Main.py, main script for this repo
- Main_Enum.py, all enums
- ModelBuilder.py, building model, this is for finding feature importance

## How To Run 
- Make sure you are using same environment. (Please create new env)
```
    pip install -r requirements.txt
```
### Find feature importance using default params
```
    python Main.py -tr='decision_tree' -target='popularity'
```
- For more information you can check the params.yaml

### Find optimal parameters for model ( On Going )
```

```

### Important notes for params.yaml
- constant -> this used within the script globally
- cmd -> this used only within Main.py


