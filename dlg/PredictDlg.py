import pickle
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load in memory the available aliments (ids)
aliments_ids = pd.read_csv('aliments.csv')

# Load in memory all the models 
models = dict()
missing_models = []
for aid in aliments_ids['id']:
    
    try: 
        # Load scaler and model
        scaler = joblib.load("models/" + aid + '.scaler.sav')
        loaded_model = pickle.load(open("models/" + aid + ".model.sav", 'rb'))

        models[aid] = [scaler, loaded_model]

    except FileNotFoundError:
        missing_models.append(aid)

def do(req): 
    '''
    Executes the prediction and returns the best n results
    Requires the following data from the req param: 
     - weekday      : the day of the week as an int from 0 to 6 (0 is Monday)
     - time         : the time of the day as a "h:mm" (or "hh:mm") string
     - nResults     : the number of predicted aliments to be returned            
    '''

    # Get the data from req
    try: 
        time = req['time']
    except KeyError: raise KeyError("'time' is a mandatory parameter or the POST request")
    try: 
        week_day = req['weekday']
    except KeyError: raise KeyError("'weekday' is a mandatory parameter or the POST request")
    try: 
        n_results = req['nResults']
    except KeyError: n_results = 3

    # Parse the time
    hour = int(time.split(':')[0])
    minute = int(time.split(':')[1])
    
    t = hour + (minute/60)
    
    features = np.array([[week_day, t]])
    
    predictions = pd.DataFrame()

    # Run all predictions for that time
    for aid in aliments_ids['id']:

        try:
            scaler = models[aid][0]
            model = models[aid][1]

            X = scaler.transform(features)

            y_pred = model.predict_proba(X)

            predicted = pd.DataFrame(np.array([[float(y_pred[0][1])]]), columns=['proba'], dtype=float)
            predicted['aliment_id'] = aid

            predictions = pd.concat([predictions, predicted], sort=False, ignore_index=True)

        except KeyError:
            pass
        
    
    # Extract the n highest proba
    highest_ranking = predictions.sort_values(by='proba', ascending=False).head(n_results)

    # Enrich the returned data with the aliment names
    highest_ranking = pd.merge(highest_ranking, aliments_ids, left_on='aliment_id', right_on="id")[['proba', 'aliment_id', 'name']]
    
    # Rename columns to fit JSON naming conventions
    highest_ranking.rename(columns={"aliment_id": "alimentId", "proba": "chosenProba"}, inplace=True)

    # Convert to json
    return highest_ranking.to_json(orient="records", force_ascii=False)
    


