# Dependencies
from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# Your API definition
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if lr:
        try:

            json_ = request.get_json(force=True)
            json_ = pd.DataFrame(json_)
            
            df = pd.DataFrame(columns = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak', 'Sex_F', 'Sex_M',
                                             'ChestPainType_ASY', 'ChestPainType_ATA', 'ChestPainType_NAP',
                                                 'ChestPainType_TA', 'RestingECG_LVH', 'RestingECG_Normal',
                                                 'RestingECG_ST', 'ExerciseAngina_N', 'ExerciseAngina_Y',
                                                 'ST_Slope_Down', 'ST_Slope_Flat', 'ST_Slope_Up'])
           
            json_ = pd.DataFrame(json_)
            
            standardScaler = StandardScaler()
            
            columns_to_scale = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
            
            json_[columns_to_scale] = standardScaler.fit_transform(json_[columns_to_scale])
            
            ds = pd.DataFrame(json_)
                
            def num(ds):
                if ((ds['Age'].iloc[0] != np.nan) & (ds['RestingBP'] != np.nan) & (ds['Cholesterol'] != np.nan)
                    & (ds['MaxHR'] != np.nan) & (ds['Oldpeak'] != np.nan)).all():

                    d_age = ds['Age'].iloc[0]
                    d_rest = ds['RestingBP'].iloc[0]
                    d_chol = ds['Cholesterol'].iloc[0]
                    d_maxhr = ds['MaxHR'].iloc[0]
                    d_oldpeak = ds['Oldpeak'].iloc[0]
                    
                return d_age, d_rest, d_chol, d_maxhr, d_oldpeak
            
            def sex(ds):
                if(ds['Sex'].iloc[0] == 'M'):
                    d_f = 0
                    d_m = 1
                else:
                    d_f = 1
                    d_m = 0
                       
                return d_f, d_m
            
            def ChestPainType(ds):
                if(ds['ChestPainType'].iloc[0] == 'ATA'):
                    d_ata = 1
                    d_nap = 0
                    d_asy = 0
                    d_ta = 0
                elif(ds['ChestPainType'].iloc[0] == 'NAP'):
                    d_ata = 0
                    d_nap = 1
                    d_asy = 0
                    d_ta = 0
                elif(ds['ChestPainType'].iloc[0] == 'ASY'):
                    d_ata = 0
                    d_nap = 0
                    d_asy = 1
                    d_ta = 0
                else:
                    d_ata = 0
                    d_nap = 0
                    d_asy = 0
                    d_ta = 1
                    
                return d_ata, d_nap, d_asy, d_ta
    
            def RestingECG(ds):
                if(ds['RestingECG'].iloc[0] == 'Normal'):
                    d_norm = 1
                    d_st = 0
                    d_lvh = 0
                elif(ds['RestingECG'].iloc[0] == 'ST'):
                    d_norm = 0
                    d_st = 1
                    d_lvh = 0
                else:
                    d_norm = 0
                    d_st = 0
                    d_lvh = 1
                    
                return d_norm, d_st, d_lvh
                
            def ExerciseAngina(ds):
                if(ds['ExerciseAngina'].iloc[0] == 'N'):
                    d_n = 1
                    d_y = 0
                else:
                    d_n = 0
                    d_y = 1
                    
                return d_n, d_y
            
            def ST_Slope(ds):
                if(ds['ST_Slope'].iloc[0] == 'Up'):
                    d_up = 1
                    d_flat = 0
                    d_down = 0
                elif(ds['ST_Slope'].iloc[0] == 'Flat'):
                    d_up = 0
                    d_flat = 1
                    d_down = 0
                else:
                    d_up = 0
                    d_flat = 0
                    d_down = 1
                    
                return d_up, d_flat, d_down
            
            
            df = df.append({'Age' : num(ds)[0], 'RestingBP' : num(ds)[1], 'Cholesterol' : num(ds)[2],
                            'MaxHR' : num(ds)[3], 'Oldpeak' : num(ds)[4], 'Sex_F' : sex(ds)[0],
                            'Sex_M' : sex(ds)[1],'ChestPainType_ASY': ChestPainType(ds)[2],'ChestPainType_ATA': ChestPainType(ds)[0], 
                            'ChestPainType_NAP': ChestPainType(ds)[1], 'ChestPainType_TA': ChestPainType(ds)[3], 
                            'RestingECG_LVH': RestingECG(ds)[2],'RestingECG_Normal': RestingECG(ds)[0], 
                            'RestingECG_ST': RestingECG(ds)[1], 'ExerciseAngina_N':ExerciseAngina(ds)[0], 
                            'ExerciseAngina_Y':ExerciseAngina(ds)[1], 'ST_Slope_Down': ST_Slope(ds)[2],
                            'ST_Slope_Flat': ST_Slope(ds)[1], 'ST_Slope_Up': ST_Slope(ds)[0]}, ignore_index=True)
                
    
            dataset = np.array(df)
                
            prediction = list(lr.predict(dataset))

            return jsonify({'prediction': str(prediction)})
        
        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except:
        port = 12345

    lr = joblib.load("Final-Model.sav")
    print ('Model loaded')
    model_columns = joblib.load("Final-Model-Columns.sav")
    print ('Model columns loaded')

    app.run(port=port, debug=True)