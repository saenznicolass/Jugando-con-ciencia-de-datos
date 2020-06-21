from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing
import json

app = Flask(__name__)
api = Api(app)

with open('modeloEscogido.pickle', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pickle', 'rb') as f:
    scaler = pickle.load(f)

parser = reqparse.RequestParser()
parser.add_argument('query')

class PredictEmotion(Resource):
    def get(self):

        emotions=['NEGATIVE','NEUTRAL','POSITIVE']
        
        args = parser.parse_args()
        user_query = args['query']
        
        data=scaler.transform(np.array(json.loads(user_query)["input"]).reshape(-1,1).T)

        prediction = model.predict(data)
        
        emotion_ = str(emotions[prediction[0]])
        output = {"output": emotion_}
        return output
    
api.add_resource(PredictEmotion, '/')

if __name__ == '__main__':
    app.run(debug=True)