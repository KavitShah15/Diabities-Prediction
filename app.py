import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('diabities.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    int_features[0] = (int_features[0] - 3.8450520833333335)/ 3.3695780626988623
    int_features[1] = (int_features[1] - 121.68676277850587)/ 30.43594886720766
    int_features[2] = (int_features[2] - 72.40518417462486)/ 12.096346184037948
    int_features[3] = (int_features[3] - 29.108072916666668)/ 8.791221023089737
    int_features[4] = (int_features[4] - 140.671875)/ 86.383059693181
    int_features[5] = (int_features[5] - 32.4552083333333)/ 6.875176818080996 
    int_features[6] = (int_features[6] - 0.4718763020833327)/ 0.33132859501277484
    int_features[7] = (int_features[7] - 33.240885416666664)/ 11.76023154067868
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    print(prediction)
    if prediction[0] == 1:
        stri = 'Take Care, you are suffering from Diabities'
    else:
        stri = 'Congratulation, you are not suffering from Diabities'
    
    return render_template('index.html', prediction_text=stri)

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)