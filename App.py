from flask import Flask, render_template, flash, request, session

import cv2

app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'


@app.route("/")
def homepage():
    return render_template('index.html')


@app.route("/Prediction")
def Prediction():
    return render_template('Prediction.html')


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':

        file = request.files['file']
        file.save('static/Out/Test.jpg')

        import warnings
        warnings.filterwarnings('ignore')

        import tensorflow as tf
        classifierLoad = tf.keras.models.load_model('model.h5')

        import numpy as np
        from keras.preprocessing import image

        test_image = image.load_img('static/Out/Test.jpg', target_size=(200, 200))
        # img1 = cv2.imread('static/Out/Test.jpg')
        # test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = classifierLoad.predict(test_image)
        print(result)
        pre = ''

        if result[0][0] == 1:
            pre = "good "
        elif result[0][1] == 1:
            pre = "poor"
            import winsound
            filename = 'alert.wav'
            winsound.PlaySound(filename, winsound.SND_FILENAME)
            sendmsg('9344809327', 'poor')
        elif result[0][2] == 1:
            pre = "satisfactory "
        elif result[0][3] == 1:
            pre = "very_poor"
            import winsound
            filename = 'alert.wav'
            winsound.PlaySound(filename, winsound.SND_FILENAME)
            sendmsg('9344809327','very_poor')

        return render_template('Prediction.html', pre=pre)


def sendmsg(targetno,message):
    import requests
    requests.post(
        "http://sms.creativepoint.in/api/push.json?apikey=6555c521622c1&route=transsms&sender=FSSMSS&mobileno=" + targetno + "&text=Dear customer your msg is " + message + "  Sent By FSMSG FSSMSS")


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
