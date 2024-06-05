from flask import *
import os
from werkzeug.utils import secure_filename
import label_image

import image_fuzzy_clustering as fem
import os
import secrets
from PIL import Image
from flask import url_for, current_app


def load_image(image):
    text = label_image.main(image)
    max_label = max(text, key=text.get)
    max_confidence = text[max_label]
    return max_label, max_confidence


def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    
    return image




app = Flask(__name__)
model = None

UPLOAD_FOLDER = os.path.join(app.root_path ,'static','img')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
@app.route('/first')
def first():
    return render_template('first.html')

 
  
    
@app.route('/login')
def login():
    return render_template('login.html')
@app.route('/chart')
def chart():
    return render_template('chart.html')

@app.route('/upload')
def upload():
    return render_template('index1.html')

@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':
        i=request.form.get('cluster')
        f = request.files['file']
        fname, f_ext = os.path.splitext(f.filename)
        original_pic_path=save_img(f, f.filename)
        destname = 'em_img.jpg'
        fem.plot_cluster_img(original_pic_path,i)
    return render_template('success.html')

def save_img(img, filename):
    picture_path = os.path.join(current_app.root_path, 'static/images', filename)
    # output_size = (300, 300)
    i = Image.open(img)
    # i.thumbnail(output_size)
    i.save(picture_path)

    return picture_path



@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload1():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        file_path = secure_filename(f.filename)
        f.save(file_path)
        # Make prediction
        result, confidence = load_image(file_path)
        result = result.title()

        if confidence < 0.98:
            result = "No deficiency"
        else:
            d = {
                "Vitamin A": " → Deficiency of vitamin A is associated with significant morbidity and mortality from common childhood infections, and is the world's leading preventable cause of childhood blindness. Vitamin A deficiency also contributes to maternal mortality and other poor outcomes of pregnancy and lactation.",
                'Vitamin B': " → Vitamin B12 deficiency may lead to a reduction in healthy red blood cells (anaemia). The nervous system may also be affected. Diet or certain medical conditions may be the cause. Symptoms are rare but can include fatigue, breathlessness, numbness, poor balance and memory trouble. Treatment includes dietary changes, B12 shots or supplements.",
                'Vitamin C': " → A condition caused by a severe lack of vitamin C in the diet. Vitamin C is found in citrus fruits and vegetables. Scurvy results from a deficiency of vitamin C in the diet. Symptoms may not occur for a few months after a person's dietary intake of vitamin C drops too low. Bruising, bleeding gums, weakness, fatigue and rash are among scurvy symptoms. Treatment involves taking vitamin C supplements and eating citrus fruits, potatoes, broccoli and strawberries.",
                'Vitamin D': " → Vitamin D deficiency can lead to a loss of bone density, which can contribute to osteoporosis and fractures (broken bones). Severe vitamin D deficiency can also lead to other diseases. In children, it can cause rickets. Rickets is a rare disease that causes the bones to become soft and bend.",
                "Vitamin E": " → Vitamin E needs some fat for the digestive system to absorb it. Vitamin E deficiency can cause nerve and muscle damage that results in loss of feeling in the arms and legs, loss of body movement control, muscle weakness, and vision problems. Another sign of deficiency is a weakened immune system."
            }
            conf = "[Probability: " + str(confidence) + "]"
            result = result + d[result] + conf

        os.remove(file_path)
        return result
    return None

if __name__ == '__main__':
    app.run()