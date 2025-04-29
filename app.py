from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# 確保上傳資料夾存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='沒有檔案部分')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='未選擇檔案')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # 這裡暫時用假資料，之後可接模型推論
            result = {'fire_detected': True, 'confidence': 0.95}
            return render_template('index.html', filename=filename, result=result)
        else:
            return render_template('index.html', error='檔案格式不支援')
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
