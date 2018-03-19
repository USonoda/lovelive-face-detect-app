from flask import Flask, render_template, request, redirect, url_for
from werkzeug import secure_filename
import os
from for_detect import detect
from PIL import Image
import tmp


app = Flask(__name__)
app.config['DEBUG'] = True

# 投稿画像の保存先
UPLOAD_FOLDER = 'tmp'
# ./tmpも静的ディレクトリにしたい
app.register_blueprint(tmp.app)


# ルーティング /にアクセス時
@app.route('/')
def index():
    return render_template('index.html')


# 画像投稿時のアクション
@app.route('/post', methods=['GET', 'POST'])
def post():
    if request.method == 'POST':
        if not request.files['file'].filename == u'':
            # 適宜フォルダを削除
            if len(os.listdir(UPLOAD_FOLDER)) > 6:
                for i in range(4):
                    os.remove(os.path.join(UPLOAD_FOLDER, os.listdir(UPLOAD_FOLDER)[i]))

            f = request.files['file']
            img_path = os.path.join(UPLOAD_FOLDER, secure_filename(f.filename))
            f.save(img_path)

            f, face_num = detect.detect_face(img_path)
            f.save(img_path)

            result = ['', img_path, face_num]  # 最初の要素は（後々の）上位3人の表示用
        else:
            result = []
        return render_template('index.html', result=result)
    else:
        # エラーなどでリダイレクトしたい場合
        return redirect(url_for('index'))


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0')
