import os
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Flatten, Dense
from keras.preprocessing import image
from keras import optimizers
import cv2
from PIL import Image
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


classes = ['honoka', 'kotori', 'umi', 'hanayo', 'rin', 'maki', 'nico', 'eli', 'nozomi', 'others']
nb_classes = len(classes)
img_width, img_height = 150, 150


# 色指定
color = {
    'honoka':(255,165,20),
    'kotori':(245,245,245),
    'umi':(65,105,255),
    'hanayo':(0,160,0),
    'rin':(255,230,0),
    'maki':(255,0,0),
    'nico':(255,105,180),
    'eli':(0,255,255),
    'nozomi':(190,0,255),
    'others':(0,0,0)
}


def model_load(result_dir):
    # VGG16, FC層は不要なので include_top=False
    input_tensor = Input(shape=(img_width, img_height, 3))
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

    # FC層の作成
    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(nb_classes, activation='softmax'))

    # VGG16とFC層を結合してモデルを作成
    model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))

    # 学習済みの重みをロード
    model.load_weights(os.path.join(result_dir, 'finetuning.h5'))

    # 多クラス分類を指定
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy'])

    return model


def evaluation(img_path):  # 上位3人の結果を表示する用　→　顔の数が1つか0だった時用
    # モデルのロード
    model = model_load('./for_detect/results_all')

    filename = img_path
    img = image.load_img(filename, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # 学習時に正規化してるので、ここでも正規化
    x = x / 255
    pred = model.predict(x)[0]

    top = 3  # 上位3つを表示
    top_indices = pred.argsort()[-top:][::-1]
    result = [(classes[i], pred[i]) for i in top_indices]

    return result


def eval_array(face_num, x, top=3):  # xは画像の配列(RGB)　　画像ファイルに書き込む用
    # if face_num == 0:
    #     model = model_load("results_all")
    #     # 学習時に正規化してるので、ここでも正規化
    #     x = x / 255
    #     pred = model.predict(x)[0]
    #
    #     top_indices = pred.argsort()[-top:][::-1]
    #     result = [(classes[i], pred[i]) for i in top_indices]
    # else:

    model = model_load('./for_detect/results_150pt')
    x = x / 255
    pred = model.predict(x)[0]

    top_indices = pred.argsort()[-top:][::-1]
    result = [(classes[i], pred[i]) for i in top_indices]
    return result


def detect_face(image):  # 引数はPIL Image、出力は結果書き込んだPIL image
    image = cv2.imread(image)  # この時点でimageは配列(GBR)になる

    # 顔抽出

    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier("./for_detect/lbpcascade_animeface.xml")
    # 顔認識結果の座標情報をface_listに入れる
    face_list = cascade.detectMultiScale(image_gs,
                                         scaleFactor=1.1,
                                         minNeighbors=2,    # いじらない
                                         minSize=(int(img_width/2), int(img_height/2)))
    # cv2とkerasでRGB違うので顔判定前に変える
    b, g, r = cv2.split(image)
    image = cv2.merge([r, g, b])

    face_num = len(face_list)
    # 顔が１つ以上検出された時
    if face_num > 0:
        for rect in face_list:  # rectは［座標、座標、幅、高さ］
            x, y, width, height = rect
            face = image[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
            face = cv2.resize(face, (img_width, img_height))
            face = np.expand_dims(face, axis=0)  # faceは画像の配列
            # まず判定
            det_result = eval_array(face_num, face)
            # 枠を書く
            cv2.rectangle(image,
                          tuple(rect[0:2]),  # 左上の座標
                          tuple(rect[0:2] + rect[2:4]),  # 幅高さ
                          color[det_result[0][0]],
                          thickness=3)   # 枠線の太さ

            for i in range(3):  # 確率高い人3人まで名前を書く
                cor_num = round(100 * det_result[i][1])

                if cor_num > 3:  # 3%以上のものを表示
                    name = det_result[i][0]
                    cor_per = str(cor_num) + '%'
                    # 文字を書く
                    cv2.putText(image, name + ':' + cor_per,
                                (x, y + height + 15 + 20 * i),
                                cv2.FONT_HERSHEY_DUPLEX,
                                0.6, (255, 255, 255), 2,  # 文字サイズ、色、太さ
                                lineType=cv2.LINE_AA)
                    cv2.putText(image, name + ':' + cor_per,
                                (x, y + height + 15 + 20 * i),
                                cv2.FONT_HERSHEY_DUPLEX,
                                0.6, (50, 50, 50), 1,  # 文字サイズ、色、太さ
                                lineType=cv2.LINE_AA)
    # 顔が検出されなかった時
    else:
        print("no face")

    image = Image.fromarray(image)  # 画像に変換
    # return [image]
    return [image, face_num]  # 顔の数によって挙動を変える



if __name__ == '__main__':
    # このディレクトリにテストしたい画像を格納しておく
    test_data_dir = '../../fine_tuning/dataset/test'

    # テスト用画像取得
    test_imagelist = os.listdir(test_data_dir)

    for test_image in test_imagelist:
        if test_image == '.DS_Store':
            continue
        # image_path = os.path.join(test_data_dir,test_image)
        result_image = detect_face(test_data_dir+'/'+test_image)[0]
        result_image.show()
