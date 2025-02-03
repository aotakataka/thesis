import json
import os
import copy
import sys
import math
import time
import japanize_matplotlib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button
import warnings
import faceorder

def load_fold_file(file_path):
    with open(file_path, 'r') as file:
        fold_data = json.load(file)
    return fold_data

def extract_number(filename):
    # ファイル名から数値を抽出
    number = int(''.join(filter(str.isdigit, filename)))
    # 並べ替え基準としてタプルを返す (数値, reverseの有無)
    return number

def extract_number_and_isreverse(filename):
    # ファイル名から数値を抽出
    number = int(''.join(filter(str.isdigit, filename)))
    # reverseが含まれるかどうかで優先度を決める
    is_reverse = 'reverse' in filename
    # 並べ替え基準としてタプルを返す (数値, reverseの有無)
    return (number, is_reverse)

def extract_number_and_reverse_for_image_title(filename):
    # ファイル名から数値を抽出
    number = int(''.join(filter(str.isdigit, filename)))
    # reverseが含まれるかどうかで優先度を決める
    is_reverse = 'reverse' in filename
    # 並べ替え基準としてタプルを返す (数値, reverseの有無)
    if is_reverse:
        return str(number) + "(裏)"
    else:
        return str(number) + "(表)"

def crop_non_white_area(image_path, output_path, margin=5):
    # 画像を開く
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    pixels = img.load()

    # 白以外のピクセルが含まれる列と行の範囲を調べる
    non_white_columns = []
    non_white_rows = []

    # 列ごとの走査
    for x in range(width):
        for y in range(height):
            r, g, b = pixels[x, y]
            if (r, g, b) != (255, 255, 255):
                non_white_columns.append(x)
                break

    # 行ごとの走査
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            if (r, g, b) != (255, 255, 255):
                non_white_rows.append(y)
                break

    # 範囲を計算
    if non_white_columns and non_white_rows:
        col_start = max(min(non_white_columns) - margin, 0)  # 左端の余白
        col_end = min(max(non_white_columns) + margin, width)  # 右端の余白
        row_start = max(min(non_white_rows) - margin, 0)  # 上端の余白
        row_end = min(max(non_white_rows) + margin, height)  # 下端の余白
        # 画像を切り取り
        cropped_img = img.crop((col_start, row_start, col_end, row_end))
        cropped_img.save(output_path)  # 結果を保存

def plot_2d(vertices, faces, face_order, faces_index, imnum):
    print(imnum[0], face_order)
    # 保存するフォルダの作成
    output_folder = 'fold_input'
    _, ax = plt.subplots()
    # 頂点のリストをタプルに変換
    vertex_list = [tuple(vertex) for vertex in vertices]
    # 面に含まれる頂点のみプロットするためのインデックスセットを作成
    displayfaces = []
    for i in range(len(face_order)):
        for j in range(len(faces_index)):
            if face_order[i] == faces_index[j]:
                displayfaces.append(faces[j])
    if len(faces) != len(displayfaces):
        print("error: len(faces) != len(displayfaces)")
        sys.exit()
    # 各面の頂点をプロット
    for face in displayfaces[::-1]:
        polygon = [vertex_list[idx] for idx in face]
        poly = Polygon(polygon, facecolor='#B4E5A2', edgecolor='black', alpha=0.5)
        ax.add_patch(poly)
    # 軸のスケール調整
    vertices = np.array(vertices)
    ax.set_xlim([min(vertices[:, 0].min()*1.2, -300), max(vertices[:, 0].max()*1.2, 300)])
    ax.set_ylim([min(vertices[:, 1].min()*1.2, -300), max(vertices[:, 1].max()*1.2, 300)])
    # アスペクト比を保持
    ax.set_aspect('equal')
    ax.axis('off')
    plt.savefig(f'{output_folder}/model_{imnum[0]}.png')
    crop_non_white_area(f'{output_folder}/model_{imnum[0]}.png', f'{output_folder}/model_{imnum[0]}.png')
    plt.close()
    # 各裏面の頂点をプロット 裏側は色変える
    _, ax = plt.subplots()
    vertex_list = [(-vertex[1], -vertex[0]) for vertex in vertices]
    for face in displayfaces:
        polygon = [vertex_list[idx] for idx in face]
        poly = Polygon(polygon, facecolor='lightblue', edgecolor='black', alpha=0.5)
        ax.add_patch(poly)
    # 軸のスケール調整
    vertices = np.array([[-vertice[1],-vertice[0]] for vertice in vertices])
    ax.set_xlim([min(vertices[:, 0].min()*1.2, -300), max(vertices[:, 0].max()*1.2, 300)])
    ax.set_ylim([min(vertices[:, 1].min()*1.2, -300), max(vertices[:, 1].max()*1.2, 300)])
    # アスペクト比を保持
    ax.set_aspect('equal')
    ax.axis('off')
    plt.savefig(f'{output_folder}/model_{imnum[0]}_reverse.png')
    crop_non_white_area(f'{output_folder}/model_{imnum[0]}_reverse.png', f'{output_folder}/model_{imnum[0]}_reverse.png')
    plt.close()

def display_and_save_images(image_files, output_folder):
    for batch_index in range(0, len(image_files), 24):
        batch_files = image_files[batch_index:batch_index + 24]
        cols = math.ceil(math.sqrt(len(batch_files)))
        if cols %2 != 0:
            cols += 1
        rows = math.ceil(len(batch_files) / cols)
        _, axs = plt.subplots(rows, cols, figsize=(15, 8))
        axs = axs.flatten()  # 2次元配列を1次元に変換

        # 生成画像を並べる
        for j, image_file in enumerate(batch_files):
            img_path = os.path.join(output_folder, image_file)
            img = Image.open(img_path)
            axs[j].imshow(img)  # アスペクト比を自動調整
            axs[j].axis('off')  # 軸を非表示に
            axs[j].set_title(extract_number_and_reverse_for_image_title(image_file), fontsize=30)

        for col in range(1, cols, 2):  # 列インデックスを2つおきに取得
            if col == 1:  # 最初の列（左端）には縦線を引かない
                continue
            for row in range(rows):
                index = row * cols + col - 1  # 現在のセルインデックス
                if index < len(axs) and axs[index].get_images():  # 画像が存在する場合のみ縦線を追加
                    axs[index].axvline(x=1, color='black', linewidth=2, linestyle='--')

        # 残りのサブプロットを非表示にする
        for k in range(len(batch_files), len(axs)):
            axs[k].axis('off')

        plt.tight_layout()
        plt.savefig(f'{output_folder}/fold_input_{batch_index//24 + 1}.png')
        plt.close()

def display_howtofold(output_folder, face_order, start_time):
    image_subfiles = [f for f in os.listdir(output_folder) if f.endswith('.png')]
    image_subfiles = sorted(image_subfiles, key=extract_number_and_isreverse)
    display_and_save_images(image_subfiles, output_folder)
    image_files = [f for f in os.listdir(output_folder) if f.startswith('fold_input') and f.endswith('.png')]
    image_files = sorted(image_files, key=extract_number)
    current_image_idx = 0

    def update_image():
        new_img_path = os.path.join(output_folder, image_files[current_image_idx])
        new_img = Image.open(new_img_path)
        ax_img.imshow(new_img)
        ax_img.axis('off')
        fig.canvas.draw_idle()

    def on_button_click_next(event):
        nonlocal current_image_idx
        current_image_idx = (current_image_idx + 1) % len(image_files)  # 画像リストの先頭に戻れる
        update_image()

    def on_button_click_back(event):
        nonlocal current_image_idx
        current_image_idx = (current_image_idx - 1) % len(image_files)  # 画像リストの先頭に戻れる
        update_image()

    if len(image_files) >= 2:
        fig = plt.figure(figsize=(15,8))
        gs = GridSpec(2, 1, height_ratios=[7, 1])
        ax_img = fig.add_subplot(gs[0])
        img_path = os.path.join(output_folder, image_files[0])
        img = Image.open(img_path)
        ax_img.imshow(img)
        ax_img.axis('off')

        ax_button1 = fig.add_axes([0.3, 0.05, 0.1, 0.075])
        button1 = Button(ax_button1, '前へ')
        button1.on_clicked(on_button_click_back)

        ax_button2 = fig.add_axes([0.6, 0.05, 0.1, 0.075])
        button2 = Button(ax_button2, '次へ')
        button2.on_clicked(on_button_click_next)
    else:
        fig, ax = plt.subplots(figsize=(15,8))
        img_path = os.path.join(output_folder, image_files[0])
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis('off')
    warnings.filterwarnings("ignore", category=UserWarning, message="This figure includes Axes that are not compatible with tight_layout")
    plt.tight_layout()
    plt.savefig(f'{output_folder}/fold_input.png')
    plt.show(block=False)  # ウィンドウをブロックせずに表示する
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"computation time: {elapsed_time:.2f}second")

    # ユーザーからモデル番号の入力を受け付ける
    num_images = len(image_subfiles) // 2
    while True:
        try:
            selected_model = int(input(f"model number: 1 ~ {num_images} : "))
            if 1 <= selected_model <= num_images:
                break
            else:
                print(f"Please select from 1 ~ {num_images}")
        except ValueError:
            print("Please input number.")

    print(f"output of model number: {selected_model}")
    print(selected_model, face_order[selected_model-1])
    plt.close()  # ユーザーの入力後にウィンドウを閉じる
    return selected_model

def decide_model(fold_file_path):
    start_time = time.time()
    fold_data = load_fold_file(fold_file_path)
    output_folder = 'fold_input'
    if os.path.exists(output_folder):
        # フォルダ内のすべてのファイルを削除
        for filename in os.listdir(output_folder):
            file_path = os.path.join(output_folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        # フォルダが存在しない場合は作成
        os.makedirs(output_folder, exist_ok=True)
    # 折り目のデータを取得
    vertices_coords = copy.deepcopy(np.array(fold_data['vertices_coords']))
    faces_vertices = copy.deepcopy(fold_data['faces_vertices'])
    faces_index = [i for i in range(len(faces_vertices))]
    if 'file_frames' in fold_data:
        fileframes = copy.deepcopy(fold_data['file_frames'])
    else:
        fileframes = [{'faceOrders': copy.deepcopy(fold_data['faceOrders'])}]
    face_order = []
    for i in range(len(fileframes)):
        faceorders = copy.deepcopy(fileframes[i]['faceOrders'])
        fold_data = faceorder.create_fold_data(copy.deepcopy(vertices_coords.tolist()), copy.deepcopy(faces_vertices), copy.deepcopy(faceorders))
        face_order.append(faceorder.make_faceorder(fold_data))
    num = 1
    print("model 1 ~", len(face_order))
    for i in range(len(face_order)):
        plot_2d(copy.deepcopy(vertices_coords), copy.deepcopy(faces_vertices), copy.deepcopy(face_order[i]), copy.deepcopy(faces_index), [num])
        num += 1
    if len(face_order) != 1:
        selected_model = display_howtofold(output_folder, face_order, start_time)
        return face_order[selected_model-1]
    else:
        return face_order[0]
