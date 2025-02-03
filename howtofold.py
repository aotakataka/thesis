import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import japanize_matplotlib
from collections import defaultdict
from itertools import groupby
from PIL import Image
import copy
import json
import sys
import os
import math
import itertools
import time
import faceorder
import fold_input
import subprocess

# foldファイルを読み込む
def load_fold_file(file_path):
    with open(file_path, 'r') as file:
        fold_data = json.load(file)
    return fold_data

# 画像を回転する角度を決める
def rotate_image(points):
    # 対角線のベクトルを計算
    vector = points[2] - points[0]  # 対角線のベクトル
    angle = np.arctan2(vector[1], vector[0])  # 対角線ベクトルと水平線の角度
    rotation_angle = np.degrees(angle)  # 度に変換
    # 水平に整列するための角度を計算
    align_angle = 45 - rotation_angle if rotation_angle > 0 else -45 - rotation_angle
    if round(align_angle)%90 == 0:
        return 0
    if round(rotation_angle)%90 == 0:
        align_angle = abs(align_angle)
    return round(align_angle)

# 画像を保存する際に必要な部分だけ切り取って保存する
def crop_non_white_area(image_path, output_path, margin=15):
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

# 2Dモデルを表示して画像を保存する
def plot_2d(vertices, faces, face_order, faces_index, imnum, fold_kind, folding_edge):
    output_folder = 'howtofold'
    _, ax = plt.subplots()
    # 頂点のリストをタプルに変換
    vertex_list = [tuple(vertex) for vertex in vertices]
    # facesを平坦化
    faces = list(itertools.chain.from_iterable(copy.deepcopy(faces)))
    faces_index = list(itertools.chain.from_iterable(copy.deepcopy(faces_index)))
    # 面に含まれる頂点のみプロットするためのインデックスセットを作成
    used_vertex_indices = set([idx for face in faces for idx in face])
    if len(faces) != len(faces_index):
        print("error: len(faces) != len(faces_index)", len(faces), len(faces_index))
        sys.exit()
    print("faces", faces)
    print("faces_index", faces_index)
    print("face_order", face_order)
    displayfaces = []
    for i in range(len(face_order)):
        for j in range(len(faces_index)):
            if face_order[i] == faces_index[j]:
                displayfaces.append(faces[j])
    print("displayfaces", displayfaces)
    if len(faces) != len(displayfaces):
        print("error: len(faces) != len(displayfaces)", len(faces) , len(displayfaces))
        sys.exit()
    # 各面の頂点をプロット
    for face in displayfaces[::-1]:
        polygon = [vertex_list[idx] for idx in face]
        poly = Polygon(polygon, facecolor='#B4E5A2', edgecolor='black', alpha=0.8)
        ax.add_patch(poly)
    for edge in folding_edge:
        [[x1, y1], [x2, y2]] = edge
        ax.plot([x1, x2], [y1, y2], 'r--', linewidth=1)
    # 頂点に番号を表示
    for idx in used_vertex_indices:
        x, y = vertices[idx]
        #ax.text(x-30, y, str(idx), fontsize=10, ha='center', va='center', color='black')
    # 軸のスケール調整
    vertices = np.array(vertices)
    ax.set_xlim([min(vertices[:, 0].min()*1.2, -300), max(vertices[:, 0].max()*1.2, 300)])
    ax.set_ylim([min(vertices[:, 1].min()*1.2, -300), max(vertices[:, 1].max()*1.2, 300)])
    ax.axis('off')
    # アスペクト比を保持
    ax.set_aspect('equal')
    plt.savefig(f'{output_folder}/{imnum[0]}{fold_kind}.png')
    plt.close()
    if len(faces) == 1 and len(faces[0]) == 4:
        if issquare(vertices, faces[0]):
            print(faces)
            points = []
            for i in range(len(faces[0])):
                points.append(vertices[faces[0][i]])
            angle = rotate_image(points)
            print("angle", angle)
            print("number of faces is 1, finish!")
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"computation time: {elapsed_time:.2f}second")
            display_howtofold(output_folder, angle)
            sys.exit()

# ファイル名から数値を抽出する
def extract_number(filename):
    return int(''.join(filter(str.isdigit, filename)))

# ファイル名から文字を抽出する
def extract_non_number(filename):
    # .pngを除去
    filename = filename.replace('.png', '')
    # 数字以外の文字を抽出
    return ''.join(filter(lambda x: not x.isdigit(), filename))

# 保存された画像を1ウィンドウで表示する
def display_howtofold(output_folder, angle):
    # 保存した画像を読み込む
    image_files = [f for f in os.listdir(output_folder) if f.endswith('.png')]
    # 画像が傾いていたら回転
    for image_file in image_files:
        image_path = os.path.join(output_folder, image_file)
        if angle != 0:
            image = Image.open(image_path)
            rotated_image = image.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor = "white")
            rotated_image.save(image_path)
        crop_non_white_area(image_path, image_path)
    image_files = sorted(image_files, key=extract_number, reverse=True)
    # 画像の数に基づいて行と列を計算
    num_images = len(image_files)
    cols = math.ceil(math.sqrt(num_images))  # 列数を正方形に近く設定
    rows = math.ceil(num_images / cols)      # 行数を列数に基づいて計算
    # ウィンドウサイズを動的に設定
    _, axs = plt.subplots(rows, cols, figsize=(15,8))
    axs = axs.flatten()  # 2次元配列を1次元に変換
    # 画像を表示
    for i, image_file in enumerate(image_files):
        img_path = os.path.join(output_folder, image_file)
        img = Image.open(img_path)
        axs[i].imshow(img)  # アスペクト比を自動調整
        axs[i].axis('off')  # 軸を非表示に
        axs[i].set_title(extract_non_number(image_file), fontsize=25, pad=0)
    # 残りのサブプロットを非表示にする
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')
    plt.tight_layout()
    plt.savefig(f'{output_folder}/fold.png')

# 点を辺に対して対称移動する
def reflect_point_over_line(point, line_point1, line_point2):
    # 直線の方向ベクトルを計算
    line_vec = np.array([line_point2[0] - line_point1[0], line_point2[1] - line_point1[1]])
    line_vec = line_vec / np.linalg.norm(line_vec)  # 単位ベクトルに正規化

    # 点から直線へのベクトル
    point_vec = np.array([point[0] - line_point1[0], point[1] - line_point1[1]])

    # 点の直線上への射影
    proj_length = np.dot(point_vec, line_vec)
    proj_point = np.array(line_point1) + proj_length * line_vec

    # 対称移動した点は、射影点を基準に反対側に移動
    reflected_point = 2 * proj_point - np.array(point)

    return reflected_point

# リストの中の重複要素を削除する
def remove_duplicates(li):
    seen = set()
    return [x for x in li if not (x in seen or seen.add(x))]

# 2つのリストの共通要素とそれ以外をそれぞれ出力する
def lists_common(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    # 共通部分
    common_elements = set1 & set2
    # 共通部分を除外して残りをそれぞれリストに戻す
    unique_to_list1 = set1 - common_elements
    unique_to_list2 = set2 - common_elements
    # 2つのリストを結合して1つのリストにする
    combined_list = list(unique_to_list1) + list(unique_to_list2)
    return combined_list, list(common_elements)

# 頂点を同じ座標を持つもの同士グループ分けする
def group_points(points):
    grouped = []
    tolerance=1E-10
    points_list = [tuple(point) for point in points]  # 座標データをタプルに変換
    for i, point in enumerate(points_list):
        found_group = False
        # 既存のグループに属するか確認
        for group in grouped:
            if any(abs(point[0] - points_list[member][0]) <= tolerance and abs(point[1] - points_list[member][1]) <= tolerance for member in group):
                group.append(i)  # インデックスを追加
                found_group = True
                break
        # どのグループにも属さない場合は新しいグループを作成
        if not found_group:
            grouped.append([i])  # 新しいグループをインデックスで作成
    return grouped

# 再度頂点を同じ座標を持つもの同士グループ分けする
def group_points_again(points, faces):
    used_point = []
    for i in range(len(faces)):
        for j in range(len(faces[i])):
            for k in range(len(faces[i][j])):
                if faces[i][j][k] not in used_point:
                    used_point.append(faces[i][j][k])
    grouped = []
    tolerance=1E-10
    points_list = [tuple(point) for point in points]  # 座標データをタプルに変換
    for i, point in enumerate(points_list):
        if i in used_point:
            found_group = False
            # 既存のグループに属するか確認
            for group in grouped:
                if any(abs(point[0] - points_list[member][0]) <= tolerance and abs(point[1] - points_list[member][1]) <= tolerance for member in group):
                    group.append(i)  # インデックスを追加
                    found_group = True
                    break
            # どのグループにも属さない場合は新しいグループを作成
            if not found_group:
                grouped.append([i])  # 新しいグループをインデックスで作成
    return grouped

# 辺を同じ座標を持つもの同士グループ分けする タプルで辺と一緒にtype保持
def group_edges(edges, vertices_groups, edges_assignment):
    group_edges = {}
    group_edges_assignment = {}

    for i, edge in enumerate(edges):
        if edges_assignment[i] == "M" or edges_assignment[i] == "V":
            start, end = edge
            for j in range(len(vertices_groups)):
                if start in vertices_groups[j]:
                    start_group = j
            for j in range(len(vertices_groups)):
                if end in vertices_groups[j]:
                    end_group = j

            # グループのペアをタプルにする (小さい方が先)
            group_pair = tuple(sorted((start_group, end_group)))

            # グループペアに基づいてエッジインデックスと対応するエッジの種類を追加
            if group_pair not in group_edges:
                group_edges[group_pair] = []
                group_edges_assignment[group_pair] = []

            group_edges[group_pair].append((edge,edges_assignment[i]))
    return list(group_edges.values())

# 再度辺を同じ座標を持つもの同士グループ分けする
def group_edges_again(edges, vertices_groups):
    group_edges = {}
    print("edges group create again")
    print(vertices_groups)
    for i in range(len(edges)):
        for _, edge in enumerate(edges[i]):
            start_group, end_group = len(vertices_groups), len(vertices_groups)
            if edge[1] == "M" or edge[1] == "V":
                start, end = edge[0]
                for j in range(len(vertices_groups)):
                    if start in vertices_groups[j]:
                        start_group = j
                for j in range(len(vertices_groups)):
                    if end in vertices_groups[j]:
                        end_group = j

                # グループのペアをタプルにする (小さい方が先)
                group_pair = tuple(sorted((start_group, end_group)))

                # グループペアに基づいてエッジインデックスと対応するエッジの種類を追加
                if group_pair not in group_edges:
                    group_edges[group_pair] = []

                group_edges[group_pair].append(edge)
    return list(group_edges.values())

# 辺を同じ座標を持つもの同士グループ分けする
def group_edges2(edges, vertices_groups):
    group_edges = {}
    for _, edge in enumerate(edges):
        start, end = edge
        for j in range(len(vertices_groups)):
            if start in vertices_groups[j]:
                start_group = j
        for j in range(len(vertices_groups)):
            if end in vertices_groups[j]:
                end_group = j
        # グループのペアをタプルにする (小さい方が先)
        group_pair = tuple(sorted((start_group, end_group)))

        # グループペアに基づいてエッジインデックスと対応するエッジの種類を追加
        if group_pair not in group_edges:
            group_edges[group_pair] = []
        group_edges[group_pair].append(edge)
    return list(group_edges.values())

# unique_edgeによって辺の順番をアレンジ
def group_edges_arrange(grouped_edges, unique_edges, vertices_coords, faces_vertices):
    edges_from_unique_edges, edges_from_unique_edges2, edges_index_for_append = [], [], []
    for i in range(len(unique_edges)):
        edges_from_unique_edges.append([])
    for i in range(len(unique_edges)):
        for j in range(len(faces_vertices)):
            if unique_edges[i][0][0] in faces_vertices[j] and unique_edges[i][0][1] in faces_vertices[j]:
                for k in range(len(grouped_edges)):
                    for m in range(len(grouped_edges[k])):
                        if grouped_edges[k][m][0][0] in faces_vertices[j] and grouped_edges[k][m][0][1] in faces_vertices[j]:
                            if grouped_edges[k][m] not in unique_edges:
                                edges_from_unique_edges[i].append(grouped_edges[k][m])
    for i in range(len(edges_from_unique_edges)):
        edges_from_unique_edges2.append([])
    for i in range(len(edges_from_unique_edges)):
        for j in range(len(edges_from_unique_edges[i])):
            for k in range(len(faces_vertices)):
                if edges_from_unique_edges[i][j][0][0] in faces_vertices[k] and edges_from_unique_edges[i][j][0][1] in faces_vertices[k]:
                    for m in range(len(grouped_edges)):
                        for n in range(len(grouped_edges[m])):
                            if grouped_edges[m][n][0][0] in faces_vertices[k] and grouped_edges[m][n][0][1] in faces_vertices[k]:
                                if grouped_edges[m][n] not in edges_from_unique_edges and grouped_edges[m][n] not in unique_edges and grouped_edges[m][n] not in edges_from_unique_edges[i] and grouped_edges[m][n] not in edges_from_unique_edges2[i]:
                                    edges_from_unique_edges2[i].append(grouped_edges[m][n])
    for i in range(len(edges_from_unique_edges2)):
        for j in range(len(edges_from_unique_edges2[i])):
            for k in range(len(unique_edges)):
                if complete_overlap_edges(unique_edges[k][0], edges_from_unique_edges2[i][j][0], vertices_coords):
                    edges_index_for_append.append(i)
    for i in range(len(unique_edges)):
        if i in edges_index_for_append:
            for j in range(len(grouped_edges)):
                if unique_edges[i] in grouped_edges[j]:
                    grouped_edges[j].remove(unique_edges[i])
            grouped_edges.insert(0, [unique_edges[i]])
    for m in edges_index_for_append:
        for i in range(len(edges_from_unique_edges[m])):
            for j in range(len(grouped_edges)):
                if edges_from_unique_edges[m][i] in grouped_edges[j]:
                    grouped_edges[j].remove(edges_from_unique_edges[m][i])
            grouped_edges.insert(0, [edges_from_unique_edges[m][i]])
    for m in edges_index_for_append:
        for i in range(len(edges_from_unique_edges2[m])):
            for j in range(len(grouped_edges)):
                if edges_from_unique_edges2[m][i] in grouped_edges[j]:
                    grouped_edges[j].remove(edges_from_unique_edges2[m][i])
            grouped_edges.insert(0, [edges_from_unique_edges2[m][i]])
    for i in range(len(unique_edges)):
        if i not in edges_index_for_append:
            for j in range(len(grouped_edges)):
                if unique_edges[i] in grouped_edges[j]:
                    grouped_edges[j].remove(unique_edges[i])
            grouped_edges.insert(0, [unique_edges[i]])
    for i in range(len(grouped_edges)):
        if [] in grouped_edges:
            grouped_edges.remove([])
    grouped_edges = remove_duplicates_list(grouped_edges)
    return grouped_edges

# 面を同じ座標を持つもの同士グループ分けする
def group_faces(faces, vertices_groups, faces_index, face_order):
    sort_face_order = copy.deepcopy(face_order)
    sorted_data = sorted(zip(sort_face_order, faces, faces_index), key=lambda x: x[0])
    _, sorted_faces, sorted_faces_index = map(list, zip(*sorted_data))
    group_faces = {}
    group_faces_index = {}  # faces_index を保持する辞書
    for face, index in zip(sorted_faces, sorted_faces_index):
        # 各面の頂点がどのグループに属するかを調べる
        face_groups = set()
        for vertex in face:
            for i, group in enumerate(vertices_groups):
                if vertex in group:
                    face_groups.add(i)
        # グループのペアをタプルにする (小さい方から順にソート)
        group_pair = tuple(sorted(face_groups))
        # グループペアに基づいて面を追加
        if group_pair not in group_faces:
            group_faces[group_pair] = []
            group_faces_index[group_pair] = []  # 新しいグループペアに対応する faces_index リストも作成
        group_faces[group_pair].append(face)
        group_faces_index[group_pair].append(index)  # faces_index を追加
    # グループごとに faces と faces_index をまとめて返す
    return list(group_faces.values()), list(group_faces_index.values())

# 再度面を同じ座標を持つもの同士グループ分けする
def group_faces_again(faces, vertices_groups, faces_index):
    group_faces = {}
    group_faces_index = {}  # faces_index を保持する辞書
    print("faces group create again")
    for i in range(len(faces)):
        for j in range(len(faces[i])):
            # 各面の頂点がどのグループに属するかを調べる
            face_groups = set()
            for vertex in faces[i][j]:
                for k, group in enumerate(vertices_groups):
                    if vertex in group:
                        face_groups.add(k)
            # グループのペアをタプルにする (小さい方から順にソート)
            group_pair = tuple(sorted(face_groups))
            # グループペアに基づいて面とそのインデックスを追加
            if group_pair not in group_faces:
                group_faces[group_pair] = []
                group_faces_index[group_pair] = []  # 新しいグループペアに対応する faces_index リストを作成
            group_faces[group_pair].append(faces[i][j])
            group_faces_index[group_pair].append(faces_index[i][j])  # インデックスを追加
    return list(group_faces.values()), list(group_faces_index.values())

# 2点の距離を計算する
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# 辺の傾きを計算する
def calculate_gradient(x1, y1, x2, y2):
    # x1 == x2 の場合、垂直な線のため傾きは定義されない
    if x1 == x2:
        return None
    return (y2 - y1) / (x2 - x1)

# 二つの辺の傾きの差が許容値以内かどうか確認する
def parallel_check(edge1, edge2):
    if distance(edge1[0], edge1[1]) < 1E-10 or distance(edge2[0], edge2[1]) < 1E-10:
        return False
    # 各辺の傾きを計算
    slope1 = calculate_gradient(edge1[0][0], edge1[0][1], edge1[1][0], edge1[1][1])
    slope2 = calculate_gradient(edge2[0][0], edge2[0][1], edge2[1][0], edge2[1][1])
    # 垂直な線の場合、両方が垂直なら平行とみなす
    if slope1 is None:
        if slope2 is None:
            return True
        elif abs(slope2) > 1E10:
            return True
        else:
            return False
    if slope2 is None:
        if abs(slope1) > 1E10:
            return True
        else:
            return False
    # 傾きが等しいかどうかを判定
    return (abs(slope1-slope2) < 1E-10) or (abs(slope1)> 1E10 and abs(slope2)> 1E10)

# 二つの面を合体させる
def merge_faces(face1, face2, edge):
    #合体される側は後ろ2つにedgeの端点が来るように
    for _ in range(len(face1)):
        # リストを回転させる
        face1 = face1[1::] + face1[:1]
        # num1, num2が先頭または最後尾でないか確認
        if face1[-2] in edge and face1[-1] in edge:
            break
    #合体する側はface1[-2]が先頭、face1[-1]が最後尾
    fl = 0
    for _ in range(len(face2)):
        # リストをiだけ回転させる
        face2 = face2[1::] + face2[:1]
        # num1, num2が先頭と最後尾にあるか確認
        if face2[0] == face1[-2] and face2[-1] == face1[-1]:
            fl = 1
            break
    if fl == 0:
        face2 = face2[::-1]
        for _ in range(len(face2)):
            # リストをiだけ回転させる
            face2 = face2[1::] + face2[:1]
            # num1, num2が先頭と最後尾にあるか確認
            if face2[0] == face1[-2] and face2[-1] == face1[-1]:
                break
    face1.pop(-1)
    face1.pop(-1)
    for i in range(len(face2)):
        face1.append(face2[i])
    return face1

# 面が正方形(菱形)かどうかの確認をする
def issquare(vertices, face):
    if len(face) != 4:
        return False
    # 頂点の順序: A, B, C, D
    A, B, C, D = vertices[face[0]], vertices[face[1]], vertices[face[2]], vertices[face[3]]
    tolerance = 1E-10
    # 辺の長さと対角線の長さを計算
    sides = [distance(A, B), distance(B, C), distance(C, D), distance(D, A)]
    # 辺の長さがすべて等しいかどうかを誤差の範囲内で判定
    side_length = sides[0]
    if not all(abs(side - side_length) <= tolerance for side in sides):
        return False
    return True

# 面が隣り合う四角形の対角線を辺としてもつか確認する
def has_diagonal_as_edge(vertices, face, square):
    if len(square) != 4:
        return False
    tolerance=1E-10
    # 四角形の対角線の頂点ペアを取得
    diag1 = (vertices[square[0]], vertices[square[2]])
    diag2 = (vertices[square[1]], vertices[square[3]])
    # 対角線の長さ
    diagonal_length = distance(diag1[0], diag1[1])
    # 面の各辺について、四角形の対角線のいずれかと一致するか確認
    for i in range(len(face)):
        p1 = vertices[face[i]]
        p2 = vertices[face[(i + 1) % len(face)]]  # 次の頂点（ループ）
        # 面の辺の長さを計算
        edge_length = distance(p1, p2)
        # 対角線の端点ペアと辺の頂点が一致し、長さがほぼ等しいかを判定
        is_diag1 = (distance(p1, diag1[0]) <= tolerance or distance(p1, diag1[1]) <= tolerance) and (distance(p2, diag1[0]) <= tolerance or distance(p2, diag1[1]) <= tolerance)
        is_diag2 = (distance(p1, diag2[0]) <= tolerance or distance(p1, diag2[1]) <= tolerance) and (distance(p2, diag2[0]) <= tolerance or distance(p2, diag2[1]) <= tolerance)
        if (is_diag1 or is_diag2) and abs(edge_length - diagonal_length) <= tolerance:
            return True
    return False

# 面が隣り合う三角形の垂線を辺としてもつか確認する
def has_perpendicular_as_edge(vertices, face, triangle):
    if len(triangle) != 3:
        return False
    tolerance=1E-10
    #垂線下ろす頂点を決定
    for i in range(3):
        tptriangle = copy.deepcopy(triangle)
        tptriangle.remove(triangle[i])
        # 垂線を下ろす点
        point = np.array(vertices[triangle[i]])
        # 対象の辺の両端の座標
        li = [0,1,2]
        li.remove(i)
        v1, v2 = [triangle[li[0]], triangle[li[1]]]
        p1 = np.array(vertices[v1])
        p2 = np.array(vertices[v2])
        # 辺をベクトルとして表現
        edge_vector = p2 - p1
        point_vector = point - p1
        # edge_vectorにpoint_vectorを射影
        edge_length_squared = np.dot(edge_vector, edge_vector)
        if edge_length_squared == 0:
            # 辺の長さがゼロの場合（異常ケース）
            continue
        t = np.dot(point_vector, edge_vector) / edge_length_squared
        t = max(0, min(1, t))  # 線分の範囲内に限定
        # 射影点（交点）
        intersection = p1 + t * edge_vector
        if abs(distance(vertices[tptriangle[0]], intersection) - distance(vertices[tptriangle[1]], intersection)) > tolerance:
            continue
        # 垂線の頂点ペアを取得
        per = (vertices[triangle[i]], intersection)
        # 対角線の長さ
        perpendicular_length = distance(per[0], per[1])
        # 面の各辺について、三角形の垂線のいずれかと一致するか確認
        for j in range(len(face)):
            p1 = vertices[face[j]]
            p2 = vertices[face[(j + 1) % len(face)]]  # 次の頂点（ループ）
            # 面の辺の長さを計算
            edge_length = distance(p1, p2)
            # 対角線の端点ペアと辺の頂点が一致し、長さがほぼ等しいかを判定
            is_per = (distance(p1, per[0]) <= tolerance or distance(p1, per[1]) <= tolerance) and (distance(p2, per[0]) <= tolerance or distance(p2, per[1]) <= tolerance)
            if is_per and abs(edge_length - perpendicular_length) <= tolerance:
                print([face[j], face[(j + 1) % len(face)]])
                # 交点をverticesに追加
                vertices = np.vstack([vertices, intersection])
                intersection_index = len(vertices) - 1
                print([triangle[i], intersection_index])
                print("add verices", intersection_index,":", intersection)
                return True, vertices, [triangle[i], int(intersection_index)]
    return False, vertices, []

# 面がタイプBの辺をもつ三角形かどうか確認する
def istriangle_typeb(edges, triangle):
    if len(triangle) != 3:
        return False
    for i in range(len(edges[-1])):
        if edges[-1][i][0][0] in triangle and edges[-1][i][0][1] in triangle:
            return True
    return False

# 三角形の頂点からタイプBの辺の垂線を下ろして交点、辺をvertices, edgesに追加する
def make_perpendicular_line(vertices, edges, vertice, edge):
    # 垂線を下ろす点
    point = np.array(vertices[vertice])
    # 対象の辺の両端の座標
    v1, v2 = edge
    p1 = np.array(vertices[v1])
    p2 = np.array(vertices[v2])
    # 辺をベクトルとして表現
    edge_vector = p2 - p1
    point_vector = point - p1
    # edge_vectorにpoint_vectorを射影
    edge_length_squared = np.dot(edge_vector, edge_vector)
    if edge_length_squared == 0:
        # 辺の長さがゼロの場合（異常ケース）
        return vertices, edges

    t = np.dot(point_vector, edge_vector) / edge_length_squared
    t = max(0, min(1, t))  # 線分の範囲内に限定
    # 射影点（交点）
    intersection = p1 + t * edge_vector

    # 交点をverticesに追加
    vertices = np.vstack([vertices, intersection])
    intersection_index = len(vertices) - 1
    print("add verices", intersection_index,":", intersection)
    # 新しいedgeをedgesに追加
    edges.append([([vertice, int(intersection_index)], 'M')])
    print("add edges", [([vertice, int(intersection_index)], 'M')])
    return vertices, edges

# 順番が同じで要素が不足している部分ルートを省略。
def remove_subroutes(routes):
    # 長さで降順にソート
    routes.sort(key=len, reverse=True)

    # 完全なルートを基準に部分ルートを削除
    filtered_routes = []
    for i, route in enumerate(routes):
        is_subroute = False
        for longer_route in filtered_routes:
            if len(route) < len(longer_route) and route == longer_route[:len(route)]:
                is_subroute = True
                break
        if not is_subroute:
            filtered_routes.append(route)
    return filtered_routes

# 各リスト内で複数の辺に共通する点を見つける。
def find_common_points(edges):
    #print(edges)
    common_points = []
    for edge_group in edges:
        points = [point for edge in edge_group for point in edge]
        common = [point for point in set(points) if points.count(point) > 1]
        common_points.extend(common)
    return list(set(common_points))

# 再帰的にルートを探索する
def find_routes(edges, start, start_vertices, visited, current_route, valid_routes, target_points, visited_edge):
    # 始点を2回以上訪れたら終了
    for i in range(len(start_vertices)):
        if visited[start_vertices[i]] > 1:
            return
    # 現在のルートに点を追加
    current_route.append(start)
    # 接続された他の点に移動
    for edge_group in edges:
        for edge in edge_group:
            #print(edge)
            if start in edge:
                next_point = edge[0] if edge[1] == start else edge[1]
                if edge not in visited_edge:  # 同じルート内で重複を避ける
                    visited_edge.append(edge)
                    #print("visited_edge", visited_edge)
                    visited[start] += 1
                    #print(start, visited[start])
                    find_routes(edges, next_point, start_vertices, visited, current_route, valid_routes, target_points, visited_edge)
                    visited[start] -= 1
                    visited_edge.remove(edge)

    # ルート中にAまたはBが含まれていれば保存
    if target_points & set(current_route):
        valid_routes.append(current_route[:])
    # 現在の点をルートから削除
    #print("--")
    current_route.pop()

#直線上にある別の直線を探す
def edges_on_line(point_a, point_b, edges_list, vertices, rmlist):
    edgesonline = []
    edgeslist = []
    edgeslist_withtype = []
    line_points = []
    tolerance = 1E-8
    x1, y1 = vertices[point_a]
    x2, y2 = vertices[point_b]
    a = y2 - y1
    b = x1 - x2
    c = -(a * x1 + b * y1)
    denominator = math.sqrt(a**2 + b**2)
    for i, vertex in enumerate(vertices):
        x0, y0 = vertex
        # 点と直線の距離を計算
        distance = abs(a * x0 + b * y0 + c) / denominator
        if distance <= tolerance:
            line_points.append(i)
    print("line_points", line_points)
    for i in range(len(edges_list)):
        for j in range(len(edges_list[i])):
            if edges_list[i][j][0][0] in line_points and edges_list[i][j][0][1] in line_points:
                edgeslist.append(edges_list[i][j][0])
                edgeslist_withtype.append(edges_list[i][j])
    vertices_grouped_list = copy.deepcopy(group_points(vertices))
    edgeslist = group_edges2(edgeslist, vertices_grouped_list)
    # 共通点を見つける
    common_points = find_common_points(edgeslist)
    print("edgeslist", edgeslist)
    print("折り返し点:", common_points)
    if common_points == []:
        print(edgeslist_withtype)
        return edgeslist_withtype, edgeslist

    # 全ルートを探索
    valid_routes = []
    target_points = {point_a, point_b}
    visited = defaultdict(int)
    visited_edge = []

    for common_point in common_points:
        find_routes(edgeslist, common_point, common_points, visited, [], valid_routes, target_points, visited_edge)
    rmli = []
    filtered_routes = remove_subroutes(valid_routes)
    for i in range(len(filtered_routes)):
        tmp = filtered_routes[i][0]
        for j in range(len(common_points)):
            if filtered_routes[i].count(common_points[j]) > 1 and common_points[j] != tmp:
                rmli.append(filtered_routes[i])

    dif = [item for item in filtered_routes if item not in rmli]
    ans = []
    for i in range(len(dif)):
        if point_a in dif[i] and point_b in dif[i]:
            ans.append(dif[i])
    ans = list(set([item for sublist in ans for item in sublist]))
    for i in range(len(edges_list)):
        for j in range(len(edges_list[i])):
            if edges_list[i][j][0][0] in ans and edges_list[i][j][0][1] in ans and edges_list[i][j] not in rmlist:
                edgesonline.append(edges_list[i][j])
    print(edgesonline)
    return edgesonline, edgeslist

#線分上にある頂点グループ数を数える
def count_points_group_on_line(point_a, point_b, vertices, vertices_group):
    line_points = []
    tolerance = 1E-8
    x1, y1 = vertices[point_a]
    x2, y2 = vertices[point_b]
    min_x, max_x = min(x1, x2), max(x1, x2)
    min_y, max_y = min(y1, y2), max(y1, y2)
    a = y2 - y1
    b = x1 - x2
    c = -(a * x1 + b * y1)
    denominator = math.sqrt(a**2 + b**2)
    for i, vertex in enumerate(vertices):
        x0, y0 = vertex
        # 点と直線の距離を計算
        distance = abs(a * x0 + b * y0 + c) / denominator
        if distance <= tolerance:
            # 線分の範囲内にあるかを判定
            if min_x - tolerance <= x0 <= max_x + tolerance and min_y - tolerance <= y0 <= max_y + tolerance:
                line_points.append(i)
    groups_containing_points = set()
    for i, group in enumerate(vertices_group):
        # 現在のグループに line_points の点が含まれているか確認
        if any(point in group for point in line_points):
            groups_containing_points.add(i)
    return len(groups_containing_points)

#線分上にある点の数を数える
def count_points_on_line(point_a, point_b, vertices):
    line_points = []
    tolerance = 1E-8
    x1, y1 = vertices[point_a]
    x2, y2 = vertices[point_b]
    min_x, max_x = min(x1, x2), max(x1, x2)
    min_y, max_y = min(y1, y2), max(y1, y2)
    a = y2 - y1
    b = x1 - x2
    c = -(a * x1 + b * y1)
    denominator = math.sqrt(a**2 + b**2)
    for i, vertex in enumerate(vertices):
        x0, y0 = vertex
        # 点と直線の距離を計算
        distance = abs(a * x0 + b * y0 + c) / denominator
        if distance <= tolerance:
            # 線分の範囲内にあるかを判定
            if min_x - tolerance <= x0 <= max_x + tolerance and min_y - tolerance <= y0 <= max_y + tolerance:
                line_points.append(i)
    return len(line_points)

# リストBに基づいてリストAを同じ順序にソート
def sort_two_lists(list_a, list_b):
    # リストをペアにする
    paired = list(zip(list_a, list_b))
    # リストBを基準にソート
    paired_sorted = sorted(paired, key=lambda x: x[1], reverse = True)
    # ペアを再び分割
    sorted_a, sorted_b = zip(*paired_sorted)
    # タプルではなくリストで返す
    return list(sorted_a), list(sorted_b)

#移動した先に別の頂点があるかどうか
def moved_vertice_same_point(moved, vertices):
    tolerance = 1E-10
    for i in range(len(vertices)):
        if abs(moved[0]- vertices[i][0]) < tolerance and abs(moved[1]- vertices[i][1]) < tolerance:
            return True
    return False

#ある折り辺について対称な形かどうか
def is_symmetric(edge, vertices, edges):
    tolerance = 1E-10
    tpedges = []
    for i in range(len(edges)):
        for j in range(len(edges[i])):
            tpedges.append(edges[i][j][0])
    # 辺を構成する頂点
    v1, v2 = vertices[edge[0]], vertices[edge[1]]
    def find_nearest_vertex_index(reflected_point, current_index):
        """反転した点に最も近い頂点のインデックスを探す"""
        for i, vertex in enumerate(vertices):
            # 反転点が近似的に一致する場合
            if np.allclose(reflected_point, vertex, atol=tolerance):
                if i != current_index or i in edge:
                    return i
        return None
    # 辺の対称性確認
    for ed in tpedges:
        for v in ed:
            ele = find_nearest_vertex_index(reflect_point_over_line(vertices[v], v1, v2), v)
            if ele == None:# 対称な辺がない場合
                return False
    return True

# 2つの座標a, bがtolerance内で一致するかを確認する
def is_close(a, b):
    tolerance=1E-8
    return math.isclose(a[0], b[0], abs_tol=tolerance) and math.isclose(a[1], b[1], abs_tol=tolerance)

# 三角形の頂点がすべて多角形の頂点リストに含まれているかを確認する
def is_triangle_vertices_in_polygon(triangle, polygon, vertices):
    triangle_vertices, polygon_vertices= [], []
    for i in range(len(triangle)):
        triangle_vertices.append(vertices[triangle[i]])
    for i in range(len(polygon)):
        polygon_vertices.append(vertices[polygon[i]])
    return all(any(is_close(vertex, poly_vertex) for poly_vertex in polygon_vertices) for vertex in triangle_vertices)

#多角形の面積を求める
def polygon_area(face, vertices):
    n = len(face)
    vers = []
    for i in range(n):
        vers.append(vertices[face[i]])
    area = 0.0
    for i in range(n):
        x1, y1 = vers[i]
        x2, y2 = vers[(i + 1) % n]
        area += (x1 * y2) - (x2 * y1)
    return abs(area) / 2.0

#一回しか登場しない面に含まれる辺を探す
def edges_of_unique_faces(faces, edges):
    faceli = []
    for i in range(len(edges)):
        for j in range(len(edges[i])):
            for k in range(len(faces)):
                if edges[i][j][0][0] in faces[k] and edges[i][j][0][1] in faces[k]:
                    faceli.append(faces[k])
    # 各面の出現回数をカウント
    face_count = {}
    for face in faceli:
        face_tuple = tuple(sorted(face))  # リストをタプルに変換してソートし、一意にする
        if face_tuple in face_count:
            face_count[face_tuple] += 1
        else:
            face_count[face_tuple] = 1
    # 出現回数が1の面を抽出
    unique_faces = [list(face) for face, count in face_count.items() if count == 1]
    edges_of_unique_faces = []
    for i in range(len(unique_faces)):
        for j in range(len(unique_faces[i])):
            for k in range(len(edges)):
                for l in range(len(edges[k])):
                    if unique_faces[i][j-1] in edges[k][l][0] and unique_faces[i][j] in edges[k][l][0]:
                        edges_of_unique_faces.append(edges[k][l])
    return edges_of_unique_faces

# リストの中の重複要素を削除する
def remove_duplicates_list(li):
    seen = set()
    unique_list = []
    for sublist in li:
        sublist_tuple = tuple((tuple(sorted(edge[0])), edge[1]) for edge in sublist)  # 内側のリストもタプルに変換
        if sublist_tuple not in seen:
            seen.add(sublist_tuple)
            unique_list.append(sublist)
    return unique_list

# 2つの辺が完全に重なっているかどうかを確認する
def complete_overlap_edges(edge1, edge2, vertices):
    tolerance = 1E-10
    if (distance(vertices[edge1[0]], vertices[edge2[0]]) < tolerance and distance(vertices[edge1[1]], vertices[edge2[1]]) < tolerance) or distance(vertices[edge1[0]], vertices[edge2[1]]) < tolerance and distance(vertices[edge1[1]], vertices[edge2[0]]) < tolerance:
        return True
    else:
        return False

# 折り返す関数
def unfold(vertices, faces, edges, faces_index, face_order, fold_edge_index, facemax, unique_edges):
    moved= []
    moved_vertices = []
    remove_edges = []
    fold_kind = "error"
    cntl = 0
    tsubushi_square = 0
    input_vertices = copy.deepcopy(vertices)
    input_faces = copy.deepcopy(faces)
    input_edges = copy.deepcopy(edges)
    input_faces_index = copy.deepcopy(faces_index)
    input_face_order = copy.deepcopy(face_order)
    if fold_edge_index >= len(edges):
        print("error: fold_edge_index >= len(edges)")
        return input_vertices, input_faces, input_edges, input_faces_index, input_face_order, 0, facemax, fold_kind, []
    fold_edges_list = copy.deepcopy(edges[fold_edge_index])
    # finish確認
    cnt = 0
    for i in range(len(faces)):
        for _ in range(len(faces[i])):
            cnt += 1
    if cnt == 1 and len(faces[0][0]) != 4:
            print("error: face is not square")
            return input_vertices, input_faces, input_edges, input_faces_index, input_face_order, 0, facemax, fold_kind, []

    #中割り折り・かぶせ折りの場合
    vers1 = []
    vers2 = []
    print("--nakawari/kabuse  check start--")
    for l in range(len(fold_edges_list)): #1つの折り辺
        fold_edges_list = copy.deepcopy(edges[fold_edge_index])
        unfold_faces=[]
        #入力された折り辺を含む2面を探す
        if len(fold_edges_list) <= l:
            print("-----nakawari/kabuse------- error: len(fold_edges_list) <= l")
            return input_vertices, input_faces, input_edges, input_faces_index, input_face_order, 0, facemax, fold_kind, []

        fla = 0
        for i in range(len(fold_edges_list)):
            if fold_edges_list[l] != fold_edges_list[i] and (fold_edges_list[l][0][0] in fold_edges_list[i][0] or fold_edges_list[l][0][1] in fold_edges_list[i][0]) and fold_edges_list[l][1] == fold_edges_list[i][1]:
                _, [tm] = lists_common(fold_edges_list[l][0], fold_edges_list[i][0])
                fla = 1
        if l == 0:
            for i in range(len(faces)):
                for j, face in enumerate(faces[i]):
                    if fold_edges_list[l][0][0] in face and fold_edges_list[l][0][1] in face:
                        unfold_faces.append([i,j])
                        ver = []
                        for k in range(len(face)):
                            ver.append(vertices[face[k]])
                        vers1.append(ver)
        else:
            #len(vers1)-1個の点の座標が一致している面を探す
            for i in range(len(faces)):
                for j, face in enumerate(faces[i]):
                    ct = 0
                    for k in range(len(faces[i][j])):
                        for m in range(len(vers1)):
                            for n in range(len(vers1[m])):
                                if distance(vertices[faces[i][j][k]], vers1[m][n]) < 1E-8:
                                    #　点の座標と、面の頂点index入れる　頂点同士比べて距離torlerance以内なものがあるか
                                    ct += 1
                    if ct >= len(vers1) - 1 and fold_edges_list[l][0][0] in face and fold_edges_list[l][0][1] in face and [i,j] not in unfold_faces:
                        unfold_faces.append([i,j])
                        break
        if len(unfold_faces) != 2:
            print("-----nakawari/kabuse------- error: len(unfold_faces) != 2", unfold_faces)
            continue
        print("fla:", fla, faces[unfold_faces[0][0]][unfold_faces[0][1]], faces[unfold_faces[1][0]][unfold_faces[1][1]])

        if fla == 1: # 別の辺でも中割り折り・かぶせ折りにできるように対応
            cntl += 1
            tpedges = []
            if ((len(faces[unfold_faces[0][0]][unfold_faces[0][1]]) == 3 and len(faces[unfold_faces[1][0]][unfold_faces[1][1]]) >= 4)
                 or (len(faces[unfold_faces[0][0]][unfold_faces[0][1]]) >= 4 and len(faces[unfold_faces[1][0]][unfold_faces[1][1]]) == 3)):
                if len(faces[unfold_faces[0][0]][unfold_faces[0][1]]) == 3:
                    facetri = faces[unfold_faces[0][0]][unfold_faces[0][1]]
                else:
                    facetri = faces[unfold_faces[1][0]][unfold_faces[1][1]]
                for i in range(len(edges)):
                    for j in range(len(edges[i])):
                        if tm in edges[i][j][0] and edges[i][j] not in fold_edges_list and edges[i][j][0][0] in facetri and edges[i][j][0][1] in facetri:
                            tpedges = edges[i][j]
                print("tpedges", tpedges)
                if tpedges == []:
                    print("error: tpedges == []")
                    continue
                fold_edges_list[l] = tpedges
                unfold_faces=[]
                if cntl == 1:
                    for i in range(len(faces)):
                        for j, face in enumerate(faces[i]):
                            if fold_edges_list[l][0][0] in face and fold_edges_list[l][0][1] in face:
                                unfold_faces.append([i,j])
                                ver = []
                                for k in range(len(face)):
                                    ver.append(vertices[face[k]])
                                vers2.append(ver)
                    print(vers2)
                else:
                    #len(2)-1個の点の座標が一致している面を探す
                    for i in range(len(faces)):
                        for j, face in enumerate(faces[i]):
                            ct = 0
                            for k in range(len(faces[i][j])):
                                for m in range(len(vers2)):
                                    for n in range(len(vers2[m])):
                                        if distance(vertices[faces[i][j][k]], vers2[m][n]) < 1E-8:
                                            #　点の座標と、面の頂点index入れる　頂点同士比べて距離torlerance以内なものがあるか
                                            ct += 1
                            if ct >= len(vers2) - 1 and fold_edges_list[l][0][0] in face and fold_edges_list[l][0][1] in face and [i,j] not in unfold_faces:
                                unfold_faces.append([i,j])
                                break
                print("edge:", fold_edges_list[l])
                if len(unfold_faces) != 2:
                    print("error: len(unfold_faces) != 2")
                    continue
                print("fla == 1", faces[unfold_faces[0][0]][unfold_faces[0][1]], faces[unfold_faces[1][0]][unfold_faces[1][1]])

        #中割り折り・かぶせ折り　辺の同グループの中でも一本だけだから全辺試さないといけない
        # 折り込まれてる同じグループの三角形がunfold_facesの必要がある
        if unfold_faces[0][0] == unfold_faces[1][0] and len(faces[unfold_faces[0][0]][unfold_faces[0][1]]) == 3 and len(faces[unfold_faces[1][0]][unfold_faces[1][1]]) == 3:
            #折り辺表示用
            folding_edge, forvertp = [], []
            for i in range(len(faces[unfold_faces[0][0]][unfold_faces[0][1]])):
                forvertp.append([faces[unfold_faces[0][0]][unfold_faces[0][1]][i-1], faces[unfold_faces[0][0]][unfold_faces[0][1]][i]])
            for i in range(len(forvertp)):
                vertp = []
                for j in range(len(forvertp[i])):
                    vertp.append(vertices[forvertp[i][j]])
                folding_edge.append(vertp)
            nextfaces = []
            nextfaces_order = []
            #nextfacesはfacesの隣の面
            #nextfaces_orderはfacesとnextfacesの隣り合う対応関係
            for i in range(len(faces)):
                if i != unfold_faces[0][0]:
                    for j in range(len(faces[i])):
                        for k in range(2):
                            cnt = 0
                            for n in range(len(faces[unfold_faces[k][0]][unfold_faces[k][1]])):
                                if faces[unfold_faces[k][0]][unfold_faces[k][1]][n] in faces[i][j] and faces[i][j] != faces[unfold_faces[k^1][0]][unfold_faces[k^1][1]]:
                                    cnt += 1
                            if cnt == 2:
                                nextfaces.append((i,j, faces[i][j]))
                                nextfaces_order.append(k)
            #↓で絞れるはず
            #nextfaces[0]がfaces[unfold_faces[nextfaces_order[0]][0]][unfold_faces[nextfaces_order[0]][1]]の隣  nextfaces[1]がfaces[unfold_faces[nextfaces_order[1]][0]][unfold_faces[nextfaces_order[1]][1]]の隣、
            if len(nextfaces) == 2 and nextfaces[0][0] == nextfaces[1][0]:
                print("-----nakawari/kabuse--start-----")
                _, common1 = lists_common(nextfaces[0][2], faces[unfold_faces[nextfaces_order[0]][0]][unfold_faces[nextfaces_order[0]][1]])
                _, common2 = lists_common(nextfaces[1][2], faces[unfold_faces[nextfaces_order[1]][0]][unfold_faces[nextfaces_order[1]][1]])
                tptype = []
                if len(common1) == 2 and len(common2) == 2:
                    for i in range(len(edges)):
                        for j in range(len(edges[i])):
                            if common1[0] in edges[i][j][0] and common1[1] in edges[i][j][0]:
                                tptype.append(edges[i][j][1])
                                print(edges[i][j])
                            if common2[0] in edges[i][j][0] and common2[1] in edges[i][j][0]:
                                tptype.append(edges[i][j][1])
                                print(edges[i][j])
                print(len(tptype), fold_edges_list[l][1], tptype[0], tptype[-1])
                fl = 0
                if len(tptype) == 2 and tptype[0] == tptype[-1] and fold_edges_list[l][1] != tptype[0]:
                    print("------nakawari folding-----", fold_edges_list[l]) #中割り折り
                    fold_kind = "中割り折り"
                    print(faces[unfold_faces[nextfaces_order[0]][0]][unfold_faces[nextfaces_order[0]][1]], nextfaces[0][2])
                    print(faces[unfold_faces[nextfaces_order[1]][0]][unfold_faces[nextfaces_order[1]][1]], nextfaces[1][2])
                    if is_triangle_vertices_in_polygon(faces[unfold_faces[nextfaces_order[0]][0]][unfold_faces[nextfaces_order[0]][1]], nextfaces[0][2], vertices) or is_triangle_vertices_in_polygon(faces[unfold_faces[nextfaces_order[1]][0]][unfold_faces[nextfaces_order[1]][1]], nextfaces[1][2], vertices):
                        fold_kind = "四角形へのつぶし折り"
                        #折り辺表示用
                        folding_edge, forvertp = [], []
                        for i in range(len(faces[unfold_faces[0][0]][unfold_faces[0][1]])):
                            forvertp.append([faces[unfold_faces[0][0]][unfold_faces[0][1]][i-1], faces[unfold_faces[0][0]][unfold_faces[0][1]][i]])
                        for i in range(len(forvertp)):
                            vertp = []
                            for j in range(len(forvertp[i])):
                                vertp.append(vertices[forvertp[i][j]])
                            folding_edge.append(vertp)
                    fl = 1
                if len(tptype) == 2 and tptype[0] == tptype[-1] and fold_edges_list[l][1] == tptype[0]:
                    print("------kabuse folding-----", fold_edges_list[l]) #かぶせ折り
                    fold_kind = "かぶせ折り"
                    fl = 1
                if fl == 1:
                    print("nextfaces", nextfaces)
                    print("nextfaces_order", nextfaces_order)
                    #点の移動
                    print(nextfaces[0][2], faces[unfold_faces[nextfaces_order[0]][0]][unfold_faces[nextfaces_order[0]][1]])
                    _, common = lists_common(nextfaces[0][2], faces[unfold_faces[nextfaces_order[0]][0]][unfold_faces[nextfaces_order[0]][1]])
                    for i in range(len(fold_edges_list[l][0])):
                        if fold_edges_list[l][0][i] not in nextfaces[0][2]:
                            print("move vertice:", fold_edges_list[l][0][i], "edge:", common)
                            vertices[fold_edges_list[l][0][i]] = reflect_point_over_line(vertices[fold_edges_list[l][0][i]], vertices[common[0]], vertices[common[1]])
                    #commonは折り辺の一つ　これと同じグループの辺のすべてについて、面を2つずつ探す
                    #面の合成
                    for i in range(len(edges)):
                        for j in range(len(edges[i])):
                            if common[0] in edges[i][j][0] and common[1] in edges[i][j][0]:
                                tmp = i
                    folding_edges = copy.deepcopy(edges[tmp])
                    print("folding_edges", folding_edges) #折り辺のグループの全辺を保持

                    unfold_faces=[]
                    fl = 0
                    for i in range(len(folding_edges)):
                        tmp = []
                        for j in range(len(faces)):
                            for k, face in enumerate(faces[j]):
                                if folding_edges[i][0][0] in face and folding_edges[i][0][1] in face:
                                    tmp.append([j, k, face])
                        if len(tmp) != 2:
                            fl = 1
                            break
                        tmp.append(folding_edges[i])
                        unfold_faces.append(tmp)
                    if fl == 1:
                        continue
                    print("unfold_faces", unfold_faces) #くっつく面のペアを保持
                    print("faces", faces)
                    print("faces_index", faces_index)
                    for i in range(len(unfold_faces)):
                        if len(faces[unfold_faces[i][0][0]][unfold_faces[i][0][1]]) < len(faces[unfold_faces[i][1][0]][unfold_faces[i][1][1]]):
                            unfold_faces[i] = [unfold_faces[i][1], unfold_faces[i][0], unfold_faces[i][2]]
                        print("merge faces",faces[unfold_faces[i][0][0]][unfold_faces[i][0][1]], faces[unfold_faces[i][1][0]][unfold_faces[i][1][1]])
                        print("delete face", faces[unfold_faces[i][1][0]][unfold_faces[i][1][1]])
                        print("edge:", unfold_faces[i][2])
                        print("face_order", face_order)
                        delete = faces_index[unfold_faces[i][1][0]][unfold_faces[i][1][1]]
                        print("delete", delete)
                        if delete in face_order:
                            face_order.remove(delete) #faceorder update 中割り・かぶせ
                        faces[unfold_faces[i][0][0]][unfold_faces[i][0][1]] = merge_faces(faces[unfold_faces[i][0][0]][unfold_faces[i][0][1]], faces[unfold_faces[i][1][0]][unfold_faces[i][1][1]], unfold_faces[i][2][0])
                        print("merged face", faces[unfold_faces[i][0][0]][unfold_faces[i][0][1]])

                    print("fold_edges_list[l]", fold_edges_list[l])
                    for i in range(len(edges)):
                        for j in range(len(edges[i])):
                            if edges[i][j] == fold_edges_list[l]:
                                if edges[i][j][1] == 'M':
                                    print(edges[i][j], "->", (edges[i][j][0], 'V'))
                                    edges[i][j] = (edges[i][j][0], 'V')
                                elif edges[i][j][1] == 'V':
                                    print(edges[i][j], "->", (edges[i][j][0], 'M'))
                                    edges[i][j] = (edges[i][j][0], 'M')
                    #辺の合成
                    #折り辺に関する面だけじゃなくて同じグループの面、辺を全部やる必要ある
                    #一緒に面の共有点で辺探し 折り辺だけ別に合成

                    for i in range(len(unfold_faces)):
                        foldedges = copy.deepcopy(unfold_faces[i][2][0])
                        print("unfold_faces[i][0]", unfold_faces[i][0])
                        print("unfold_faces[i][1]", unfold_faces[i][1])
                        print("foldedges", foldedges)

                        for j in range(len(foldedges)): #折り辺の1点 i=0or1
                            #まずは途中点の特定
                            link_edges = [[],[]] #index
                            if unfold_faces[i][0][2].index(foldedges[j])+1 == len(unfold_faces[i][0][2]):
                                link_edges[0].append(foldedges[j])
                                link_edges[0].append(unfold_faces[i][0][2][unfold_faces[i][0][2].index(foldedges[j])-1])
                                link_edges[0].append(unfold_faces[i][0][2][0])
                            else:
                                link_edges[0].append(foldedges[j])
                                link_edges[0].append(unfold_faces[i][0][2][unfold_faces[i][0][2].index(foldedges[j])-1])
                                link_edges[0].append(unfold_faces[i][0][2][unfold_faces[i][0][2].index(foldedges[j])+1])
                            if unfold_faces[i][1][2].index(foldedges[j])+1 == len(unfold_faces[i][1][2]):
                                link_edges[1].append(foldedges[j])
                                link_edges[1].append(unfold_faces[i][1][2][unfold_faces[i][1][2].index(foldedges[j])-1])
                                link_edges[1].append(unfold_faces[i][1][2][0])
                            else:
                                link_edges[1].append(foldedges[j])
                                link_edges[1].append(unfold_faces[i][1][2][unfold_faces[i][1][2].index(foldedges[j])-1])
                                link_edges[1].append(unfold_faces[i][1][2][unfold_faces[i][1][2].index(foldedges[j])+1])
                            if foldedges[j^1] not in link_edges[0] or foldedges[j^1] not in link_edges[1]:
                                print("error: foldedges[j^1] not in link_edges")
                                continue
                            link_edges[0].remove(foldedges[j^1])
                            link_edges[1].remove(foldedges[j^1])
                            if link_edges[0]== link_edges[1]:
                                print("error: link_edges[0]== link_edges[1]")
                                continue
                            print("link_edges", link_edges)

                            #ここまででくっつく可能性のある2辺を挙げた
                            #この2辺の山折り谷折りが異なっていればこの折り辺は矛盾する折り方である
                            cnt = 0
                            for m in range(len(edges)):
                                for n in range(len(edges[m])):
                                    if edges[m][n][0] == [link_edges[0][0],link_edges[0][1]] or edges[m][n][0] == [link_edges[0][1],link_edges[0][0]]:
                                        tpm, tpn = m, n
                                        cnt += 1
                                    if edges[m][n][0] == [link_edges[1][0],link_edges[1][1]] or edges[m][n][0] == [link_edges[1][1],link_edges[1][0]]:
                                        tp2m, tp2n = m, n
                                        cnt += 1
                            if cnt == 2:
                                if edges[tpm][tpn][1] != edges[tp2m][tp2n][1]:
                                    print("edge types are different.", edges[tpm][tpn], edges[tp2m][tp2n])
                                    continue
                                tmpedgetype = edges[tpm][tpn][1]
                            #この2辺が平行か確認し、平行なら途中点であり、2辺をくっつけ、点を合成後の面から削除
                            #平行でなければ角であるため、スルー
                            print("link_edges", link_edges)
                            if parallel_check([vertices[link_edges[0][0]],vertices[link_edges[0][1]]], [vertices[link_edges[1][0]],vertices[link_edges[1][1]]]):
                                print("parallel:", link_edges)
                                #つながる前の線を削除してつながった辺を追加
                                li, com = lists_common(link_edges[0], link_edges[1])
                                cnt, fl = 0, 0
                                type = 'U'
                                for m in range(len(edges)):
                                    if com != []:
                                        for n in range(len(edges[m])):
                                            if [link_edges[0][0], link_edges[0][1]] == edges[m][n][0]:
                                                tmpm, order, type = m, 0, edges[m][n][1]
                                                cnt += 1
                                            elif [link_edges[0][1], link_edges[0][0]] == edges[m][n][0]:
                                                tmpm, order, type = m, 1, edges[m][n][1]
                                                cnt += 1
                                            if [link_edges[1][0], link_edges[1][1]] == edges[m][n][0]:
                                                tmp2m, order2, type = m, 0, edges[m][n][1]
                                                cnt += 1
                                            elif [link_edges[1][1], link_edges[1][0]] == edges[m][n][0]:
                                                tmp2m, order2, type = m, 1, edges[m][n][1]
                                                cnt += 1
                                            if cnt == 2:
                                                if order == 0:
                                                    edges[tmpm].remove(([link_edges[0][0], link_edges[0][1]],type))
                                                elif order == 1:
                                                    edges[tmpm].remove(([link_edges[0][1], link_edges[0][0]],type))
                                                if order2 == 0:
                                                    edges[tmp2m].remove(([link_edges[1][0], link_edges[1][1]],type))
                                                elif order2 == 1:
                                                    edges[tmp2m].remove(([link_edges[1][1], link_edges[1][0]],type))
                                                edges.insert(len(edges)-1, [(li, tmpedgetype)])
                                                print("insert edges", [(li, tmpedgetype)])
                                                fl = 1
                                                break
                                        if fl == 1:
                                            break
                                for _ in range(len(edges)):
                                    if [] in edges:
                                        edges.remove([])
                                #合成後の面から途中点を削除
                                if len(faces) != 2 or (len(faces) == 2 and ((abs(abs(vertices[foldedges[j]][0])-200.0) > 1E-10) or (abs(abs(vertices[foldedges[j]][1])-200.0) > 1E-10))):
                                    print("remove vertex:", com[0], faces[unfold_faces[i][0][0]][unfold_faces[i][0][1]])
                                    faces[unfold_faces[i][0][0]][unfold_faces[i][0][1]].remove(com[0])
                                    print("face after", faces[unfold_faces[i][0][0]][unfold_faces[i][0][1]])
                                    if len(faces[unfold_faces[i][0][0]][unfold_faces[i][0][1]]) <= 2:
                                        print("remove face len <= 2", faces[unfold_faces[i][0][0]][unfold_faces[i][0][1]])
                                        faces[unfold_faces[i][0][0]].pop(unfold_faces[i][0][1])
                    #いらなくなった面を削除
                    print("faces", faces)
                    print("remove faces", faces[unfold_faces[0][1][0]])
                    print("faces_index remove", faces_index[unfold_faces[0][1][0]])
                    faces.pop(unfold_faces[0][1][0])
                    faces_index.pop(unfold_faces[0][1][0])
                    print("faces", faces)
                    cnt = 0
                    for k in range(len(faces)):
                        cnt += len(faces[k])
                    if cnt == 1:
                        break

                    #折り線をedgesから削除 同じ座標の辺をまとめて削除
                    print("remove edgelist", folding_edges)
                    if folding_edges in edges:
                        edges.remove(folding_edges)
                    for _ in range(len(edges)):
                        if [] in edges:
                            edges.remove([])
                    for _ in range(len(faces)):
                        if [] in faces:
                            faces.remove([])
                        if [] in faces_index:
                            faces_index.remove([])
                    print("edges",edges)
                    print("faces",faces)
                    return vertices, faces, edges, faces_index, face_order, 1, facemax, fold_kind, folding_edge

    print("--nakawari/kabuse  check end--")
    #中割り折り・かぶせ折り以外
    unfold_face_index = 1
    vers1 = []
    vers2 = []
    for l in range(len(fold_edges_list)): #1つの折り辺
        print("------------")
        unfold_faces=[]
        #入力された折り辺を含む2面を探す
        if len(fold_edges_list) <= l:
            print("error: len(fold_edges_list) <= l")
            return input_vertices, input_faces, input_edges, input_faces_index, input_face_order, 0, facemax, fold_kind, []
        if l == 0:
            for i in range(len(faces)):
                for j, face in enumerate(faces[i]):
                    if fold_edges_list[l][0][0] in face and fold_edges_list[l][0][1] in face:
                        unfold_faces.append([i,j])
                        ver = []
                        for k in range(len(face)):
                            ver.append(vertices[face[k]])
                        vers1.append(ver)
        else:
            print("vers1", vers1)
            #len(vers1)-1個の点の座標が一致している面を探す
            for i in range(len(faces)):
                for j, face in enumerate(faces[i]):
                    ct = 0
                    for k in range(len(faces[i][j])):
                        for m in range(len(vers1)):
                            for n in range(len(vers1[m])):
                                if distance(vertices[faces[i][j][k]], vers1[m][n]) < 1E-8:
                                    ct += 1
                    if ct >= len(vers1) - 1 and fold_edges_list[l][0][0] in face and fold_edges_list[l][0][1] in face and [i,j] not in unfold_faces:
                        unfold_faces.append([i,j])

        if len(unfold_faces) != 2:
            print("error: len(unfold_faces) != 2", unfold_faces)
            return input_vertices, input_faces, input_edges, input_faces_index, input_face_order, 0, facemax, fold_kind, []
        print(faces[unfold_faces[0][0]][unfold_faces[0][1]], faces[unfold_faces[1][0]][unfold_faces[1][1]])

        # 折り返す面を決定 unfold_face_index
        cnt_face = 0
        for i in range(len(faces)):
            for j in range(len(faces[i])):
                cnt_face += 1
        cnt_faces_index = 0
        for i in range(len(faces_index)):
            for j in range(len(faces_index[i])):
                cnt_faces_index += 1
        print("len(faces), len(faces_index)", cnt_face, cnt_faces_index)
        fold_edges = [[0,0],[0,0]]#座標
        fold_edges[0] = vertices[fold_edges_list[l][0][0]]#座標
        fold_edges[1] = vertices[fold_edges_list[l][0][1]]#座標
        print("fold_edges", fold_edges_list[l][0])
        moved_point_in_face0 = copy.deepcopy(faces[unfold_faces[0][0]][unfold_faces[0][1]])
        moved_point_in_face1 = copy.deepcopy(faces[unfold_faces[1][0]][unfold_faces[1][1]])
        for i in range(len(fold_edges_list[l][0])):
            moved_point_in_face0.remove(fold_edges_list[l][0][i])
            moved_point_in_face1.remove(fold_edges_list[l][0][i])
        cnt0, cnt1 = 0, 0
        cnt0_face, cnt1_face = [], []
        len0, len1 = len(faces[unfold_faces[0][0]][unfold_faces[0][1]]), len(faces[unfold_faces[1][0]][unfold_faces[1][1]])
        print("len0", len0, "len1", len1)
        for i in range(len0):
            coedge = [faces[unfold_faces[0][0]][unfold_faces[0][1]][i], faces[unfold_faces[0][0]][unfold_faces[0][1]][(i + 1) % len(faces[unfold_faces[0][0]][unfold_faces[0][1]])]]
            for j in range(len(faces)):
                for k in range(len(faces[j])):
                    if coedge[0] in faces[j][k] and coedge[1] in faces[j][k] and unfold_faces[0][0] != j:
                        cnt0 += 1
                        cnt0_face.append(faces[j][k])
        for i in range(len1):
            coedge = [faces[unfold_faces[1][0]][unfold_faces[1][1]][i], faces[unfold_faces[1][0]][unfold_faces[1][1]][(i + 1) % len(faces[unfold_faces[1][0]][unfold_faces[1][1]])]]
            for j in range(len(faces)):
                for k in range(len(faces[j])):
                    if coedge[0] in faces[j][k] and coedge[1] in faces[j][k] and unfold_faces[1][0] != j:
                        cnt1 += 1
                        cnt1_face.append(faces[j][k])
        print("cnt0:", cnt0, cnt0_face, "cnt1:", cnt1, cnt1_face)
        if ((len0 <= 4 and len1 <= 4) and ((cnt0 >= 3 and cnt1 >= 3) or max(cnt0, cnt1) >= 4)):
            print("error: cnt1, cnt2")
            return input_vertices, input_faces, input_edges, input_faces_index, input_face_order, 0, facemax, fold_kind, []
        com0, com1 = 0, 0
        for i in range(len0):
            if faces[unfold_faces[0][0]][unfold_faces[0][1]][i] in moved_vertices:
                com0 += 1
        for i in range(len1):
            if faces[unfold_faces[1][0]][unfold_faces[1][1]][i] in moved_vertices:
                com1 += 1
        print("moved_vertices", moved_vertices)
        print("com0:", com0, "com1:", com1)
        face0_ver_bool, face1_ver_bool = False, False
        for i in range(len(moved_point_in_face0)):
            if moved_vertice_same_point(reflect_point_over_line(vertices[moved_point_in_face0[i]], fold_edges[0], fold_edges[1]), vertices):
                face0_ver_bool = True
        for i in range(len(moved_point_in_face1)):
            if moved_vertice_same_point(reflect_point_over_line(vertices[moved_point_in_face1[i]], fold_edges[0], fold_edges[1]), vertices):
                face1_ver_bool = True
        print("unique", unique_edges)
        nextface_unique = [0,0]
        nextface_unique_edge, cntface = 0, 0
        if len(unique_edges) != 0:
            #もし隣の面がunique辺持ってたらそっちを折り返す
            for i in range(2):#faces[unfold_faces[i][0]][unfold_faces[i][1]]の隣の面を探す
                nextfacesi = []
                for j in range(len(faces[unfold_faces[i][0]][unfold_faces[i][1]])):
                    tped = [faces[unfold_faces[i][0]][unfold_faces[i][1]][j-1], faces[unfold_faces[i][0]][unfold_faces[i][1]][j]]
                    for k in range(len(faces)):
                        for m in range(len(faces[k])):
                            if tped[0] in faces[k][m] and tped[1] in faces[k][m] and faces[k][m] != faces[unfold_faces[0][0]][unfold_faces[0][1]] and faces[k][m] != faces[unfold_faces[1][0]][unfold_faces[1][1]]:
                                nextfacesi = faces[k][m]
                                print("nextfacesi", nextfacesi)
                    #faces[unfold_faces[i][0]][unfold_faces[i][1]]の隣の面がnextfacesi
                    if nextfacesi != []:
                        for k in range(len(unique_edges)):
                            for j in range(len(nextfacesi)):
                                if (unique_edges[k][0][0] == nextfacesi[j-1] and unique_edges[k][0][1] == nextfacesi[j]) or (unique_edges[k][0][0] == nextfacesi[j] and unique_edges[k][0][1] == nextfacesi[j-1]):
                                    nextface_unique[i] = 1
        if nextface_unique[0] == 1 and nextface_unique[1] != 1:
            unfold_face_index = 0
            nextface_unique_edge = 1
            print("nextface_unique")
        elif nextface_unique[0] != 1 and nextface_unique[1] == 1:
            unfold_face_index = 1
            nextface_unique_edge = 1
            print("nextface_unique")
        else:
            if (cnt0 == 1 and cnt1 != 1 and cnt0_face[0] == faces[unfold_faces[1][0]][unfold_faces[1][1]]) or (cnt0 == 0 and cnt1 != 0):
                unfold_face_index = 0
                cntface = 1
                print("cnt0_face")
            elif (cnt0 != 1 and cnt1 == 1 and cnt1_face[0] == faces[unfold_faces[0][0]][unfold_faces[0][1]]) or (cnt0 != 0 and cnt1 == 0):
                unfold_face_index = 1
                cntface = 1
                print("cnt1_face")
            else:
                if face0_ver_bool and not face1_ver_bool:
                    unfold_face_index = 0
                    print("moved_vertice_same_point")
                elif not face0_ver_bool and face1_ver_bool:
                    unfold_face_index = 1
                    print("moved_vertice_same_point")
                else:
                    if com0 > com1:
                        unfold_face_index = 0
                        print("com0 > com1")
                    elif com0 < com1:
                        unfold_face_index = 1
                        print("com0 < com1")
                    else:
                        if len0 < len1:
                            unfold_face_index = 0
                            print("len(faces)")
                        elif len0 > len1:
                            unfold_face_index = 1
                            print("len(faces)")
                        else:
                            if polygon_area(faces[unfold_faces[0][0]][unfold_faces[0][1]], vertices) < polygon_area(faces[unfold_faces[1][0]][unfold_faces[1][1]], vertices):
                                unfold_face_index = 0
                                print("polygon_area")
                            else:
                                unfold_face_index = 1
                                print("polygon_area")
        print(faces[unfold_faces[unfold_face_index^1][0]][unfold_faces[unfold_face_index^1][1]], faces[unfold_faces[unfold_face_index][0]][unfold_faces[unfold_face_index][1]])
        print("unfold_face_index", unfold_face_index)

        #正方形が開かれている場合の展開(花弁折り)
        #faces[unfold_faces[m^1][0]][unfold_faces[m^1][1]]がB辺もつ三角形、faces[unfold_faces[m][0]][unfold_faces[m][1]]が普通の三角形
        for m in range(2):
            if istriangle_typeb(edges, faces[unfold_faces[m^1][0]][unfold_faces[m^1][1]]) and len(faces[unfold_faces[m][0]][unfold_faces[m][1]]) == 3 and tsubushi_square != 1:
                next_faces = []
                for i in range(len(faces)):
                    for j in range(len(faces[i])):
                        cnt = 0
                        for k in range(len(faces[unfold_faces[0][0]][unfold_faces[0][1]])):
                            if faces[unfold_faces[m][0]][unfold_faces[m][1]][k] in faces[i][j] and faces[unfold_faces[m^1][0]][unfold_faces[m^1][1]] != faces[i][j]:
                                cnt += 1
                        if cnt == 2:
                            next_faces.append((i,j, faces[i][j]))
                if len(next_faces) == 2 and len(next_faces[0][2]) == 3 and len(next_faces[-1][2]) == 3 and (istriangle_typeb(edges, next_faces[0][2]) or istriangle_typeb(edges, next_faces[-1][2])):
                    print("----------------close to square start----------------------")
                    for p in range(2):
                    #next_faces[p][2]がB辺もつ三角形、next_faces[p^1][2]が三角形
                    #まずB持ち三角形の2面でB辺に垂線下ろして交点を頂点として追加 辺も追加
                        tpface = next_faces[p^1][2]
                        Bedge1, Bedge2 = (), ()
                        Bedge2 = ()
                        rmedges, rmfaces = [], []
                        for i in range(len(edges[-1])):
                            if edges[-1][i][0][0] in faces[unfold_faces[m^1][0]][unfold_faces[m^1][1]] and edges[-1][i][0][1] in faces[unfold_faces[m^1][0]][unfold_faces[m^1][1]]:
                                Bedge1 = edges[-1][i][0]
                                rmedges.append(edges[-1][i])
                            if edges[-1][i][0][0] in next_faces[p][2] and edges[-1][i][0][1] in next_faces[p][2]:
                                Bedge2 = edges[-1][i][0]
                                rmedges.append(edges[-1][i])
                        print(Bedge1, Bedge2)
                        if Bedge1 != () and Bedge2 != ():
                            [oppositevertice1], _ = lists_common(Bedge1, faces[unfold_faces[m^1][0]][unfold_faces[m^1][1]])
                            print(Bedge2, next_faces[p][2])
                            [oppositevertice2], _ = lists_common(Bedge2, next_faces[p][2])
                            print(oppositevertice1, oppositevertice2)
                            next_faces_point = []
                            for i in range(len(tpface)):
                                if tpface[i] != oppositevertice1 and tpface[i] != oppositevertice2:
                                    next_faces_point.append(tpface[i])
                            vertices, edges = make_perpendicular_line(vertices, edges, oppositevertice1, Bedge1)
                            vertices, edges = make_perpendicular_line(vertices, edges, oppositevertice2, Bedge2)
                            #halftriangle1は折り辺含む三角の半分の折り辺含む方、halftriangle2は折り辺含む三角の半分の折り辺含まない方
                            #halftriangle3は折り辺含まない三角の半分のhalftriangle1側、halftriangle4は折り辺含まない三角の半分のhalftriangle2側
                            halftriangle1 = copy.deepcopy(edges[-2][0][0])
                            halftriangle1.extend(fold_edges_list[l][0])
                            halftriangle1 = list(set(halftriangle1))
                            print("halftriangle1", halftriangle1)
                            halftriangle2 = copy.deepcopy(edges[-2][0][0])
                            halftriangle2.extend(faces[unfold_faces[m^1][0]][unfold_faces[m^1][1]])
                            halftriangle2.remove(fold_edges_list[l][0][0])
                            halftriangle2.remove(fold_edges_list[l][0][1])
                            print("halftriangle2", halftriangle2)
                            _, common1 = lists_common(faces[unfold_faces[m][0]][unfold_faces[m][1]], next_faces[p][2])
                            halftriangle3 = copy.deepcopy(edges[-1][0][0])
                            halftriangle3.extend(common1)
                            halftriangle3 = list(set(halftriangle3))
                            print("halftriangle3", halftriangle3)
                            halftriangle4 = copy.deepcopy(edges[-1][0][0])
                            halftriangle4.extend(next_faces[p][2])
                            halftriangle4.remove(common1[0])
                            halftriangle4.remove(common1[1])
                            print("halftriangle4", halftriangle4)
                            #折り辺表示用
                            folding_edge = []
                            _, c = lists_common(faces[unfold_faces[m][0]][unfold_faces[m][1]], next_faces[p^1][2])
                            _, [c2] = lists_common(halftriangle1, halftriangle3)
                            c.append(c2)
                            for i in range(len(c)):
                                folding_edge.append([vertices[c[i-1]], vertices[c[i]]])
                            #点の移動
                            #三角形側の二面の共通点を垂線下ろした2点の辺で折り返す
                            _, common2 = lists_common(halftriangle1, halftriangle3)
                            if common2[0] not in moved:
                                print("move vertice:", common2[0], "edge:", [oppositevertice1, oppositevertice2])
                                vertices[common2[0]] = reflect_point_over_line(vertices[common2[0]], vertices[oppositevertice1], vertices[oppositevertice2])
                                moved.append(common2[0])
                            #垂線の足をもう一方の二辺で折り返す
                            tplist = copy.deepcopy(halftriangle2)
                            tplist.remove(oppositevertice1)
                            tplist.remove(len(vertices)-2)
                            point1 = tplist[0]
                            print("point1", point1)
                            tplist = copy.deepcopy(halftriangle4)
                            tplist.remove(oppositevertice2)
                            tplist.remove(len(vertices)-1)
                            point2 = tplist[0]
                            print("point2", point2)
                            if len(vertices)-2 not in moved:
                                print("move vertice:", len(vertices)-2, "edge:", [point1, oppositevertice1])
                                vertices[len(vertices)-2] = reflect_point_over_line(vertices[len(vertices)-2], vertices[point1], vertices[oppositevertice1])
                                moved.append(len(vertices)-2)
                            if len(vertices)-1 not in moved:
                                print("move vertice:", len(vertices)-1, "edge:", [point2,oppositevertice2])
                                vertices[len(vertices)-1] = reflect_point_over_line(vertices[len(vertices)-1], vertices[point2], vertices[oppositevertice2])
                                moved.append(len(vertices)-1)
                            #面の合成
                            #下半分の面は隣り合ってる面に単に合成
                            print(next_faces_point)
                            tpi,tpj,tp2i,tp2j,tmpi,tmpj,tmp2i,tmp2j = np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf
                            for i in range(len(edges)):
                                for j in range(len(edges[i])):
                                    if oppositevertice1 in edges[i][j][0] and point1 in edges[i][j][0]:
                                        tpi, tpj = i, j
                                    if oppositevertice2 in edges[i][j][0] and point2 in edges[i][j][0]:
                                        tp2i, tp2j = i, j
                            for i in range(len(faces)):
                                for j in range(len(faces[i])):
                                    if point1 in faces[i][j] and oppositevertice1 in faces[i][j] and next_faces_point[0] in faces[i][j]:
                                        tmpi, tmpj = i, j
                                    if point2 in faces[i][j] and oppositevertice2 in faces[i][j] and next_faces_point[0] in faces[i][j]:
                                        tmp2i, tmp2j = i, j
                            if tpi==np.inf or tpj==np.inf or tp2i==np.inf or tp2j==np.inf or tmpi==np.inf or tmpj==np.inf or tmp2i==np.inf or tmp2j==np.inf:
                                return input_vertices, input_faces, input_edges, input_faces_index, input_face_order, 0, facemax, fold_kind, []
                            print("merge faces",faces[tmpi][tmpj], halftriangle2, "edge:", edges[tpi][tpj][0])
                            faces[tmpi][tmpj] = merge_faces(faces[tmpi][tmpj], halftriangle2, edges[tpi][tpj][0])
                            print("merge faces",faces[tmp2i][tmp2j], halftriangle4, "edge:", edges[tp2i][tp2j][0])
                            faces[tmp2i][tmp2j] = merge_faces(faces[tmp2i][tmp2j], halftriangle4, edges[tp2i][tp2j][0])
                            print(faces)
                            #上半分の面はもう一方の上側と共通して隣り合ってる面とそのさらに隣の面と合成
                            print("merge faces",faces[unfold_faces[m][0]][unfold_faces[m][1]], halftriangle1, "edge:", fold_edges_list[l][0])
                            delete = faces_index[unfold_faces[m^1][0]][unfold_faces[m^1][1]]
                            print("face_order", face_order)
                            print("delete", delete)
                            face_order.remove(delete) #faceorder update　花弁
                            faces[unfold_faces[m][0]][unfold_faces[m][1]] = merge_faces(faces[unfold_faces[m][0]][unfold_faces[m][1]], halftriangle1, fold_edges_list[l][0])
                            for i in range(len(faces)):
                                for j in range(len(faces[i])):
                                    if faces[i][j] == next_faces[p][2]:
                                        delete = faces_index[i][j]
                            print("face_order", face_order)
                            print("delete", delete)
                            face_order.remove(delete) #faceorder update　花弁
                            print("merge faces",faces[unfold_faces[m][0]][unfold_faces[m][1]], halftriangle3, "edge:", [oppositevertice2, common2[0]])
                            faces[unfold_faces[m][0]][unfold_faces[m][1]] = merge_faces(faces[unfold_faces[m][0]][unfold_faces[m][1]], halftriangle3, [oppositevertice2, common2[0]])
                            delete = faces_index[unfold_faces[m][0]][unfold_faces[m][1]]
                            print("face_order", face_order)
                            print("delete", delete)
                            face_order.remove(delete) #faceorder update　花弁
                            print("merge faces",faces[next_faces[p^1][0]][next_faces[p^1][1]], faces[unfold_faces[m][0]][unfold_faces[m][1]], "edge:", [oppositevertice1, oppositevertice2])
                            faces[next_faces[p^1][0]][next_faces[p^1][1]] = merge_faces(faces[next_faces[p^1][0]][next_faces[p^1][1]], faces[unfold_faces[m][0]][unfold_faces[m][1]], [oppositevertice1, oppositevertice2])
                            print(faces)
                            rmfaces.append(faces[unfold_faces[m^1][0]][unfold_faces[m^1][1]])
                            rmfaces.append(faces[next_faces[p][0]][next_faces[p][1]])
                            rmfaces.append(faces[unfold_faces[m][0]][unfold_faces[m][1]])
                            rmedges.append(edges[tpi][tpj])
                            rmedges.append(edges[tp2i][tp2j])
                            rmedges.append(fold_edges_list[l])
                            for i in range(len(edges)):
                                for j in range(len(edges[i])):
                                    if edges[i][j][0] == [oppositevertice2, common2[0]] or edges[i][j][0] == [common2[0], oppositevertice2]:
                                        rmedges.append(edges[i][j])
                                    if edges[i][j][0] == [oppositevertice1, oppositevertice2] or edges[i][j][0] == [oppositevertice2, oppositevertice1]:
                                        rmedges.append(edges[i][j])
                            adedges = [[([Bedge1[0],len(vertices)-2], 'B')],[([Bedge1[1],len(vertices)-2], 'B')],[([Bedge2[0],len(vertices)-1], 'B')],[([Bedge2[1],len(vertices)-1], 'B')]]
                            rmedges.append(Bedge1)
                            #辺の合成
                            #追加した垂線が隣の隣の面の辺とくっつくかどうか
                            #tpfaceと垂線がくっつくか
                            opposite = [oppositevertice1, oppositevertice2]
                            for i in range(-1, len(tpface)-1):
                                for o in range(1,3):
                                    if parallel_check([vertices[tpface[i]], vertices[tpface[i+1]]], [vertices[opposite[o-1]], vertices[len(vertices)-3+o]]):
                                        print("parallel:", [tpface[i], tpface[i+1]], [opposite[o-1], len(vertices)-3+o])
                                        #つながる前の線を削除してつながった辺を追加 common[0]は途中点
                                        li, common = lists_common([tpface[i], tpface[i+1]], [opposite[o-1], len(vertices)-3+o])
                                        order = 2
                                        fl = 0
                                        rmedges.append(edges[-3+o][0])
                                        tpedge_type = 'M'
                                        type = 'U'
                                        for j in range(len(edges)):
                                            if common != []:
                                                for k in range(len(edges[j])):
                                                    if [tpface[i],tpface[i+1]] == edges[j][k][0]:
                                                        tp3j, order, type = j, 0, edges[j][k][1]
                                                    elif [tpface[i+1],tpface[i]] == edges[j][k][0]:
                                                        tp3j, order, type = j, 1, edges[j][k][1]
                                                    if order == 0 or order == 1:
                                                        if tpedge_type != type:
                                                            print("edge types are different.", tpedge_type, type)
                                                            return input_vertices, input_faces, input_edges, input_faces_index, input_face_order, 0, facemax, fold_kind, []
                                                        if order == 0:
                                                            rmedges.append(([tpface[i],tpface[i+1]],type))
                                                        elif order == 1:
                                                            rmedges.append(([tpface[i+1],tpface[i]],type))
                                                        fl = 1
                                                        break
                                                if fl == 1:
                                                    adedges.append([(li, type)])
                                                    break
                                        if common != []:
                                            #合成後の面から途中点を削除
                                            print(common)
                                            if common[0] in faces[tmpi][tmpj]:
                                                faces[tmpi][tmpj].remove(common[0])
                                                print("remove vertex:", common[0], faces[tmpi][tmpj])
                                            if common[0] in faces[tmp2i][tmp2j]:
                                                faces[tmp2i][tmp2j].remove(common[0])
                                                print("remove vertex:", common[0], faces[tmp2i][tmp2j])
                                            if common[0] in faces[next_faces[p^1][0]][next_faces[p^1][1]]:
                                                faces[next_faces[p^1][0]][next_faces[p^1][1]].remove(common[0])
                                                print("remove vertex:", common[0], faces[next_faces[p^1][0]][next_faces[p^1][1]])
                                            print("faces", faces)
                                            print("edges", edges)
                            print(rmfaces)
                            print(rmedges)
                            #いらなくなった面を削除
                            for i in range(len(rmfaces)):
                                for j in range(len(faces)):
                                    if rmfaces[i] in faces[j]:
                                        print("remove faces", rmfaces[i])
                                        faces_index[j].pop(faces[j].index(rmfaces[i]))
                                        faces[j].remove(rmfaces[i])
                            for i in range(len(faces)):
                                if [] in faces:
                                    faces.remove([])
                                if [] in faces_index:
                                    faces_index.remove([])
                            #辺をedgesから削除
                            for i in range(len(rmedges)):
                                for j in range(len(edges)):
                                    if rmedges[i] in edges[j]:
                                        print("remove edges", rmedges[i])
                                        edges[j].remove(rmedges[i])
                            for i in range(len(edges)):
                                if [] in edges:
                                    edges.remove([])
                            for i in range(len(adedges)):
                                print("insert edges", adedges[i])
                                edges.append(adedges[i])
                            print("edges",edges)
                            print("faces",faces)
                            break
                    print("----------------close to square end----------------------")
                    fold_kind = "花弁折り"
                    return vertices, faces, edges, faces_index, face_order, 1, facemax, fold_kind, folding_edge

        #つぶし折り（四角形）
        #faces[unfold_faces[n^1][0]][unfold_faces[n^1][1]]が四角形、faces[unfold_faces[n][0]][unfold_faces[n][1]]が隣り合う面
        for n in range(2):
            if issquare(vertices, faces[unfold_faces[n^1][0]][unfold_faces[n^1][1]]) and has_diagonal_as_edge(vertices, faces[unfold_faces[n][0]][unfold_faces[n][1]], faces[unfold_faces[n^1][0]][unfold_faces[n^1][1]]):
                for i in range(len(faces[unfold_faces[n^1][0]][unfold_faces[n^1][1]])):
                    for j in range(len(faces)):
                        for k in range(len(faces[j])):
                            if faces[unfold_faces[n^1][0]][unfold_faces[n^1][1]][i-1] in faces[j][k] and faces[unfold_faces[n^1][0]][unfold_faces[n^1][1]][i-1] in faces[j][k] and j != unfold_faces[0][0] and j != unfold_faces[1][0]:
                                tpface = faces[j][k]
                print("tpface", tpface)
                if not has_diagonal_as_edge(vertices, tpface, faces[unfold_faces[n^1][0]][unfold_faces[n^1][1]]):
                    continue
                print("-------------tsubushi(square) start-------------------------")
                tpface1, tpface2 = faces[unfold_faces[n^1][0]][unfold_faces[n^1][1]], faces[unfold_faces[n][0]][unfold_faces[n][1]]
                tmpi, tmpj, flag = 0,0,0
                for i in range(len(edges)):
                    for j in range(len(edges[i])):
                        if edges[i][j][0][0] in faces[unfold_faces[n^1][0]][unfold_faces[n^1][1]] and edges[i][j][0][1] in faces[unfold_faces[n^1][0]][unfold_faces[n^1][1]] and edges[i][j][0] != fold_edges_list[l][0] and edges[i][j][1] != 'B':
                            tmpi, tmpj = i, j
                            flag = 1
                            break
                    if flag == 1:
                        break
                #四角形の対角線の2点を探す
                diagonal = []
                print(faces[unfold_faces[n^1][0]][unfold_faces[n^1][1]])
                for k in range(len(faces[unfold_faces[n^1][0]][unfold_faces[n^1][1]])):
                    if faces[unfold_faces[n^1][0]][unfold_faces[n^1][1]][k] not in edges[tmpi][tmpj][0] and faces[unfold_faces[n^1][0]][unfold_faces[n^1][1]][k] not in fold_edges_list[l][0]:
                        diagonal.append(faces[unfold_faces[n^1][0]][unfold_faces[n^1][1]][k])
                        print(faces[unfold_faces[n^1][0]][unfold_faces[n^1][1]][k], "append")
                print(edges[tmpi][tmpj][0], fold_edges_list[l][0])
                _, common = lists_common(edges[tmpi][tmpj][0], fold_edges_list[l][0])
                if common == []:
                    continue
                diagonal.append(common[0])
                if len(diagonal) != 2:
                    return input_vertices, input_faces, input_edges, input_faces_index, input_face_order, 0, facemax, fold_kind, []
                print("------diagonal---",diagonal)
                #fold_edges_list[l][0]は折り目
                #edges[tmpi][tmpj][0]は四角形のもう1つの折り目
                #diagonalは四角形の対角線(1つ)

                #まずは点を移動
                #隣り合ってる面との共有辺を対角線に対して対称移動
                for vertice in fold_edges_list[l][0]:
                    if vertice not in moved:
                        vertices[vertice] = reflect_point_over_line(vertices[vertice], vertices[diagonal[0]], vertices[diagonal[1]])
                        moved.append(vertice)
                #対角線をもう一方の辺に対して対称移動
                for vertice in diagonal:
                    if vertice not in moved:
                        vertices[vertice] = reflect_point_over_line(vertices[vertice], vertices[edges[tmpi][tmpj][0][0]], vertices[edges[tmpi][tmpj][0][1]])
                        moved.append(vertice)
                #次に面を合体
                #四角形を対角線で分割 halfsquare1, halfsquare2
                halfsquare1, halfsquare2 = [], []
                for i in range(len(diagonal)):
                    halfsquare1.append(diagonal[i])
                    halfsquare2.append(diagonal[i])
                for i in range(len(fold_edges_list[l][0])):
                    halfsquare1.append(fold_edges_list[l][0][i])
                for i in range(len(edges[tmpi][tmpj][0])):
                    halfsquare2.append(edges[tmpi][tmpj][0][i])
                halfsquare1 = list(set(halfsquare1))
                halfsquare2 = list(set(halfsquare2))
                folding_edge = []
                for _ in range(len(diagonal)):
                    folding_edge.append([vertices[diagonal[0]], vertices[diagonal[1]]])
                for i in range(len(halfsquare1)):
                    folding_edge.append([vertices[halfsquare1[i-1]], vertices[halfsquare1[i]]])
                for i in range(len(halfsquare2)):
                    folding_edge.append([vertices[halfsquare2[i-1]], vertices[halfsquare2[i]]])
                #表面を合体させる
                #四角形の隣の面の隣の面を探す
                print(edges)
                tmpedge = []
                tpi, tpj = 0, 0
                for i in range(len(edges)):
                    for j in range(len(edges[i])):
                        if edges[i][j][0][0] in faces[unfold_faces[n][0]][unfold_faces[n][1]] and edges[i][j][0][1] in faces[unfold_faces[n][0]][unfold_faces[n][1]] and edges[i][j][1]!='B' and edges[i][j][0]!=fold_edges_list[l][0]:
                            tmpedge.append(edges[i][j][0])
                for i in range(len(faces)):
                    for j in range(len(faces[i])):
                        if tmpedge[0][0] in faces[i][j] and tmpedge[0][1] in faces[i][j] and faces[i][j] != faces[unfold_faces[n][0]][unfold_faces[n][1]]:
                            tpi, tpj = i, j
                tmpface1 = copy.deepcopy(faces[tpi][tpj])
                #tpface1は四角形、tpface2はhalfsquare1の隣、tmpface1はtpface2の隣、tmpface2はhalfsquare2の隣
                print("merge faces",faces[tpi][tpj], faces[unfold_faces[n][0]][unfold_faces[n][1]], "edge:", tmpedge[0])
                delete = faces_index[unfold_faces[n][0]][unfold_faces[n][1]]
                face_order.remove(delete) #faceorder update　つぶし(四角形)
                print("delete", delete)
                faces[tpi][tpj] = merge_faces(faces[tpi][tpj], faces[unfold_faces[n][0]][unfold_faces[n][1]], tmpedge[0])
                print("merge faces",faces[tpi][tpj], halfsquare1, "edge:", fold_edges_list[l][0])
                delete = faces_index[unfold_faces[n^1][0]][unfold_faces[n^1][1]]
                face_order.remove(delete) #faceorder update　つぶし(四角形)
                print("delete", delete)
                faces[tpi][tpj] = merge_faces(faces[tpi][tpj], halfsquare1, fold_edges_list[l][0])
                print(faces[tpi][tpj])
                rmedge1 = tmpedge[0]
                rmedge2 = fold_edges_list[l]

                #裏面を合体させる
                #halfsquare2と隣り合っている面を探す
                tp2i, tp2j = 0,0
                for i in range(len(faces)):
                    for j in range(len(faces[i])):
                        if edges[tmpi][tmpj][0][0] in faces[i][j] and edges[tmpi][tmpj][0][1] in faces[i][j] and faces[i][j] != faces[unfold_faces[n^1][0]][unfold_faces[n^1][1]]:
                            tp2i, tp2j = i, j
                tmpface2 = faces[tp2i][tp2j]
                print("merge faces",faces[tp2i][tp2j], halfsquare2, "edge:", edges[tmpi][tmpj][0])
                faces[tp2i][tp2j] = merge_faces(faces[tp2i][tp2j], halfsquare2, edges[tmpi][tmpj][0])
                rmedge3 = edges[tmpi][tmpj]
                print(faces[tp2i][tp2j])
                print(faces)
                #次に辺をくっつける
                #まずは対角線
                diagonal_link = []
                for i in range(len(diagonal)):#対角線の1点
                    if diagonal[i] in tmpface1:
                        if tmpface1.index(diagonal[i])+1 == len(tmpface1):
                            diagonal_link.append([diagonal[i], tmpface1[tmpface1.index(diagonal[i])-1]])
                            diagonal_link.append([diagonal[i], tmpface1[0]])
                        else:
                            diagonal_link.append([diagonal[i], tmpface1[tmpface1.index(diagonal[i])+1]])
                            diagonal_link.append([diagonal[i], tmpface1[tmpface1.index(diagonal[i])-1]])
                print(diagonal_link)
                fl = 0
                for i in range(len(diagonal_link)):
                    if parallel_check([vertices[diagonal[0]],vertices[diagonal[1]]], [vertices[diagonal_link[i][0]],vertices[diagonal_link[i][1]]]):
                        print("parallel_link_edges", diagonal, diagonal_link[i])
                        #つながる前の線を削除してつながった辺を追加 common[0]は途中点
                        print(edges)
                        li, common = lists_common(diagonal, diagonal_link[i])
                        order = 2
                        for j in range(len(edges)):
                            if common != []:
                                type = 'U'
                                for k in range(len(edges[j])):
                                    n = 0
                                    if [diagonal_link[i][n], diagonal_link[i][n^1]] == edges[j][k][0]:
                                        tp3j, order, type = j, n, edges[j][k][1]
                                    elif [diagonal_link[i][n^1], diagonal_link[i][n]] == edges[j][k][0]:
                                        tp3j, order, type = j, n^1, edges[j][k][1]
                                    if order == 0 or order ==1:
                                        edges[tp3j].remove(([diagonal_link[i][order], diagonal_link[i][order^1]],type))
                                        fl = 1
                                        break
                                if fl == 1:
                                    break
                        edges.insert(len(edges)-1, [(li, type)])
                        print("insert edges", [(li, type)])
                        for j in range(len(edges)):
                            if [] in edges:
                                edges.remove([])
                        #合成後の面から途中点を削除
                        faces[tpi][tpj].remove(common[0])
                        faces[tp2i][tp2j].remove(common[0])
                        print("remove vertex:", common[0], faces[tpi][tpj])
                        print("remove vertex:", common[0], faces[tp2i][tp2j])
                        print(faces)
                        print(edges)
                        break
                if fl == 0:
                    #対角線が他の辺とくっつかなかったらedgesに追加
                    edges.insert(len(edges)-1, [(diagonal, 'M')])
                    print("insert diagonal edges", [(diagonal, 'M')])

                #あとは各辺について隣りあう面の辺かつもともと座標が同じだった辺同士がくっつく
                #四角形の各辺をedgesから探して同じグループの辺の中で隣り合う面の辺があるかどうかを探す
                fl=0
                tp4j = 0
                for i in range(len(tpface1)):
                    tpedge = [tpface1[i-1] , tpface1[i]]
                    for j in range(len(edges)):
                        fl = 0
                        for k in range(len(edges[j])):
                            if edges[j][k][0] == tpedge or edges[j][k][0] == tpedge[::-1]:
                                tp4j=j
                                fl=1
                                tpedge_type = edges[j][k][1]
                                break
                        if fl==1:
                            break
                    #edges[tp4j]に四角形の辺が存在
                    #同じグループの中で隣り合う面の辺があるかどうかを探す
                    cnt = 0
                    rmedges = []
                    adedges = []
                    for j in range(len(edges[tp4j])):
                        #tpedgeとedges[tp4j][j][0]がくっつく
                        if ((edges[tp4j][j][0][0] in tpface2 and edges[tp4j][j][0][1] in tpface2) or (edges[tp4j][j][0][0] in tmpface2 and edges[tp4j][j][0][1] in tmpface2)) and (set(tpedge) != set(edges[tp4j][j][0])):
                            if (distance(input_vertices[edges[tp4j][j][0][0]], input_vertices[tpedge[0]]) < 1E-10 and distance(input_vertices[edges[tp4j][j][0][1]], input_vertices[tpedge[1]]) < 1E-10) or (distance(input_vertices[edges[tp4j][j][0][1]], input_vertices[tpedge[0]]) < 1E-10 and distance(input_vertices[edges[tp4j][j][0][0]], input_vertices[tpedge[1]]) < 1E-10):
                                if parallel_check([vertices[edges[tp4j][j][0][0]],vertices[edges[tp4j][j][0][1]]], [vertices[tpedge[0]],vertices[tpedge[1]]]):
                                    print("parallel:", tpedge, edges[tp4j][j][0])
                                    #つながる前の線を削除してつながった辺を追加 common[0]は途中点
                                    li, common = lists_common(tpedge, edges[tp4j][j][0])
                                    order = 2
                                    fl = 0
                                    rmedges.append(edges[tp4j][j])
                                    for j in range(len(edges)):
                                        if common != []:
                                            type = 'U'
                                            for k in range(len(edges[j])):
                                                if tpedge == edges[j][k][0]:
                                                    tp3j, order, type = j, 0, edges[j][k][1]
                                                elif tpedge[::-1] == edges[j][k][0]:
                                                    tp3j, order, type = j, 1, edges[j][k][1]
                                                if order == 0 or order == 1:
                                                    if tpedge_type != type:
                                                        print("edge types are different.", tpedge_type, type)
                                                        return input_vertices, input_faces, input_edges, input_faces_index, input_face_order, 0, facemax, fold_kind, []
                                                    if order == 0:
                                                        rmedges.append((tpedge,type))
                                                    elif order == 1:
                                                        rmedges.append((tpedge[::-1],type))
                                                    fl = 1
                                                    break
                                            if fl == 1:
                                                break
                                    adedges.append([(li, type)])

                                    #合成後の面から途中点を削除
                                    if common[0] in faces[tpi][tpj]:
                                        faces[tpi][tpj].remove(common[0])
                                        print("remove vertex:", common[0], faces[tpi][tpj])
                                    if common[0] in faces[tp2i][tp2j]:
                                        faces[tp2i][tp2j].remove(common[0])
                                        print("remove vertex:", common[0], faces[tp2i][tp2j])
                                    print("faces", faces)
                                    print("edges", edges)
                    for j in range(len(rmedges)):
                        for k in range(len(edges)):
                            if rmedges[j] in edges[k]:
                                edges[k].remove(rmedges[j])
                    for j in range(len(edges)):
                        if [] in edges:
                            edges.remove([])
                    for j in range(len(adedges)):
                        edges.insert(len(edges)-1, adedges[j])
                        print("insert edges", adedges[j])
                for i in range(len(edges)):
                    if [] in edges:
                        edges.remove([])
                print("edges", edges)
                #面を合体する際に不要になった辺を削除
                fl = 0
                for i in range(len(edges)):
                    for j in range(len(edges[i])):
                        if rmedge1 == edges[i][j][0]:
                            print("remove edges", edges[i][j])
                            edges[i].pop(j)
                            fl = 1
                            break
                    if fl== 1:
                        break
                for i in range(len(edges)):
                    if rmedge2 in edges[i]:
                        edges[i].remove(rmedge2)
                        print("remove edges", rmedge2)
                    if rmedge3 in edges[i]:
                        edges[i].remove(rmedge3)
                        print("remove edges", rmedge3)
                for i in range(len(edges)):
                    if [] in edges:
                        edges.remove([])
                print("edges", edges)
                #不要な面を削除
                print("faces",faces)
                print("remove faces", faces[unfold_faces[n^1][0]][unfold_faces[n^1][1]], faces[unfold_faces[n][0]][unfold_faces[n][1]])
                tpface1, tpface2 = faces[unfold_faces[n^1][0]][unfold_faces[n^1][1]], faces[unfold_faces[n][0]][unfold_faces[n][1]]
                for i in range(len(faces)):
                    if tpface1 in faces[i]:
                        faces_index[i].pop(faces[i].index(tpface1))
                        faces[i].remove(tpface1)
                    if tpface2 in faces[i]:
                        faces_index[i].pop(faces[i].index(tpface2))
                        faces[i].remove(tpface2)
                for i in range(len(faces)):
                    if [] in faces:
                        faces.remove([])
                    if [] in faces_index:
                        faces_index.remove([])
                print(faces)
                print("-------------tsubushi(square) end-------------------------")
                fold_kind = "四角形へのつぶし折り"
                return vertices, faces, edges, faces_index, face_order, 1, facemax, fold_kind, folding_edge

        #つぶし折り（三角形）
        #faces[unfold_faces[n^1][0]][unfold_faces[n^1][1]]が三角形、faces[unfold_faces[n][0]][unfold_faces[n][1]]が隣り合う面
        for n in range(2):
            if len(faces[unfold_faces[n^1][0]][unfold_faces[n^1][1]])==3:
                cont, vertices, perpendicular = has_perpendicular_as_edge(vertices, faces[unfold_faces[n][0]][unfold_faces[n][1]], faces[unfold_faces[n^1][0]][unfold_faces[n^1][1]])
                if cont:
                    if len(perpendicular) != 2:
                        return input_vertices, input_faces, input_edges, input_faces_index, input_face_order, 0, facemax, fold_kind, []
                    #頂点の末尾が垂線の足の座標
                    print("-------------tsubushi(triangle) start-------------------------")
                    print("---perpendicular---",perpendicular) #perpendicularは垂線の[〇,〇]

                    tpface1, tpface2 = faces[unfold_faces[n^1][0]][unfold_faces[n^1][1]], faces[unfold_faces[n][0]][unfold_faces[n][1]]
                    tmpi, tmpj, flag = 0,0,0
                    for i in range(len(edges)):
                        for j in range(len(edges[i])):
                            if edges[i][j][0][0] in faces[unfold_faces[n^1][0]][unfold_faces[n^1][1]] and edges[i][j][0][1] in faces[unfold_faces[n^1][0]][unfold_faces[n^1][1]] and edges[i][j][0] != fold_edges_list[l][0] and edges[i][j][1] != 'B':
                                tmpi, tmpj = i, j
                                flag = 1
                                break
                        if flag == 1:
                            break
                    #fold_edges_list[l][0]は折り目
                    #edges[tmpi][tmpj][0]は三角形のもう1つの折り目
                    #perpendicularは三角形の垂線(1つ)
                    #まずは点を移動
                    #隣り合ってる面との共有辺を垂線に対して対称移動
                    for vertice in fold_edges_list[l][0]:
                        if vertice not in moved:
                            vertices[vertice] = reflect_point_over_line(vertices[vertice], vertices[perpendicular[0]], vertices[perpendicular[1]])
                            print("move vertice", vertice, perpendicular)
                            moved.append(vertice)
                    #垂線をもう一方の辺に対して対称移動
                    for vertice in perpendicular:
                        if vertice not in moved:
                            vertices[vertice] = reflect_point_over_line(vertices[vertice], vertices[edges[tmpi][tmpj][0][0]], vertices[edges[tmpi][tmpj][0][1]])
                            print("move vertice", vertice, edges[tmpi][tmpj][0])
                            moved.append(vertice)
                    #次に面を合体
                    #三角形を垂線で分割 halftriangle1, halftriangle2
                    halftriangle1, halftriangle2 = [], []
                    for i in range(len(perpendicular)):
                        halftriangle1.append(perpendicular[i])
                        halftriangle2.append(perpendicular[i])
                    for i in range(len(fold_edges_list[l][0])):
                        halftriangle1.append(fold_edges_list[l][0][i])
                    for i in range(len(edges[tmpi][tmpj][0])):
                        halftriangle2.append(edges[tmpi][tmpj][0][i])
                    halftriangle1 = list(set(halftriangle1))
                    halftriangle2 = list(set(halftriangle2))
                    #折り辺表示用
                    folding_edge = []
                    for _ in range(len(perpendicular)):
                        folding_edge.append([vertices[perpendicular[0]], vertices[perpendicular[1]]])
                    for i in range(len(halftriangle1)):
                        folding_edge.append([vertices[halftriangle1[i-1]], vertices[halftriangle1[i]]])
                    for i in range(len(halftriangle2)):
                        folding_edge.append([vertices[halftriangle2[i-1]], vertices[halftriangle2[i]]])
                    for i in range(len(tpface2)):
                        folding_edge.append([vertices[tpface2[i-1]], vertices[tpface2[i]]])
                    #表面を合体させる
                    #三角形の隣の面の隣の面を探す
                    tmpedge = []
                    tpi, tpj = 0, 0
                    for i in range(len(edges)):
                        for j in range(len(edges[i])):
                            if edges[i][j][0][0] in faces[unfold_faces[n][0]][unfold_faces[n][1]] and edges[i][j][0][1] in faces[unfold_faces[n][0]][unfold_faces[n][1]] and edges[i][j][1]!='B' and edges[i][j][0]!=fold_edges_list[l][0]:
                                tmpedge.append(edges[i][j][0])
                    print(tmpedge)
                    if tmpedge == []:
                        return input_vertices, input_faces, input_edges, input_faces_index, input_face_order, 0, facemax, fold_kind, []
                    for i in range(len(faces)):
                        for j in range(len(faces[i])):
                            if tmpedge[0][0] in faces[i][j] and tmpedge[0][1] in faces[i][j] and faces[i][j] != faces[unfold_faces[n][0]][unfold_faces[n][1]]:
                                tpi, tpj = i, j
                    tmpface1 = copy.deepcopy(faces[tpi][tpj])
                    #tpface1は三角形、tpface2はhalftriangle1の隣、tmpface1はtpface2の隣、tmpface2はhalftriangle2の隣
                    print("merge faces",faces[tpi][tpj], faces[unfold_faces[n][0]][unfold_faces[n][1]], "edge:", tmpedge[0])
                    delete = faces_index[unfold_faces[n][0]][unfold_faces[n][1]]
                    face_order.remove(delete) #faceorder update　つぶし(三角形)
                    print("delete", delete)
                    faces[tpi][tpj] = merge_faces(faces[tpi][tpj], faces[unfold_faces[n][0]][unfold_faces[n][1]], tmpedge[0])
                    print("merge faces",faces[tpi][tpj], halftriangle1, "edge:", fold_edges_list[l][0])
                    delete = faces_index[unfold_faces[n^1][0]][unfold_faces[n^1][1]]
                    face_order.remove(delete) #faceorder update　つぶし(三角形)
                    print("delete", delete)
                    faces[tpi][tpj] = merge_faces(faces[tpi][tpj], halftriangle1, fold_edges_list[l][0])
                    print(faces[tpi][tpj])
                    rmedge1 = tmpedge[0]
                    rmedge2 = fold_edges_list[l]

                    #裏面を合体させる
                    #halftriangle2と隣り合っている面を探す
                    tp2i, tp2j = 0,0
                    for i in range(len(faces)):
                        for j in range(len(faces[i])):
                            if edges[tmpi][tmpj][0][0] in faces[i][j] and edges[tmpi][tmpj][0][1] in faces[i][j] and faces[i][j] != faces[unfold_faces[n^1][0]][unfold_faces[n^1][1]]:
                                tp2i, tp2j = i, j
                    tmpface2 = faces[tp2i][tp2j]
                    print("merge faces",faces[tp2i][tp2j], halftriangle2, "edge:", edges[tmpi][tmpj][0])
                    faces[tp2i][tp2j] = merge_faces(faces[tp2i][tp2j], halftriangle2, edges[tmpi][tmpj][0])
                    rmedge3 = edges[tmpi][tmpj]
                    print(faces[tp2i][tp2j])
                    print("faces", faces)
                    #次に辺をくっつける
                    #まずは垂線
                    perpendicular_link = []
                    for i in range(len(perpendicular)):#垂線の1点
                        if perpendicular[i] in tmpface1:
                            if tmpface1.index(perpendicular[i])+1 == len(tmpface1):
                                perpendicular_link.append([perpendicular[i], tmpface1[tmpface1.index(perpendicular[i])-1]])
                                perpendicular_link.append([perpendicular[i], tmpface1[0]])
                            else:
                                perpendicular_link.append([perpendicular[i], tmpface1[tmpface1.index(perpendicular[i])+1]])
                                perpendicular_link.append([perpendicular[i], tmpface1[tmpface1.index(perpendicular[i])-1]])
                    print("perpendicular_link", perpendicular_link)
                    fl = 0
                    for i in range(len(perpendicular_link)):
                        if parallel_check([vertices[perpendicular[0]],vertices[perpendicular[1]]], [vertices[perpendicular_link[i][0]],vertices[perpendicular_link[i][1]]]):
                            print("parallel_link_edges", perpendicular, perpendicular_link[i])
                            #つながる前の線を削除してつながった辺を追加 common[0]は途中点
                            print(edges)
                            li, common = lists_common(perpendicular, perpendicular_link[i])
                            order = 2
                            for j in range(len(edges)):
                                if common != []:
                                    type = 'U'
                                    for k in range(len(edges[j])):
                                        n = 0
                                        if [perpendicular_link[i][n], perpendicular_link[i][n^1]] == edges[j][k][0]:
                                            tp3j, order, type = j, n, edges[j][k][1]
                                        elif [perpendicular_link[i][n^1], perpendicular_link[i][n]] == edges[j][k][0]:
                                            tp3j, order, type = j, n^1, edges[j][k][1]
                                        if order == 0 or order ==1:
                                            edges[tp3j].remove(([perpendicular_link[i][order], perpendicular_link[i][order^1]],type))
                                            fl = 1
                                            break
                                    if fl == 1:
                                        break
                            edges.insert(len(edges)-1, [(li, type)])
                            print("insert edges", [(li, type)])
                            for j in range(len(edges)):
                                if [] in edges:
                                    edges.remove([])
                            #合成後の面から途中点を削除
                            if common[0] not in faces[tpi][tpj]:
                                print("error")
                                return input_vertices, input_faces, input_edges, input_faces_index, input_face_order, 0, facemax, fold_kind, []
                            faces[tpi][tpj].remove(common[0])
                            if common[0] not in faces[tp2i][tp2j]:
                                print("error")
                                return input_vertices, input_faces, input_edges, input_faces_index, input_face_order, 0, facemax, fold_kind, []
                            faces[tp2i][tp2j].remove(common[0])
                            print("remove vertex:", common[0], faces[tpi][tpj])
                            print("remove vertex:", common[0], faces[tp2i][tp2j])
                            print(faces)
                            print(edges)
                            break
                    if fl == 0:
                        #垂線が他の辺とくっつかなかったらedgesに追加
                        edges.insert(len(edges)-1, [(perpendicular, 'M')])
                        print("insert perpendicular edges", [(perpendicular, 'M')])

                    #垂線によって分けられた2辺を追加
                    points = []
                    print("tpface1", tpface1)#もとの三角形
                    for i in range(len(tpface1)):
                        if tpface1[i] not in perpendicular:
                            points.append(tpface1[i])
                        else:
                            point = tpface1[i]
                    if point == perpendicular[0]:
                        point = perpendicular[1]
                    elif point == perpendicular[1]:
                        point = perpendicular[0]
                    print("points:", points, "point:", point)
                    edgetype = 'B'
                    for i in range(len(edges)):
                        for j in range(len(edges[i])):
                            if edges[i][j][0][0] in points and edges[i][j][0][1] in points:
                                edgetype = edges[i][j][1]
                                print("remove edge", edges[i][j])
                                edges[i].pop(j)
                                break
                    edges.insert(len(edges)-1, [([point, points[0]], edgetype)])
                    print("insert perpendicular edges", [([point, points[0]], edgetype)])
                    edges.insert(len(edges)-1, [([point, points[1]], edgetype)])
                    print("insert perpendicular edges", [([point, points[1]], edgetype)])
                    print("edges", edges)

                    #あとは各辺について隣りあう面の辺かつもともと座標が同じだった辺同士がくっつく
                    #三角形の各辺をedgesから探して同じグループの辺の中で隣り合う面の辺があるかどうかを探す
                    fl=0
                    tp4j = 0
                    for i in range(len(tpface1)):
                        if tpface1[0] in points and tpface1[-1] in points:
                            tpface1.append(point)
                            break
                        else:
                            tpface1 = tpface1[1::] + tpface1[:1]
                    print(tpface1)
                    for i in range(len(tpface1)):
                        tpedge = [tpface1[i-1] , tpface1[i]]
                        print(tpedge)
                        for j in range(len(edges)):
                            fl = 0
                            for k in range(len(edges[j])):
                                if edges[j][k][0] == tpedge or edges[j][k][0] == tpedge[::-1]:
                                    tp4j=j
                                    fl=1
                                    tpedge_type = edges[j][k][1]
                                    break
                            if fl==1:
                                break
                        #edges[tp4j]に三角形の辺が存在
                        cnt = 0
                        rmedges = []
                        adedges = []
                        for j in range(len(edges)):
                            for k in range(len(edges[j])):
                        #for j in range(len(edges[tp4j])):
                                #tpedgeとedges[tp4j][j][0]がくっつく
                                if ((edges[j][k][0][0] in tpface2 and edges[j][k][0][1] in tpface2) or (edges[j][k][0][0] in tmpface2 and edges[j][k][0][1] in tmpface2)) and (set(tpedge) != set(edges[j][k][0])):
                                    if parallel_check([vertices[edges[j][k][0][0]],vertices[edges[j][k][0][1]]], [vertices[tpedge[0]],vertices[tpedge[1]]]) and (tpedge[0] in edges[j][k][0] or tpedge[1] in edges[j][k][0]):
                                        if ((distance(vertices[edges[j][k][0][0]], vertices[tpedge[0]]) > 1E-10) or (distance(vertices[edges[j][k][0][1]], vertices[tpedge[1]]) > 1E-10)) and ((distance(vertices[edges[j][k][0][0]], vertices[tpedge[1]]) > 1E-10) or (distance(vertices[edges[j][k][0][1]], vertices[tpedge[0]]) > 1E-10)):
                                            print("parallel:", tpedge, edges[j][k][0])
                                            #つながる前の線を削除してつながった辺を追加 common[0]は途中点
                                            li, common = lists_common(tpedge, edges[j][k][0])
                                            order = 2
                                            fl = 0
                                            rmedges.append(edges[j][k])
                                            type = 'U'
                                            for m in range(len(edges)):
                                                if common != []:
                                                    for n in range(len(edges[m])):
                                                        if tpedge == edges[m][n][0]:
                                                            tp3j, order, type = m, 0, edges[m][n][1]
                                                        elif tpedge[::-1] == edges[m][n][0]:
                                                            tp3j, order, type = m, 1, edges[m][n][1]
                                                        if order == 0 or order == 1:
                                                            if tpedge_type != type:
                                                                print("edge types are different.", tpedge_type, type)
                                                                return input_vertices, input_faces, input_edges, input_faces_index, input_face_order, 0, facemax, fold_kind, []
                                                            if order == 0:
                                                                rmedges.append((tpedge,type))
                                                            elif order == 1:
                                                                rmedges.append((tpedge[::-1],type))
                                                            fl = 1
                                                            break
                                                    if fl == 1:
                                                        break
                                            if type == 'U':
                                                print("type == 'U")
                                                return input_vertices, input_faces, input_edges, input_faces_index, input_face_order, 0, facemax, fold_kind, []
                                            adedges.append([(li, type)])

                                            #合成後の面から途中点を削除
                                            if common[0] in faces[tpi][tpj]:
                                                faces[tpi][tpj].remove(common[0])
                                                print("remove vertex:", common[0], faces[tpi][tpj])
                                            if common[0] in faces[tp2i][tp2j]:
                                                faces[tp2i][tp2j].remove(common[0])
                                                print("remove vertex:", common[0], faces[tp2i][tp2j])
                                            print("faces", faces)
                                            print("edges", edges)
                        for j in range(len(rmedges)):
                            for k in range(len(edges)):
                                if rmedges[j] in edges[k]:
                                    edges[k].remove(rmedges[j])
                        for j in range(len(edges)):
                            if [] in edges:
                                edges.remove([])
                        for j in range(len(adedges)):
                            edges.insert(len(edges)-1, adedges[j])
                            print("insert edges", adedges[j])

                    for i in range(len(tmpface1)):
                        tpedge = [tmpface1[i-1] , tmpface1[i]]
                        for j in range(len(edges)):
                            fl = 0
                            for k in range(len(edges[j])):
                                if edges[j][k][0] == tpedge or edges[j][k][0] == tpedge[::-1]:
                                    tp5j=j
                                    fl=1
                                    tpedge_type = edges[j][k][1]
                                    break
                            if fl==1:
                                break
                        if fl == 0:
                            continue
                        #edges[tp5j]にtpedgeが存在
                        #tpface2の辺とくっつくかどうか
                        cnt = 0
                        rmedges = []
                        adedges = []
                        for j in range(len(edges[tp5j])):
                            #tpedgeとedges[tp5j][j][0]がくっつく
                            if (edges[tp5j][j][0][0] in tpface2 and edges[tp5j][j][0][1] in tpface2) and (set(tpedge) != set(edges[tp5j][j][0])):
                                if parallel_check([vertices[edges[tp5j][j][0][0]],vertices[edges[tp5j][j][0][1]]], [vertices[tpedge[0]],vertices[tpedge[1]]]):
                                    print("parallel:", tpedge, edges[tp5j][j][0])
                                    #つながる前の線を削除してつながった辺を追加 common[0]は途中点
                                    li, common = lists_common(tpedge, edges[tp5j][j][0])
                                    order = 2
                                    fl = 0
                                    rmedges.append(edges[tp5j][j])
                                    type = 'U'
                                    print(common)
                                    for k in range(len(edges)):
                                        if common != []:
                                            for m in range(len(edges[k])):
                                                if tpedge == edges[k][m][0]:
                                                    tp3j, order, type = k, 0, edges[k][m][1]
                                                elif tpedge[::-1] == edges[k][m][0]:
                                                    tp3j, order, type = k, 1, edges[k][m][1]
                                                if order == 0 or order == 1:
                                                    if tpedge_type != type:
                                                        print("edge types are different.", tpedge_type, type)
                                                        return input_vertices, input_faces, input_edges, input_faces_index, input_face_order, 0, facemax, fold_kind, []
                                                    if order == 0:
                                                        rmedges.append((tpedge,type))
                                                    elif order == 1:
                                                        rmedges.append((tpedge[::-1],type))
                                                    fl = 1
                                                    break
                                            if fl == 1:
                                                break
                                    if type == 'U':
                                        print("type == 'U'")
                                        return input_vertices, input_faces, input_edges, input_faces_index, input_face_order, 0, facemax, fold_kind, []
                                    adedges.append([(li, type)])

                                    #合成後の面から途中点を削除
                                    if common[0] in faces[tpi][tpj]:
                                        faces[tpi][tpj].remove(common[0])
                                        print("remove vertex:", common[0], faces[tpi][tpj])
                                    if common[0] in faces[tp2i][tp2j]:
                                        faces[tp2i][tp2j].remove(common[0])
                                        print("remove vertex:", common[0], faces[tp2i][tp2j])
                                    print("faces", faces)
                                    print("edges", edges)
                        for j in range(len(rmedges)):
                            for k in range(len(edges)):
                                if rmedges[j] in edges[k]:
                                    edges[k].remove(rmedges[j])
                        for j in range(len(edges)):
                            if [] in edges:
                                edges.remove([])
                        for j in range(len(adedges)):
                            edges.insert(len(edges)-1, adedges[j])
                            print("insert edges", adedges[j])
                    for i in range(len(edges)):
                        if [] in edges:
                            edges.remove([])
                    print("edges", edges)
                    #面を合体する際に不要になった辺を削除
                    fl = 0
                    for i in range(len(edges)):
                        for j in range(len(edges[i])):
                            if rmedge1 == edges[i][j][0]:
                                print("remove edges", edges[i][j])
                                edges[i].pop(j)
                                fl = 1
                                break
                        if fl== 1:
                            break
                    for i in range(len(edges)):
                        if rmedge2 in edges[i]:
                            edges[i].remove(rmedge2)
                            print("remove edges", rmedge2)
                        if rmedge3 in edges[i]:
                            edges[i].remove(rmedge3)
                            print("remove edges", rmedge3)
                    for i in range(len(edges)):
                        if [] in edges:
                            edges.remove([])
                    print("edges", edges)
                    #不要な面を削除
                    print("faces",faces)
                    print("remove faces", faces[unfold_faces[n^1][0]][unfold_faces[n^1][1]], faces[unfold_faces[n][0]][unfold_faces[n][1]])
                    tpface1, tpface2 = faces[unfold_faces[n^1][0]][unfold_faces[n^1][1]], faces[unfold_faces[n][0]][unfold_faces[n][1]]
                    for i in range(len(faces)):
                        if tpface1 in faces[i]:
                            faces_index[i].pop(faces[i].index(tpface1))
                            faces[i].remove(tpface1)
                        if tpface2 in faces[i]:
                            faces_index[i].pop(faces[i].index(tpface2))
                            faces[i].remove(tpface2)
                    for i in range(len(faces)):
                        if [] in faces:
                            faces.remove([])
                        if [] in faces_index:
                            faces_index.remove([])
                    print("faces",faces)
                    print("-------------tsubushi(triangle) end-------------------------")
                    fold_kind = "三角形へのつぶし折り"
                    return vertices, faces, edges, faces_index, face_order, 1, facemax, fold_kind, folding_edge

        #普通の山折り谷折りの折り返し
        #fold_edges 折り辺の2点の座標保持
        print("------simple folding------")
        print("faces",faces)
        print("faces_index", faces_index)
        print(faces[unfold_faces[unfold_face_index^1][0]][unfold_faces[unfold_face_index^1][1]], faces[unfold_faces[unfold_face_index][0]][unfold_faces[unfold_face_index][1]])
        print("unfold_face_index", unfold_face_index)
        moved_point_in_face = copy.deepcopy(faces[unfold_faces[unfold_face_index][0]][unfold_faces[unfold_face_index][1]])
        #折り返す面の全部の点を折り辺に対して対称移動
        for vertice in moved_point_in_face:
            if vertice not in moved_vertices:
                vertices[vertice] = reflect_point_over_line(vertices[vertice], fold_edges[0], fold_edges[1])
                moved_vertices.append(vertice)
                print("move vertice", vertice)
        #折り辺上にある点を含めた折られる辺
        edgesonline, edgeslist = edges_on_line(fold_edges_list[l][0][0], fold_edges_list[l][0][1], edges, vertices, fold_edges_list)
        if fold_edges_list[l] not in edgesonline:
            edgesonline.append(fold_edges_list[l])
        if nextface_unique_edge == 1:
            edgesonline = [fold_edges_list[l]]
            print("edgeslist", edgeslist)
            for i in range(len(unique_edges)):
                for j in range(len(edgeslist)):
                    if unique_edges[i][0] in edgeslist[j] and not complete_overlap_edges(fold_edges_list[l][0], unique_edges[i][0], vertices):
                        edgesonline.append(unique_edges[i])
        if cntface == 1:
            edgesonline = [fold_edges_list[l]]
        print("edgesonline", edgesonline)
        #折り辺表示用
        folding_edge = []
        for i in range(len(edgesonline)):
            vertp = []
            for j in range(len(edgesonline[i][0])):
                vertp.append(vertices[edgesonline[i][0][j]])
            folding_edge.append(vertp)
        fl1 = 1
        for ed in edgesonline:
            print("ed", ed)
            print("edges", edges)
            unfold_faces=[]
            if l == 0:
                for i in range(len(faces)):
                    for j, face in enumerate(faces[i]):
                        if ed[0][0] in face and ed[0][1] in face:
                            unfold_faces.append([i,j])
                            ver = []
                            for k in range(len(face)):
                                ver.append(vertices[face[k]])
                            vers2.append(ver)
            else:
                #len(vers2)-1個の点の座標が一致している面を探す
                for i in range(len(faces)):
                    for j, face in enumerate(faces[i]):
                        ct = 0
                        for k in range(len(faces[i][j])):
                            for m in range(len(vers2)):
                                for n in range(len(vers2[m])):
                                    if distance(vertices[faces[i][j][k]], vers2[m][n]) < 1E-8:
                                        #　点の座標と、面の頂点index入れる　頂点同士比べて距離torlerance以内なものがあるか
                                        ct += 1
                        if ct >= len(vers2) - 1 and ed[0][0] in face and ed[0][1] in face and [i,j] not in unfold_faces:
                            unfold_faces.append([i,j])
            if len(unfold_faces) != 2:
                print("error: len(unfold_faces) != 2", ed)
                continue
            print(faces[unfold_faces[0][0]][unfold_faces[0][1]], faces[unfold_faces[1][0]][unfold_faces[1][1]])
            moved_point_in_face0 = copy.deepcopy(faces[unfold_faces[0][0]][unfold_faces[0][1]])
            moved_point_in_face1 = copy.deepcopy(faces[unfold_faces[1][0]][unfold_faces[1][1]])
            for i in range(len(ed[0])):
                moved_point_in_face0.remove(ed[0][i])
                moved_point_in_face1.remove(ed[0][i])
            len0, len1 = len(faces[unfold_faces[0][0]][unfold_faces[0][1]]), len(faces[unfold_faces[1][0]][unfold_faces[1][1]])
            print("len0", len0, "len1", len1)
            com0, com1 = 0, 0
            for i in range(len0):
                if faces[unfold_faces[0][0]][unfold_faces[0][1]][i] in moved_vertices:
                    com0 += 1
            for i in range(len1):
                if faces[unfold_faces[1][0]][unfold_faces[1][1]][i] in moved_vertices:
                    com1 += 1
            print("moved_vertices", moved_vertices)
            print("com0:", com0, "com1:", com1)
            face0_ver_bool, face1_ver_bool = False, False
            for i in range(len(moved_point_in_face0)):
                if moved_vertice_same_point(reflect_point_over_line(vertices[moved_point_in_face0[i]], fold_edges[0], fold_edges[1]), vertices):
                    face0_ver_bool = True
            for i in range(len(moved_point_in_face1)):
                if moved_vertice_same_point(reflect_point_over_line(vertices[moved_point_in_face1[i]], fold_edges[0], fold_edges[1]), vertices):
                    face1_ver_bool = True
            print("unique", unique_edges)
            nextface_unique = [0,0]
            nextface_unique_edge = 0
            if len(unique_edges) != 0:
                #もし隣の面がunique辺持ってたらそっちを折り返す
                for i in range(2):#faces[unfold_faces[i][0]][unfold_faces[i][1]]の隣の面を探す
                    nextfacesi = []
                    for j in range(len(faces[unfold_faces[i][0]][unfold_faces[i][1]])):
                        tped = [faces[unfold_faces[i][0]][unfold_faces[i][1]][j-1], faces[unfold_faces[i][0]][unfold_faces[i][1]][j]]
                        for k in range(len(faces)):
                            for m in range(len(faces[k])):
                                if tped[0] in faces[k][m] and tped[1] in faces[k][m] and faces[k][m] != faces[unfold_faces[0][0]][unfold_faces[0][1]] and faces[k][m] != faces[unfold_faces[1][0]][unfold_faces[1][1]]:
                                    nextfacesi = faces[k][m]
                                    print("nextfacesi", nextfacesi)
                        #faces[unfold_faces[i][0]][unfold_faces[i][1]]の隣の面がnextfacesi
                        if nextfacesi != []:
                            for k in range(len(unique_edges)):
                                if unique_edges[k][0][0] in nextfacesi and unique_edges[k][0][1] in nextfacesi:
                                    nextface_unique[i] = 1
            if nextface_unique[0] == 1 and nextface_unique[1] != 1:
                unfold_face_index = 0
                nextface_unique_edge = 1
                print("nextface_unique")
            elif nextface_unique[0] != 1 and nextface_unique[1] == 1:
                unfold_face_index = 1
                nextface_unique_edge = 1
                print("nextface_unique")
            else:
                if face0_ver_bool and not face1_ver_bool:
                    unfold_face_index = 0
                    print("moved_vertice_same_point")
                elif not face0_ver_bool and face1_ver_bool:
                    unfold_face_index = 1
                    print("moved_vertice_same_point")
                else:
                    if com0 > com1:
                        unfold_face_index = 0
                        print("com0 > com1")
                    elif com0 < com1:
                        unfold_face_index = 1
                        print("com0 < com1")
                    else:
                        if len0 < len1:
                            unfold_face_index = 0
                            print("len(faces)")
                        elif len0 > len1:
                            unfold_face_index = 1
                            print("len(faces)")
                        else:
                            if polygon_area(faces[unfold_faces[0][0]][unfold_faces[0][1]], vertices) < polygon_area(faces[unfold_faces[1][0]][unfold_faces[1][1]], vertices):
                                unfold_face_index = 0
                                print("polygon_area")
                            else:
                                unfold_face_index = 1
                                print("polygon_area")
            #折り線1つ解除した際に面を1つとして認識するようにする
            faces_for_edges = [[],[]]
            faces_for_edges[0] = copy.deepcopy(faces[unfold_faces[unfold_face_index][0]][unfold_faces[unfold_face_index][1]])
            faces_for_edges[1] =copy.deepcopy(faces[unfold_faces[unfold_face_index^1][0]][unfold_faces[unfold_face_index^1][1]])
            print("merge faces",faces[unfold_faces[unfold_face_index^1][0]][unfold_faces[unfold_face_index^1][1]], faces[unfold_faces[unfold_face_index][0]][unfold_faces[unfold_face_index][1]])
            print("edge:", ed)
            print(faces_index)
            delete = faces_index[unfold_faces[unfold_face_index][0]][unfold_faces[unfold_face_index][1]]
            print("faces", faces)
            print("faces_index", faces_index)
            print("face_order", face_order)
            face_order_face = []
            for i in range(len(face_order)):
                for j in range(len(faces_index)):
                    for k in range(len(faces_index[j])):
                        if faces_index[j][k]==face_order[i]:
                            face_order_face.append(faces[j][k])
            print("face_order_face", face_order_face)
            face_order.remove(delete) #faceorder update　山折り・谷折り
            print("delete", delete)
            print("face_order", face_order)
            faces[unfold_faces[unfold_face_index^1][0]][unfold_faces[unfold_face_index^1][1]] = merge_faces(faces[unfold_faces[unfold_face_index^1][0]][unfold_faces[unfold_face_index^1][1]], faces[unfold_faces[unfold_face_index][0]][unfold_faces[unfold_face_index][1]], ed[0])
            print("merged face", faces[unfold_faces[unfold_face_index^1][0]][unfold_faces[unfold_face_index^1][1]])
            #対称移動した後に辺がくっつく候補を全部挙げる
            #その中で平行のものをくっつける
            foldedges = copy.deepcopy(ed)
            errcnt = 0
            for i in range(len(foldedges)): #折り辺の1点 i=0or1
                print("edges", edges)
                #まずは途中点の特定
                link_edges = [[],[]] #index
                if faces_for_edges[0].index(foldedges[0][i])+1 == len(faces_for_edges[0]):
                    link_edges[0].append(foldedges[0][i])
                    link_edges[0].append(faces_for_edges[0][faces_for_edges[0].index(foldedges[0][i])-1])
                    link_edges[0].append(faces_for_edges[0][0])
                else:
                    link_edges[0].append(foldedges[0][i])
                    link_edges[0].append(faces_for_edges[0][faces_for_edges[0].index(foldedges[0][i])-1])
                    link_edges[0].append(faces_for_edges[0][faces_for_edges[0].index(foldedges[0][i])+1])
                if faces_for_edges[1].index(foldedges[0][i])+1 == len(faces_for_edges[1]):
                    link_edges[1].append(foldedges[0][i])
                    link_edges[1].append(faces_for_edges[1][faces_for_edges[1].index(foldedges[0][i])-1])
                    link_edges[1].append(faces_for_edges[1][0])
                else:
                    link_edges[1].append(foldedges[0][i])
                    link_edges[1].append(faces_for_edges[1][faces_for_edges[1].index(foldedges[0][i])-1])
                    link_edges[1].append(faces_for_edges[1][faces_for_edges[1].index(foldedges[0][i])+1])
                if foldedges[0][i^1] not in link_edges[0] or foldedges[0][i^1] not in link_edges[1]:
                    print("error: foldedges[0][i^1] not in link_edges", ed)
                    continue
                link_edges[0].remove(foldedges[0][i^1])
                link_edges[1].remove(foldedges[0][i^1])
                if link_edges[0]== link_edges[1]:
                    print("error: link_edges[0]== link_edges[1]", ed)
                    continue
                #ここまででくっつく可能性のある2辺を挙げた
                #この2辺の山折り谷折りが異なっていればこの折り辺は矛盾する折り方である
                cnt = 0
                tpj, tpk, tp2j, tp2k = 0,0,0,0
                tmpedgetype = 'U'
                for j in range(len(edges)):
                    for k in range(len(edges[j])):
                        if edges[j][k][0] == [link_edges[0][0],link_edges[0][1]] or edges[j][k][0] == [link_edges[0][1],link_edges[0][0]]:
                            tpj, tpk = j, k
                            cnt += 1
                        if edges[j][k][0] == [link_edges[1][0],link_edges[1][1]] or edges[j][k][0] == [link_edges[1][1],link_edges[1][0]]:
                            tp2j, tp2k = j, k
                            cnt += 1
                if cnt == 2:
                    if edges[tpj][tpk][1] != edges[tp2j][tp2k][1]:
                        print("error: edge types are different.", edges[tpj][tpk], edges[tp2j][tp2k], "edges: ",ed)
                        errcnt += 1
                        continue
                    tmpedgetype = edges[tpj][tpk][1]
                #この2辺が平行か確認し、平行なら途中点であり、2辺をくっつけ、点を合成後の面から削除
                #平行でなければ角であるため、スルー
                print("link_edges", link_edges)
                if parallel_check([vertices[link_edges[0][0]],vertices[link_edges[0][1]]], [vertices[link_edges[1][0]],vertices[link_edges[1][1]]]):
                    print("parallel:", link_edges)
                    #つながる前の線を削除してつながった辺を追加
                    li, common = lists_common(link_edges[0], link_edges[1])
                    cnt, fl = 0, 0
                    order, order2 = np.inf, np.inf
                    print("edges", edges)
                    for j in range(len(edges)):
                        if common != []:
                            type = 'U'
                            for k in range(len(edges[j])):
                                if [link_edges[0][0], link_edges[0][1]] == edges[j][k][0]:
                                    tmpj, order, type = j, 0, edges[j][k][1]
                                    cnt += 1
                                elif [link_edges[0][1], link_edges[0][0]] == edges[j][k][0]:
                                    tmpj, order, type = j, 1, edges[j][k][1]
                                    cnt += 1
                                if [link_edges[1][0], link_edges[1][1]] == edges[j][k][0]:
                                    tmp2j, order2, type = j, 0, edges[j][k][1]
                                    cnt += 1
                                elif [link_edges[1][1], link_edges[1][0]] == edges[j][k][0]:
                                    tmp2j, order2, type = j, 1, edges[j][k][1]
                                    cnt += 1
                                if cnt == 2:
                                    if order == 0:
                                        edges[tmpj].remove(([link_edges[0][0], link_edges[0][1]],type))
                                        print("edge remove", ([link_edges[0][0], link_edges[0][1]],type))
                                    elif order == 1:
                                        edges[tmpj].remove(([link_edges[0][1], link_edges[0][0]],type))
                                        print("edge remove", ([link_edges[0][1], link_edges[0][0]],type))
                                    if order2 == 0:
                                        edges[tmp2j].remove(([link_edges[1][0], link_edges[1][1]],type))
                                        print("edge remove", ([link_edges[1][0], link_edges[1][1]],type))
                                    elif order2 == 1:
                                        edges[tmp2j].remove(([link_edges[1][1], link_edges[1][0]],type))
                                        print("edge remove", ([link_edges[1][1], link_edges[1][0]],type))
                                    edges.insert(len(edges)-1, [(li, tmpedgetype)])
                                    print("insert edges", [(li, tmpedgetype)])
                                    fl = 1
                                    break
                            if fl == 1:
                                break
                    for j in range(len(edges)):
                        if [] in edges:
                            edges.remove([])
                    #合成後の面から途中点を削除
                    if len(faces) != 2 or (len(faces) == 2 and (abs(abs(fold_edges[i][0])-200.0) > 1E-10 or abs(abs(fold_edges[i][1])-200.0) > 1E-10)):
                        faces[unfold_faces[unfold_face_index^1][0]][unfold_faces[unfold_face_index^1][1]].remove(common[0])
                        print("remove vertex:", common[0], faces[unfold_faces[unfold_face_index^1][0]][unfold_faces[unfold_face_index^1][1]])
                        if len(faces[unfold_faces[unfold_face_index^1][0]][unfold_faces[unfold_face_index^1][1]]) <= 2:
                            print("remove face  len <= 2", faces[unfold_faces[unfold_face_index^1][0]][unfold_faces[unfold_face_index^1][1]])
                            faces[unfold_faces[unfold_face_index^1][0]].pop(unfold_faces[unfold_face_index^1][1])
                            faces_index[unfold_faces[unfold_face_index^1][0]].pop(unfold_faces[unfold_face_index^1][1])
                    fl1 = 0
                else:
                    print("not parallel", link_edges)
                    errcnt += 1
            #いらなくなった面を削除
            print("remove faces", faces_for_edges[0])
            if faces_for_edges[0] in faces[unfold_faces[unfold_face_index][0]]:
                print("faces_index remove", faces_index[unfold_faces[unfold_face_index][0]][faces[unfold_faces[unfold_face_index][0]].index(faces_for_edges[0])])
                faces_index[unfold_faces[unfold_face_index][0]].pop(faces[unfold_faces[unfold_face_index][0]].index(faces_for_edges[0]))
                faces[unfold_faces[unfold_face_index][0]].remove(faces_for_edges[0])
            print("faces", faces)
            print("faces_index", faces_index)
            print("errcnt", errcnt)
            if errcnt == 2:
                #折り辺に対して完全対称なら許す
                if is_symmetric(ed[0], vertices, edges) or nextface_unique_edge == 1 or cntface == 1:
                    fl1 = 0
                    fmax = 0
                    for i in range(len(faces)):
                        for j in range(len(faces[i])):
                            if len(faces[i][j]) > fmax:
                                fmax = len(faces[i][j])
                    facemax = fmax
            remove_edges.append(ed)
        cnt = 0
        for k in range(len(faces)):
            cnt += len(faces[k])
        if cnt == 1:
            print("cnt == 1")
            break
        if fl1 == 1:
            print("error -> return")
            return input_vertices, input_faces, input_edges, input_faces_index, input_face_order, 0, facemax, fold_kind, []
    #折り線をedgesから削除 同じ座標の辺をまとめて削除
    print("remove edgelist", remove_edges)
    for j in range(len(remove_edges)):
        for k in range(len(edges)):
            if remove_edges[j] in edges[k]:
                print("remove edge", remove_edges[j])
                edges[k].remove(remove_edges[j])
    for j in range(len(edges)):
        if [] in edges:
            edges.remove([])
    print("-------------------")
    print("faces",faces)
    print("faces_index", faces_index)
    for i in range(len(faces)):
        if [] in faces:
            if len(faces) == len(faces_index):
                faces_index.pop(faces.index([]))
            faces.remove([])
        if [] in faces_index:
            faces_index.remove([])
    print("edges",edges)
    print("faces",faces)
    print("faces_index", faces_index)
    fold_kind = "山折り・谷折り"
    return vertices, faces, edges, faces_index, face_order, 1, facemax, fold_kind, folding_edge

# dfsの再帰関数
def dfs_permutations(arr, path, visited, vertices, faces, edges, faces_index, face_order, imnum, return_vertices, return_faces, return_edges, return_faces_index, return_face_order, facemax, unique_edges):
    inputvertices = copy.deepcopy(vertices)
    inputfaces = copy.deepcopy(faces)
    inputedges = copy.deepcopy(edges)
    inputfaces_index = copy.deepcopy(faces_index)
    inputface_order = copy.deepcopy(face_order)
    print("input_edges",inputedges)
    print("input_faces", inputfaces)
    print("input_faces_index", inputfaces_index)
    imnum[0] += 1
    flag = 0
    path = []
    for i in range(len(arr)):
        vercnt = 0
        for j in range(len(visited)):
            if visited[j] == True:
                visited = visited[j+1::]
                break
        if len(arr) > len(visited):
            for j in range(len(arr)-len(visited)):
                visited.append(False)
        print(visited)
        if not visited[i]:
            visited[i] = True
            path.append(arr[i])
            vertices = copy.deepcopy(inputvertices)
            faces = copy.deepcopy(inputfaces)
            edges = copy.deepcopy(inputedges)
            faces_index = copy.deepcopy(inputfaces_index)
            face_order = copy.deepcopy(inputface_order)
            for j in range(len(edges)):
                if [] in edges:
                    edges.remove([])
            print("arr", arr)
            print("path", path)
            print("edges", edges)
            print("faces", faces)
            print("faces_index", faces_index)
            print("inputedges",inputedges)
            print("inputfaces", inputfaces)
            print("inputfaces_index", inputfaces_index)
            print("before unfold, i:", i)
            vertices, faces, edges, faces_index, face_order, flag, facemax, fold_kind, folding_edge = unfold(copy.deepcopy(vertices), copy.deepcopy(faces), copy.deepcopy(edges), copy.deepcopy(faces_index), copy.deepcopy(face_order), i, facemax, unique_edges)
            print("------------")
            print("after unfold edges ", edges)
            print("after unfold faces ", faces)
            print("after unfold faces_index ", faces_index)
            print("inputedges",inputedges)
            print("inputfaces", inputfaces)
            print("inputfaces_index", inputfaces_index)
            for j in range(len(faces)):
                for k in range(len(faces[j])):
                    if len(faces[j][k]) >= max(facemax+1, 5):
                        print("len(faces[j][k]) >= max(facemax+1, 5)", len(faces[j][k]), max(facemax+1, 5))
                        vercnt = 1
            print("vercnt:", vercnt, "flag:", flag)
            if vercnt != 1:
                if flag == 1:
                    print("plot-----", imnum[0], faces)
                    plot_2d(copy.deepcopy(vertices), copy.deepcopy(faces), copy.deepcopy(face_order), copy.deepcopy(faces_index), imnum, fold_kind, folding_edge)
                # edgesのgroupを再編成
                Blist = []
                for j in range(len(edges)):
                    for k in range(len(edges[j])):
                        if edges[j][k][1] == "B":
                            Blist.append((edges[j][k]))
                grouped_points = copy.deepcopy(group_points_again(vertices, faces))
                print("point group", grouped_points)
                print(edges)
                grouped_edges = copy.deepcopy(group_edges_again(edges, grouped_points)) #edges group create again
                grouped_edges.append(copy.deepcopy(Blist))
                grouped_faces, faces_index = copy.deepcopy(group_faces_again(faces, grouped_points, faces_index)) #faces group create again
                if flag == 1:
                    print("----------------------------前---------------------------------", faces)
                    vertices, faces, edges, faces_index, face_order, return_vertices, return_faces, return_edges, return_faces_index, return_face_order, facemax = dfs_permutations(copy.deepcopy(grouped_edges), path, visited, copy.deepcopy(vertices), copy.deepcopy(grouped_faces), copy.deepcopy(grouped_edges), copy.deepcopy(faces_index), copy.deepcopy(face_order), imnum, copy.deepcopy(inputvertices), copy.deepcopy(inputfaces), copy.deepcopy(inputedges), copy.deepcopy(inputfaces_index), copy.deepcopy(inputface_order), facemax, unique_edges)
                    print("----------------------------後---------------------------------", faces, " , " , return_faces)
            path.pop()
            visited[i] = False
            print(len(arr), i+1)
            if len(arr) == i+1 or vercnt == 1:
                print("faces", faces)
                print("return_faces", return_faces)
                vertices = copy.deepcopy(inputvertices)
                faces = copy.deepcopy(inputfaces)
                edges = copy.deepcopy(inputedges)
                faces_index = copy.deepcopy(inputfaces_index)
                face_order = copy.deepcopy(inputface_order)
                print("edges",edges)
                print("faces", faces)
                print("faces_index", faces_index)
                if vercnt == 1:
                    print("len(faces[j][k]) >= facemax")
                    continue
                else:
                #全部エラーで一つ前に戻る
                    print("all error return")
                    #sys.exit()
                    output_folder = 'howtofold'
                    image_files = [f for f in os.listdir(output_folder) if f.endswith('.png')]
                    for image_file in image_files:
                        if extract_number(image_file) == imnum[0]-1:
                            file_path = os.path.join(output_folder, image_file)
                            if os.path.isfile(file_path) and (file_path != r"howtofold\0完成形.png"):
                                print("remove plot", file_path)
                                os.remove(file_path)
                                imnum[0] -= 1
                            break
    return return_vertices, return_faces, return_edges, return_faces_index, return_face_order, return_vertices, return_faces, return_edges, return_faces_index, return_face_order, facemax

input_opx = 'pig.opx' #入力する展開図

# opxファイルからfoldファイルを生成
if os.path.exists('goal.fold'):
    os.remove('goal.fold')
cmd = f"java -jar ./oripa-1.73.jar --fold goal.fold {input_opx}"
subprocess.call(cmd.split())
# foldファイルを読み込む
fold_file_path = 'goal.fold'
fold_data = load_fold_file(fold_file_path)

# 保存するフォルダの作成（もしくは中身を空にする）
output_folder = 'howtofold'
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
flag = 0
faces_vertices = copy.deepcopy(fold_data['faces_vertices'])
faces_index = [i for i in range(len(faces_vertices))]
edges_vertices = copy.deepcopy(fold_data['edges_vertices'])
edges_assignment = copy.deepcopy(fold_data['edges_assignment'])
vertices_coords = copy.deepcopy(np.array(fold_data['vertices_coords']))
vertices_grouped_list = copy.deepcopy(group_points(vertices_coords))
print("vertices_grouped_list", vertices_grouped_list)
Blist = []
for j, edge in enumerate(copy.deepcopy(fold_data['edges_vertices'])):
    if edges_assignment[j] == "B":
        Blist.append((edge,'B'))
grouped_edges = copy.deepcopy(group_edges(edges_vertices, vertices_grouped_list, edges_assignment))
unique_edges = edges_of_unique_faces(faces_vertices, grouped_edges)
print("unique_edges", unique_edges)
print("--sort--")
#grouped_edgesを辺上に点が多い順に並べる
pointgroupcounts, pointcounts, result = [], [], []
for i in range(len(grouped_edges)):
    pointgroupcounts.append(count_points_group_on_line(grouped_edges[i][0][0][0], grouped_edges[i][0][0][1], vertices_coords, vertices_grouped_list))
grouped_edges, pointgroupcounts = sort_two_lists(grouped_edges, pointgroupcounts)
for i in range(len(grouped_edges)):
    pointcounts.append(count_points_on_line(copy.deepcopy(grouped_edges[i][0][0][0]), copy.deepcopy(grouped_edges[i][0][0][1]), vertices_coords))
data = list(zip(pointgroupcounts, grouped_edges, pointcounts))
# pointgroupcountsでグループ化し、pointcountsの降順で並べ替える
for key, group in groupby(data, key=lambda x: x[0]):
    sorted_group = sorted(group, key=lambda x: x[2], reverse=True)
    result.extend(sorted_group)
grouped_edges = [x[1] for x in result]
grouped_edges = group_edges_arrange(grouped_edges, unique_edges, vertices_coords, faces_vertices)
grouped_edges.append(copy.deepcopy(Blist))
print("grouped_edges", grouped_edges)
print("faces_vertices", faces_vertices)
print("faceorder computing")
if 'file_frames' in fold_data: #all_foldファイル
    print(len(fold_data['file_frames']),"pattern")
    face_order = fold_input.decide_model(fold_file_path)
else: #single_foldファイル
    faceorders = copy.deepcopy(fold_data['faceOrders'])
    vertices = vertices_coords.tolist()
    fold_data = faceorder.create_fold_data(copy.deepcopy(vertices), copy.deepcopy(faces_vertices), copy.deepcopy(faceorders))
    face_order = faceorder.make_faceorder(fold_data)
facemax = 0
for i in range(len(faces_vertices)):
    if facemax < len(faces_vertices[i]):
        facemax = len(faces_vertices[i])
grouped_faces, faces_index = group_faces(copy.deepcopy(faces_vertices), copy.deepcopy(vertices_grouped_list), copy.deepcopy(faces_index), copy.deepcopy(face_order))
print("grouped_faces", grouped_faces)
print("faces_index", faces_index)
print("face_order", face_order)
visited = [False] * len(grouped_edges)
arr = copy.deepcopy(grouped_edges)
print("----------------------------前---------------------------------", grouped_faces)
print("plot first")
start_time = time.time()
plot_2d(copy.deepcopy(vertices_coords), copy.deepcopy(grouped_faces), copy.deepcopy(face_order), copy.deepcopy(faces_index), [0], "完成形", [])
dfs_permutations(arr, [], visited, vertices_coords, grouped_faces, grouped_edges, faces_index, face_order, [0], vertices_coords, grouped_faces, grouped_edges, faces_index, face_order, facemax, unique_edges)

'''
あとやること

　卒論の展開図太くする？


'''
