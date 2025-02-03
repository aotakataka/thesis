import numpy as np

EPS = 1E-6

def unit(vector):
    norm = np.linalg.norm(vector)
    if norm < EPS:
        return None
    return vector / norm

def parallel(a, b):
    ua = unit(a)
    ub = unit(b)
    if ua is None or ub is None:
        return None
    return 1 - abs(np.dot(ua, ub)) < EPS

def polygon_normal(points):
    cross_products = [
        np.cross(points[i], points[(i + 1) % len(points)])
        for i in range(len(points))
    ]
    normal = np.sum(cross_products, axis=0)
    return unit(normal)

def topological_sort(vs):
    def visit(v, list):
        v["visited"] = True
        for u in v["children"]:
            if not u["visited"]:
                u["parent"] = v
                list = visit(u, list)
        return list + [v]

    for v in vs:
        v["visited"] = False
        v["parent"] = None
    list = []
    for v in vs:
        if not v["visited"]:
            list = visit(v, list)
    return list

def faceAbove(f1, f2, n):
    ord = f1['ord'].get(f"f{f2['i']}")
    if ord is not None:
        return np.dot(f2['n'], n) * ord < 0
    return None

def order_faces(model):
    faces = model["fs"]
    for i, f1 in enumerate(faces):
        for j, f2 in enumerate(faces):
            if i >= j:
                continue
            # [0,0,-1]方向から見て、2つの面 f1 と f2 の間で、どちらがもう一方の上にあるかを判定
            f1_above = faceAbove(f1, f2, [0,0,-1])
            if f1_above is not None:
                # f1_above が真の場合、f1 が上。偽の場合、f2 が上。
                p, c = (f1, f2) if f1_above else (f2, f1)
                p["children"].append(c)
    model["fs"] = topological_sort(model["fs"])

def make_model(fold):
    m = {
        "vs": None,
        "fs": None,
    }
    m["vs"] = [{"i": i, "cs": cs} for i, cs in enumerate(fold["vertices_coords"])]
    for v in m["vs"]:
        if len(v["cs"]) == 2:
            v["cs"].append(0)
    m["fs"] = []
    for i, vs in enumerate(fold["faces_vertices"]):
        face_vertices = [m["vs"][v] for v in vs]
        m["fs"].append({
            "i": i,
            "vs": face_vertices,
            "n": polygon_normal([v["cs"] for v in face_vertices]), #面の法線ベクトル計算
            "ord": {},
            "visited": False,
            "children": [],
            "parent": []
        })
    if "faceOrders" in fold:
        for f1, f2, o in fold["faceOrders"]:
            if o != 0:
                if parallel(m["fs"][f1]["n"], m["fs"][f2]["n"]):
                    m["fs"][f1]["ord"][f"f{f2}"] = o
    return m

def create_fold_data(vertices_coords, faces_vertices, faceOrders):
    fold_data = {
        "vertices_coords": vertices_coords,
        "faces_vertices": faces_vertices,
        "faceOrders": faceOrders,
    }
    return fold_data

def make_faceorder(fold_data):
    order = []
    model = make_model(fold_data)
    order_faces(model)
    for face in model["fs"]:
        order.append(face["i"])
    return order