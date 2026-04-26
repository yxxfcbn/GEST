
import os
import time
import sys
import math
import glob
import argparse


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import networkx as nx
from pyscipopt import Model, quicksum

try:
    from scipy.spatial import Delaunay
except Exception:
    Delaunay = None

from tsp_train import build_graph_data, TSP_EdgeGNN

def read_tsp_coords_file(filename):

    coords = []
    start = False
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not start:
                if line == "NODE_COORD_SECTION":
                    start = True
                continue

            if line == "EOF":
                break

            parts = line.split()
            if len(parts) >= 3:
                x = float(parts[1])
                y = float(parts[2])
                coords.append((x, y))

    return coords


def get_full_graph_edges_and_costs(raw_coords):

    n = len(raw_coords)
    V = list(range(n))
    edges = []
    c = {}
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi = raw_coords[i][0], raw_coords[i][1]
            xj, yj = raw_coords[j][0], raw_coords[j][1]
            d = math.hypot(xi - xj, yi - yj)
            edges.append((i, j))
            c[i, j] = d
    return V, edges, c


def get_delaunay_edges_and_costs(raw_coords):

    n = len(raw_coords)
    V = list(range(n))
    if n < 2:
        return V, [], {}
    if n == 2:
        i, j = 0, 1
        return V, [(i, j)], {(i, j): math.hypot(raw_coords[i][0] - raw_coords[j][0], raw_coords[i][1] - raw_coords[j][1])}


    tri = Delaunay(raw_coords)
    edge_set = set()
    for simplex in tri.simplices:
        a, b, c = map(int, simplex)
        for u, v in ((a, b), (b, c), (a, c)):
            edge_set.add((u, v) if u < v else (v, u))

    edges = sorted(edge_set)
    costs = {}
    for i, j in edges:
        xi, yi = raw_coords[i]
        xj, yj = raw_coords[j]
        costs[(i, j)] = math.hypot(xi - xj, yi - yj)
    return V, edges, costs


def get_sparsified_edges_and_costs(data, raw_coords, model, device, threshold=0.5, topk=None):

    data = data.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_attr)
        probs = torch.sigmoid(logits)

    n = data.x.size(0)
    V = list(range(n))
    row, col = data.edge_index[0], data.edge_index[1]


    edge_info = {}
    for k in range(row.size(0)):
        u, v = row[k].item(), col[k].item()
        i, j = min(u, v), max(u, v)
        p = probs[k].item()
        if (i, j) not in edge_info:
            edge_info[(i, j)] = p
        else:
            edge_info[(i, j)] = max(edge_info[(i, j)], p)


    candidate = [(i, j, p) for (i, j), p in edge_info.items() if p > threshold]


    if topk is None:
        edges = [(i, j) for i, j, _p in candidate]
        c = {}
        for i, j in edges:
            xi, yi = raw_coords[i][0], raw_coords[i][1]
            xj, yj = raw_coords[j][0], raw_coords[j][1]
            c[i, j] = math.hypot(xi - xj, yi - yj)
        return V, edges, c

    topk = max(1, int(topk))
    incident = {u: [] for u in range(n)}
    for i, j, p in candidate:
        incident[i].append((p, i, j))
        incident[j].append((p, i, j))

    allow = {u: set() for u in range(n)}
    for u in range(n):
        incident[u].sort(key=lambda x: x[0], reverse=True)
        for _, i, j in incident[u][:topk]:
            allow[u].add((i, j))

    edges = []
    c = {}
    for i, j, _p in candidate:
        if (i, j) in allow[i] or (i, j) in allow[j]:
            edges.append((i, j))
            xi, yi = raw_coords[i][0], raw_coords[i][1]
            xj, yj = raw_coords[j][0], raw_coords[j][1]
            c[i, j] = math.hypot(xi - xj, yi - yj)

    return V, edges, c


def christofides_tour_edges(raw_coords):

    n = len(raw_coords)
    if n <= 1:
        return set()

    def dist(i, j):
        xi, yi = raw_coords[i]
        xj, yj = raw_coords[j]
        return math.hypot(xi - xj, yi - yj)

    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            graph.add_edge(i, j, weight=dist(i, j))

    mst = nx.minimum_spanning_tree(graph, weight="weight")
    odd = [v for v, d in mst.degree() if d % 2 == 1]
    if not odd:
        return {(min(i, (i + 1) % n), max(i, (i + 1) % n)) for i in range(n)}

    matching_graph = nx.Graph()
    matching_graph.add_nodes_from(odd)
    for a in range(len(odd)):
        for b in range(a + 1, len(odd)):
            u = odd[a]
            v = odd[b]
            matching_graph.add_edge(u, v, weight=dist(u, v))

    matching = nx.algorithms.matching.min_weight_matching(matching_graph, weight="weight")
    if len(matching) * 2 != len(odd):
        raise RuntimeError("Christofides failed")

    multi = nx.MultiGraph()
    multi.add_nodes_from(range(n))
    multi.add_edges_from(mst.edges())
    for u, v in matching:
        multi.add_edge(u, v)

    euler = list(nx.eulerian_circuit(multi, source=0))
    visited = set()
    tour = []
    for u, _ in euler:
        if u not in visited:
            visited.add(u)
            tour.append(u)
    for v in range(n):
        if v not in visited:
            tour.append(v)

    edges = set()
    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]
        edges.add((min(a, b), max(a, b)))
    return edges


def add_missing_costs(edges_set, costs, raw_coords):
    for i, j in edges_set:
        if (i, j) not in costs:
            xi, yi = raw_coords[i]
            xj, yj = raw_coords[j]
            costs[(i, j)] = math.hypot(xi - xj, yi - yj)


def solve_one_instance(tsp_path, model, device, args):
    raw_coords = read_tsp_coords_file(tsp_path)


    data = build_graph_data(
        raw_coords,
        labels_path=None,
        coarse_K=args.coarse_K,
        coarse_use_mst=True,

        enable_first_round_coarse=not args.no_first_round_coarse,
    )
    n = data.x.size(0)
    print("use(LKCChard): %s， nodes: %d" % (os.path.basename(tsp_path), n))


    obj_full = None
    time_full = None

    threshold = float(args.p)
    V_sp, edges_sp, c_sp = get_sparsified_edges_and_costs(
        data, raw_coords, model, device, threshold=threshold, topk=args.topk
    )
    print("  GNN sparse graph: %d (p=%.2f, topk=%s, coarse_K=%d)" %
          (len(edges_sp), threshold, str(args.topk), args.coarse_K))

    _, delaunay_edges, _ = get_delaunay_edges_and_costs(raw_coords)
    add_missing_costs(delaunay_edges, c_sp, raw_coords)
    print("  Delaunay edges: %d" % len(delaunay_edges))


    christ_edges = christofides_tour_edges(raw_coords)
    add_missing_costs(christ_edges, c_sp, raw_coords)
    edges_scip = set(edges_sp) | set(delaunay_edges) | set(christ_edges)
    edges_scip_list = sorted(edges_scip)
    print("  Christofides : %d" % len(christ_edges))
    print("  SCIP edges: %d" % len(edges_scip_list))

    t0 = time.perf_counter()
    obj_sp, _ = solve_tsp_sparse(V_sp, edges_scip_list, c_sp)
    time_sp = time.perf_counter() - t0

    instance_id = os.path.splitext(os.path.basename(tsp_path))[0]
    if obj_sp is None:
        print("  infeasible\n")
        return {"id": instance_id, "infeasible": True}

    print("  sparse edges: obj=%.6f, time=%.3fs\n" % (obj_sp, time_sp))

    return {
        "id": instance_id,
        "infeasible": False,
        "obj_sp": obj_sp,
        "time_sp": time_sp,
        "obj_full": obj_full,
        "time_full": time_full,
        "edges_sp": len(edges_sp),
        "edges_scip": len(edges_scip_list),
    }


def _scip_is_infeasible(model):
    st = str(model.getStatus()).lower()
    return st == "infeasible"


def solve_tsp_sparse(V, edges, c):

    adj = {i: [] for i in V}
    for (i, j) in edges:
        adj[i].append((j, (i, j)))
        adj[j].append((i, (i, j)))


    for i in V:
        if len(adj[i]) < 2:
            return None, None

    model = Model("tsp_sparse")
    model.hideOutput()

    x = {}
    for (i, j) in edges:
        x[i, j] = model.addVar(ub=1, name="x(%s,%s)" % (i, j))


    for i in V:
        model.addCons(
            quicksum(
                x[e] for (_, e) in adj[i]
            ) == 2,
            "Degree(%s)" % i
        )

    model.setObjective(
        quicksum(c[i, j] * x[i, j] for (i, j) in edges),
        "minimize"
    )

    EPS = 1.e-6
    is_mip = False
    sol_edges = []
    while True:
        model.optimize()
        if _scip_is_infeasible(model):
            return None, None

        sol_edges = []
        for (i, j) in edges:
            if model.getVal(x[i, j]) > EPS:
                sol_edges.append((i, j))

        G = nx.Graph()
        G.add_edges_from(sol_edges)
        components = list(nx.connected_components(G))

        if len(components) == 1:
            if is_mip:
                break

            model.freeTransform()
            for (i, j) in edges:
                model.chgVarType(x[i, j], "B")
            is_mip = True
            continue


        model.freeTransform()
        for S in components:
            in_S = [(i, j) for (i, j) in edges if i in S and j in S]
            if in_S:
                model.addCons(
                    quicksum(x[i, j] for (i, j) in in_S) <= len(S) - 1,
                    "Subtour(%s)" % (list(S)[:3],)
                )
        print("  add %d subtour cut" % len(components))

    if _scip_is_infeasible(model):
        return None, None
    try:
        obj = model.getObjVal()
    except Exception:
        return None, None
    return obj, sol_edges


def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    TSP_ROOT = os.path.join(SCRIPT_DIR, "tsp")
    CLKhard_COORD_DIR = os.path.join(TSP_ROOT, "CLKeasy")


    model_candidate = os.path.join(SCRIPT_DIR, "tsp_gnn_model.pt")

    MODEL_PATH = model_candidate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, default=None, help=" LKCChard ，如 000/001/010 ...")
    parser.add_argument("--coarse_K", type=int, default=15, help=" KNN ")
    parser.add_argument("--p", type=float, default=0.5, help="GNNp，")
    parser.add_argument("--topk", type=int, default=None, help="noTOPK")
    parser.add_argument("--no_first_round_coarse", action="store_true", help="noDelaunay+MST")
    parser.add_argument("--no_full", action="store_true", help="no complete")
    args = parser.parse_args()


    model = TSP_EdgeGNN(node_in_dim=2, edge_in_dim=4, hidden_dim=64).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    if args.id is not None:
        tsp_path = os.path.join(CLKhard_COORD_DIR, f"{args.id}.tsp")

        res = solve_one_instance(tsp_path, model, device, args)
        if res and res.get("infeasible"):
            print("infeasible")
        return


    all_files = sorted(glob.glob(os.path.join(CLKhard_COORD_DIR, "*.tsp")))

    train_n = int(len(all_files) * 0)
    test_files = all_files[train_n:]
    print(" CLKhard : %d （all %d，test %d）\n" %
          (len(test_files), len(all_files), train_n))

    results = []
    infeasible_ids = []
    for p in test_files:
        res = solve_one_instance(p, model, device, args)
        if res is None:
            continue
        if res.get("infeasible"):
            infeasible_ids.append(res["id"])
        else:
            results.append(res)

    print("\n" + "-" * 60)
    print("sparse infeasible: %d" % len(infeasible_ids))
    if infeasible_ids:
        print("infeasible id : %s" % ", ".join(infeasible_ids))

    if not results:
        print("infeasible")
        return

    avg_obj_sp = sum(r["obj_sp"] for r in results) / len(results)
    avg_time_sp = sum(r["time_sp"] for r in results) / len(results)
    avg_edges_sp = sum(r["edges_sp"] for r in results) / len(results)
    avg_edges_scip = sum(r["edges_scip"] for r in results) / len(results)

    print("\n" + "=" * 60)
    print("CLKhard index")
    print("=" * 60)
    print("feasibile: %d  infeasible: %d  total: %d" %
          (len(results), len(infeasible_ids), len(results) + len(infeasible_ids)))
    print("sparse edges: %.2f" % avg_edges_sp)
    print("sparse edges and 3/2: %.2f" % avg_edges_scip)
    print("opt sparse: %.6f" % avg_obj_sp)
    print("time sparse: %.3fs" % avg_time_sp)

    full_ok = [r for r in results if r["obj_full"] is not None and r["time_full"] is not None]
    if full_ok:
        avg_obj_full = sum(r["obj_full"] for r in full_ok) / len(full_ok)
        avg_time_full = sum(r["time_full"] for r in full_ok) / len(full_ok)
        avg_gap = sum(r["obj_sp"] - r["obj_full"] for r in full_ok) / len(full_ok)
        print("opt full: %.6f" % avg_obj_full)
        print("time full: %.3fs" % avg_time_full)
        if avg_time_full > 0:
            print("time speedup %.2f%%" % (100.0 * avg_time_sp / avg_time_full))
        print("opt gap: %.6f" % avg_gap)


if __name__ == "__main__":
    main()
