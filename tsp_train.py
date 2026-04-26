from pickle import FALSE

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from torch_geometric.loader import DataLoader
import networkx as nx
import math
import os
import glob
import time
try:
    from scipy.spatial import Delaunay
except Exception:
    Delaunay = None



def calculate_imst_weights(n, norm_coords):

    if n <= 1:
        return {}


    G = nx.complete_graph(n)


    for u, v in G.edges():
        dist = math.hypot(norm_coords[u][0] - norm_coords[v][0],
                          norm_coords[u][1] - norm_coords[v][1])
        G.edges[u, v]['weight'] = dist

    imst_weights = {}
    rounds = math.ceil(math.log2(n)) if n > 1 else 1
    current_G = G.copy()

    for k in range(1, rounds + 1):
        if current_G.number_of_edges() == 0:
            break

        mst_edges = list(nx.minimum_spanning_edges(current_G, weight='weight', data=False))
        edges_to_remove = []
        for u, v in mst_edges:
            u_s, v_s = min(u, v), max(u, v)
            if (u_s, v_s) not in imst_weights:
                imst_weights[(u_s, v_s)] = 1.0 / k
            edges_to_remove.append((u, v))

        current_G.remove_edges_from(edges_to_remove)

    return imst_weights



def read_tsp_file(filename):
    coords = []
    with open(filename, "r") as f:
        lines = f.readlines()

    start = False
    for line in lines:
        line = line.strip()

        if line == "NODE_COORD_SECTION":
            start = True
            continue


        if line == "EOF":
            break

        if start:
            parts = line.split()
            if len(parts) >= 3:

                x = float(parts[1])
                y = float(parts[2])
                coords.append((x, y))

    return coords


def normalize_coords(coords):
    if not coords: return []
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    range_x = max_x - min_x if max_x != min_x else 1.0
    range_y = max_y - min_y if max_y != min_y else 1.0
    return [((x - min_x) / range_x, (y - min_y) / range_y) for x, y in coords]


def build_graph_data(
        coords,
        labels_path=None,
        k=5,
        coarse_K=None,
        coarse_use_mst=False,
        enable_first_round_coarse=False,
        use_imst_feature=True,
        use_knn_feature=True,
        use_q_feature=True,
):
    n = len(coords)

    if enable_first_round_coarse and coarse_K is not None:
        k = coarse_K
    norm_coords = normalize_coords(coords)
    x_tensor = torch.tensor(norm_coords, dtype=torch.float)


    knn_edge_set = set()
    min_out_dists = [float("inf")] * n
    for i in range(n):
        dists = []
        xi, yi = norm_coords[i]
        for j in range(n):
            if i == j:
                continue
            xj, yj = norm_coords[j]
            d = math.hypot(xi - xj, yi - yj)
            dists.append((d, j))
        dists.sort()

        if len(dists) > 0:
            min_out_dists[i] = dists[0][0]
        else:
            min_out_dists[i] = 0.0

        for _, j in dists[:k]:
            u, v = min(i, j), max(i, j)
            knn_edge_set.add((u, v))


    coarse_edge_set = None

    if enable_first_round_coarse and coarse_K is not None:
        coarse_edge_set = set()

        if Delaunay is not None and n >= 3:
            try:
                tri = Delaunay(norm_coords)
                for simplex in tri.simplices:
                    a, b, c = int(simplex[0]), int(simplex[1]), int(simplex[2])
                    for u, v in ((a, b), (b, c), (a, c)):
                        i, j = (u, v) if u < v else (v, u)
                        coarse_edge_set.add((i, j))
            except Exception:

                pass

        if coarse_use_mst and n > 1:

            in_mst = [False] * n
            key = [float("inf")] * n
            parent = [-1] * n
            key[0] = 0.0
            for _ in range(n):
                u = -1
                best = float("inf")
                for i in range(n):
                    if (not in_mst[i]) and key[i] < best:
                        best = key[i]
                        u = i
                if u < 0:
                    break
                in_mst[u] = True
                ux, uy = norm_coords[u]
                for v in range(n):
                    if in_mst[v] or v == u:
                        continue
                    vx, vy = norm_coords[v]
                    w = math.hypot(ux - vx, uy - vy)
                    if w < key[v]:
                        key[v] = w
                        parent[v] = u
            for v in range(1, n):
                u = parent[v]
                if u >= 0:
                    a, b = (u, v) if u < v else (v, u)
                    coarse_edge_set.add((a, b))


    imst_weights_map = calculate_imst_weights(n, norm_coords)



    edge_labels = []
    if labels_path and os.path.exists(labels_path):
        with open(labels_path, "r") as f:
            for line in f:
                edge_labels.append(float(line.strip()))

    expected_undirected = n * (n - 1) // 2
    if len(edge_labels) != expected_undirected:

        pass
    full_real_pos_directed = float(sum(1.0 for v in edge_labels if v >= 0.5) * 2.0)
    full_total_directed = float(n * (n - 1))

    def label_index(u, v):
        # u < v
        # edges before row u: sum_{i=0}^{u-1} (n - i - 1) = u*(n-1) - u*(u-1)/2
        before = u * (n - 1) - (u * (u - 1)) // 2
        return before + (v - u - 1)

    edges_temp = []

    for u in range(n):
        for v in range(u + 1, n):
            if coarse_edge_set is not None and (u, v) not in coarse_edge_set:
                continue
            dist = math.hypot(norm_coords[u][0] - norm_coords[v][0],
                              norm_coords[u][1] - norm_coords[v][1])
            imst_w = imst_weights_map.get((u, v), 0.0) if use_imst_feature else 0.0
            is_knn = 1.0 if (use_knn_feature and (u, v) in knn_edge_set) else 0.0

            idx = label_index(u, v)
            label = edge_labels[idx] if idx < len(edge_labels) else 0.0
            edges_temp.append((u, v, dist, imst_w, is_knn, label))


    final_src, final_dst = [], []
    final_attr_list = []
    final_y = []

    for (u, v, dist, imst_w, is_knn, label) in edges_temp:

        min_dist_u = min_out_dists[u]
        q_uv = ((1.0 + dist) / (1.0 + min_dist_u)) if use_q_feature else 0.0


        min_dist_v = min_out_dists[v]
        q_vu = ((1.0 + dist) / (1.0 + min_dist_v)) if use_q_feature else 0.0


        feat_uv = [dist, imst_w, is_knn, q_uv]
        feat_vu = [dist, imst_w, is_knn, q_vu]


        final_src.append(u)
        final_dst.append(v)
        final_attr_list.append(feat_uv)
        final_y.append(label)


        final_src.append(v)
        final_dst.append(u)
        final_attr_list.append(feat_vu)
        final_y.append(label)

    edge_index = torch.tensor([final_src, final_dst], dtype=torch.long)
    edge_attr = torch.tensor(final_attr_list, dtype=torch.float)
    y = torch.tensor(final_y, dtype=torch.float)

    data = Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr, y=y)

    data.full_real_pos_directed = torch.tensor([full_real_pos_directed], dtype=torch.float)
    data.full_total_directed = torch.tensor([full_total_directed], dtype=torch.float)
    return data


def load_dataset_group(group_dir, target_node_count=None, flat_layout=False, display_name=None):

    if flat_layout:
        coord_dir = group_dir
        label_dir = group_dir
    else:
        coord_dir = os.path.join(group_dir, "coords")
        label_dir = os.path.join(group_dir, "labels")

    tag = display_name if display_name is not None else os.path.basename(os.path.normpath(group_dir))



    coord_files = glob.glob(os.path.join(coord_dir, "*.tsp"))
    coord_files.sort()

    data_list = []
    print(f"loading [{tag}]， {len(coord_files)}  .tsp files...")

    start_time = time.time()
    for coord_path in coord_files:
        filename = os.path.basename(coord_path)
        name, _ = os.path.splitext(filename)
        label_path = os.path.join(label_dir, name + "_label.txt")
        if not os.path.exists(label_path):
            label_path = os.path.join(label_dir, name + "_labels.txt")

        if os.path.exists(label_path):
            try:
                coords = read_tsp_file(coord_path)
                if target_node_count is not None and len(coords) != target_node_count:
                    continue
                data = build_graph_data(coords, label_path)
                data_list.append(data)
            except Exception as e:
                print(f"  wrong {filename}: {e}")

    duration = time.time() - start_time
    print(f"finish [{tag}] ，time {duration:.2f}s，succeed {len(data_list)}/{len(coord_files)}")
    return data_list


def load_dataset_entries(
        entries,
        target_node_count=None,
        display_name=None,
        coarse_K=None,
        coarse_use_mst=True,
        enable_first_round_coarse=True,
        use_imst_feature=True,
        use_knn_feature=True,
        use_q_feature=True,
):

    tag = display_name if display_name is not None else "dataset(entries)"
    data_list = []
    print(f"loading [{tag}]，counts: {len(entries)} ...")

    start_time = time.time()
    for coord_path, label_dir in entries:
        filename = os.path.basename(coord_path)
        name, _ = os.path.splitext(filename)

        label_path = os.path.join(label_dir, name + "_label.txt")
        if not os.path.exists(label_path):
            label_path = os.path.join(label_dir, name + "_labels.txt")



        try:
            coords = read_tsp_file(coord_path)
            if target_node_count is not None and len(coords) != target_node_count:
                continue
            data = build_graph_data(
                coords,
                label_path,
                coarse_K=coarse_K,
                coarse_use_mst=coarse_use_mst,
                enable_first_round_coarse=enable_first_round_coarse,
                use_imst_feature=use_imst_feature,
                use_knn_feature=use_knn_feature,
                use_q_feature=use_q_feature,
            )
            data_list.append(data)
        except Exception as e:
            print(f"  wrong {filename}: {e}")

    duration = time.time() - start_time
    print(f"finish [{tag}] load，time {duration:.2f}s，succeed {len(data_list)}/{len(entries)}")
    return data_list



class TSP_EdgeGNN(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim=64):
        super().__init__()
        self.node_emb = nn.Linear(node_in_dim, hidden_dim)

        self.conv1 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_in_dim, heads=4, concat=False)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_in_dim, heads=4, concat=False)
        self.conv3 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_in_dim, heads=4, concat=False)

        self.edge_pred_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, edge_index, edge_attr):
        h = self.node_emb(x)
        h = F.relu(h)

        h = self.conv1(h, edge_index, edge_attr=edge_attr)
        h = F.relu(h)
        h = self.conv2(h, edge_index, edge_attr=edge_attr)
        h = F.relu(h)
        h = self.conv3(h, edge_index, edge_attr=edge_attr)

        row, col = edge_index
        edge_feat = torch.cat([h[row], h[col], edge_attr], dim=1)
        out = self.edge_pred_mlp(edge_feat).squeeze(-1)
        return out



def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_examples = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.edge_attr)
        loss = criterion(logits, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.y.size(0)
        total_examples += batch.y.size(0)
    return total_loss / total_examples if total_examples > 0 else 0


@torch.no_grad()
def eval_epoch(model, loader, device, threshold=0.5):
    model.eval()
    correct = 0
    total = 0
    tp = 0  # True Positives
    fp = 0  # False Positives
    fn = 0  # False Negatives

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr)
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        labels = batch.y

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        tp += ((preds == 1) & (labels == 1)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()

    acc = correct / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return acc, precision, recall, f1




if __name__ == "__main__":

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    TSP_ROOT = os.path.join(SCRIPT_DIR, "tsp")
    CLK_HARD_DIR = os.path.join(TSP_ROOT, "CLKhard")
    LKC_CHARD_DIR = os.path.join(TSP_ROOT, "LKCChard")
    CLK_EASY_DIR = os.path.join(TSP_ROOT, "CLKeasy")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    clk_coords_dir = os.path.join(CLK_HARD_DIR, "coords")
    if os.path.isdir(clk_coords_dir):
        clk_coords = glob.glob(os.path.join(clk_coords_dir, "*.tsp"))
    else:
        clk_coords = glob.glob(os.path.join(CLK_HARD_DIR, "*.tsp"))
    clk_coords.sort()
    lkc_coords = glob.glob(os.path.join(LKC_CHARD_DIR, "*.tsp"))
    lkc_coords.sort()
    easy_coords = glob.glob(os.path.join(CLK_EASY_DIR, "*.tsp"))
    easy_coords.sort()




    train_ratio = 1.0 / 3.0
    clk_train_n = int(len(clk_coords) * train_ratio)
    lkc_train_n = int(len(lkc_coords) * train_ratio)

    clk_train_files = clk_coords[:clk_train_n]
    lkc_train_files = lkc_coords[:lkc_train_n]

    clk_rem_files = clk_coords[clk_train_n:]
    lkc_rem_files = lkc_coords[lkc_train_n:]


    clk_label_dir = os.path.join(CLK_HARD_DIR, "labels")
    if not os.path.isdir(clk_label_dir):
        clk_label_dir = CLK_HARD_DIR
    lkc_label_dir = LKC_CHARD_DIR
    easy_label_dir = CLK_EASY_DIR

    val_entries = [(p, clk_label_dir) for p in clk_rem_files] + [(p, lkc_label_dir) for p in lkc_rem_files]
    test_entries = [(p, clk_label_dir) for p in clk_rem_files]




    train_entries = []
    for p in clk_train_files:
        train_entries.append((p, clk_label_dir))
    for p in lkc_train_files:
        train_entries.append((p, lkc_label_dir))




    COARSE_K = 15

    ENABLE_FIRST_ROUND_COARSE = True

    USE_IMST_FEATURE = True
    USE_KNN_FEATURE = True
    USE_Q_FEATURE = True
    USE_DIST_FEATURE = True


    train_data_list = load_dataset_entries(train_entries, display_name="train(CLKhard(1/3)+LKCChard(1/3))",
                                           coarse_K=COARSE_K, coarse_use_mst=True,
                                           enable_first_round_coarse=ENABLE_FIRST_ROUND_COARSE,
                                           use_imst_feature=USE_IMST_FEATURE,
                                           use_knn_feature=USE_KNN_FEATURE,
                                           use_q_feature=USE_Q_FEATURE)
    val_data_list = load_dataset_entries(val_entries, display_name="val(hard剩余(2/3))",
                                         coarse_K=COARSE_K, coarse_use_mst=True,
                                         enable_first_round_coarse=ENABLE_FIRST_ROUND_COARSE,
                                         use_imst_feature=USE_IMST_FEATURE,
                                         use_knn_feature=USE_KNN_FEATURE,
                                         use_q_feature=USE_Q_FEATURE)
    test_data_list = load_dataset_entries(test_entries, display_name="test(CLKhard剩余(2/3))",
                                          coarse_K=COARSE_K, coarse_use_mst=True,
                                          enable_first_round_coarse=ENABLE_FIRST_ROUND_COARSE,
                                          use_imst_feature=USE_IMST_FEATURE,
                                          use_knn_feature=USE_KNN_FEATURE,
                                          use_q_feature=USE_Q_FEATURE)




    train_loader = DataLoader(train_data_list, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_data_list, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_data_list, batch_size=4, shuffle=False)

    print(f"\n dataset ready:")
    print(f"  trianset numbers: {len(train_data_list)}")
    print(f"  validset numbers: {len(val_data_list)}")
    print(f"  testset numbers: {len(test_data_list)}")


    model = TSP_EdgeGNN(node_in_dim=2, edge_in_dim=4, hidden_dim=64).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    pos_weight = torch.tensor([300.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


    epochs = 50
    print(f"\n begin train({epochs} epochs)...")


    BEST_METRIC = "f1"
    best_metric_value = -1.0
    best_epoch = -1
    best_state = None

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)


        if (epoch + 1) % 5 == 0:
            acc, prec, rec, f1 = eval_epoch(model, val_loader, device, threshold=0.5)
            print(f"Epoch {epoch + 1:03d} | Train Loss: {train_loss:.4f} | "
                  f"Val Acc: {acc:.4f}, Prec: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

            metric_value = rec if BEST_METRIC == "recall" else f1
            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_epoch = epoch + 1

                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n best model：epoch={best_epoch}, best_{BEST_METRIC}={best_metric_value:.6f}")
    else:
        print("\n wrong")

    print("\nfinish！")


    model_save_path = os.path.join(SCRIPT_DIR, "tsp_gnn_model.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"model save: {model_save_path}")


    LAMBDA = 1.0  # score = recall - LAMBDA * keep_ratio
    threshold_candidates = [i / 100.0 for i in range(10, 96, 10)]

    @torch.no_grad()
    def eval_list_score(data_list, threshold: float):
        model.eval()
        recalls = []
        keep_ratios = []
        for sample in data_list:
            sample = sample.to(device)
            logits = model(sample.x, sample.edge_index, sample.edge_attr)
            probs = torch.sigmoid(logits)
            preds = probs > threshold

            labels = sample.y
            true_pos = int((preds & (labels == 1)).sum().item())
            kept = int(preds.sum().item())

            full_real_pos = float(sample.full_real_pos_directed.item()) if hasattr(sample, "full_real_pos_directed") else float((labels == 1).sum().item())
            full_total = float(sample.full_total_directed.item()) if hasattr(sample, "full_total_directed") else float(labels.numel())

            recall = true_pos / full_real_pos if full_real_pos > 0 else 0.0
            keep_ratio = kept / full_total if full_total > 0 else 0.0
            recalls.append(recall)
            keep_ratios.append(keep_ratio)
        mean_recall = sum(recalls) / len(recalls)
        mean_keep = sum(keep_ratios) / len(keep_ratios)
        score = mean_recall - LAMBDA * mean_keep
        return mean_recall, mean_keep, score

    print(f"\n (COARSE_K={COARSE_K}, lambda={LAMBDA})")
    best_p = None
    best_score = float("-inf")
    best_val = None
    for p in threshold_candidates:
        r, krr, sc = eval_list_score(val_data_list, threshold=p)
        print(f"  p={p:.2f} | recall={r:.4f} keep_ratio={krr:.4f} score={sc:.6f}")
        if sc > best_score:
            best_score = sc
            best_p = p
            best_val = (r, krr, sc)

    print(f" p={best_p:.2f}（val recall={best_val[0]:.4f}, keep_ratio={best_val[1]:.4f}, score={best_val[2]:.6f}）")

    if len(test_data_list) > 0:
        r, krr, _ = eval_list_score(test_data_list, threshold=best_p)

        print(f" testset numbers: {len(test_data_list)}")
        print(f"Average Recall: {r:.4f}")
        print(f"Average keep ratio : {krr:.4f}")
