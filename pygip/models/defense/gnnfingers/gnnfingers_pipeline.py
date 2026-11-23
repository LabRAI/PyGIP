# gnnfingers_full_pipeline.py
# Full Unified Pipeline for You et al. (2024) — Node/Graph Classification + Link Prediction + Graph Matching
# Saves per-dataset metrics.json, RU plots (if --save-plots), and a final summary CSV/Markdown (if --summarize)

import os, copy, json, random, argparse, glob
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from collections import OrderedDict
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from torch_geometric.datasets import Planetoid, TUDataset, GEDDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling, to_undirected, degree
from torch_geometric.nn import GCNConv, global_mean_pool

import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Utils ----------------
def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def ensure_dir(path): os.makedirs(path, exist_ok=True)

def np_trapezoid(y,x):
    if hasattr(np,"trapezoid"): return np.trapezoid(y,x)
    return np.trapz(y,x)

def plot_ru_from_scores(y_true,y_score,aruc,out_png):
    fpr,tpr,_=roc_curve(y_true,y_score)
    U,R=1-fpr,tpr; idx=np.argsort(U); U,R=U[idx],R[idx]
    plt.figure(figsize=(5,4)); plt.plot(U,R,marker=".",linewidth=1)
    plt.xlabel("Uniqueness (1 - FPR)"); plt.ylabel("Robustness (TPR)")
    plt.title(f"R–U Curve (ARUC ≈ {aruc:.3f})"); plt.tight_layout()
    plt.savefig(out_png); plt.close()

def write_metrics(outdir, name, payload):
    dstdir = os.path.join(outdir, name)
    ensure_dir(dstdir)
    with open(os.path.join(dstdir, "metrics.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2)

def _safe_read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def collect_metrics(outdir="outputs"):
    rows = []
    for p in sorted(glob.glob(os.path.join(outdir, "*", "metrics.json"))):
        dsname = os.path.basename(os.path.dirname(p))
        m = _safe_read_json(p)
        row = OrderedDict()
        row["dataset"]       = dsname
        row["val_acc"]       = m.get("val_acc", None)
        row["test_acc"]      = m.get("test_acc", None)
        row["lp_auc_test"]   = m.get("lp_auc_test", None)
        row["fp_auc_train"]  = m.get("fp_auc_train", None)
        row["fp_auc_test"]   = m.get("fp_auc_test", None)
        row["mean_test_acc"] = m.get("mean_test_acc", None)
        row["aruc"]          = m.get("aruc", None)
        row["matcher_auc"]   = m.get("matcher_auc", None)
        row["error"]         = m.get("error", None)
        rows.append(row)
    return rows

def write_summary_csv(rows, out_csv):
    ensure_dir(os.path.dirname(out_csv))
    cols = ["dataset", "val_acc", "test_acc", "lp_auc_test",
            "fp_auc_train", "fp_auc_test", "mean_test_acc", "aruc", "matcher_auc", "error"]
    with open(out_csv, "w", encoding="utf-8", newline="\n") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            vals = []
            for c in cols:
                v = r.get(c, "")
                if v is None:
                    v = ""
                vals.append(str(v))
            f.write(",".join(vals) + "\n")

def write_summary_markdown(rows, out_md):
    ensure_dir(os.path.dirname(out_md))
    cols = ["dataset", "val_acc", "test_acc", "lp_auc_test",
            "fp_auc_train", "fp_auc_test", "mean_test_acc", "aruc", "matcher_auc", "error"]

    def fmt(v):
        if v is None or v == "":
            return "-"
        if isinstance(v, float):
            return f"{v:.4f}"
        try:
            return f"{float(v):.4f}"
        except Exception:
            return str(v)

    best_aruc = max([(r["dataset"], r.get("aruc", 0.0)) for r in rows], key=lambda x: (x[1] or 0.0), default=("–", 0.0))
    best_fp   = max([(r["dataset"], r.get("fp_auc_test", 0.0)) for r in rows], key=lambda x: (x[1] or 0.0), default=("–", 0.0))

    with open(out_md, "w", encoding="utf-8", newline="\n") as f:
        f.write("# GNNFingers – Summary Report\n\n")
        f.write("**Auto-generated from pipeline outputs.**\n\n")
        f.write("## Highlights\n")
        f.write(f"- Best ARUC: **{best_aruc[0]}** (≈ **{(best_aruc[1] or 0.0):.4f}**)\n")
        f.write(f"- Best Fingerprint AUROC (test): **{best_fp[0]}** (≈ **{(best_fp[1] or 0.0):.4f}**)\n\n")
        f.write("## Results Table\n\n")
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"]*len(cols)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |\n")
        f.write("\n## Notes\n- `val_acc` for node datasets; graph datasets show `test_acc`.\n"
                "- `lp_auc_test` for link prediction tasks.\n"
                "- `mean_test_acc` = average accuracy across thresholds in [0,1] on the FP hold-out.\n"
                "- `aruc` from ROC-based R–U curve (U=1−FPR vs R=TPR).\n"
                "- `matcher_auc` is the held-out AUC of the trained probe verifier (if enabled).\n"
                "- Rows with `error` indicate datasets skipped due to download issues.\n")

# ---------------- Models ----------------
class GCNNode(nn.Module):
    def __init__(self,in_dim,hid_dim,num_classes):
        super().__init__()
        self.c1=GCNConv(in_dim,hid_dim); self.c2=GCNConv(hid_dim,num_classes)
    def forward(self,x,e):
        h=F.relu(self.c1(x,e)); return self.c2(h,e)

class GCNGraph(nn.Module):
    def __init__(self,in_dim,hid_dim,num_classes):
        super().__init__()
        self.in_dim=in_dim
        self.c1=GCNConv(in_dim,hid_dim); self.c2=GCNConv(hid_dim,hid_dim)
        self.lin=nn.Linear(hid_dim,num_classes)
    def _ensure_x(self,x,e,b):
        if x is not None: return x
        n=int(e.max().item())+1 if e is not None else 1
        return torch.ones((n,self.in_dim),dtype=torch.float32,device=e.device)
    def forward(self,x,e,b):
        x=self._ensure_x(x,e,b)
        h=F.relu(self.c1(x,e)); h=F.relu(self.c2(h,e))
        return self.lin(global_mean_pool(h,b.batch))
    def forward_features(self,x,e,b):
        x=self._ensure_x(x,e,b)
        h=F.relu(self.c1(x,e)); h=F.relu(self.c2(h,e))
        return global_mean_pool(h,b.batch)

class GCNEncoder(nn.Module):
    def __init__(self,in_dim,hid_dim):
        super().__init__(); self.c1=GCNConv(in_dim,hid_dim); self.c2=GCNConv(hid_dim,hid_dim)
    def forward(self,x,e): return self.c2(F.relu(self.c1(x,e)),e)

class FingerprintNet(nn.Module):
    def __init__(self,in_dim):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(in_dim,128), nn.LeakyReLU(0.2,True),
            nn.Linear(128,64), nn.LeakyReLU(0.2,True),
            nn.Linear(64,32), nn.LeakyReLU(0.2,True),
            nn.Linear(32,1), nn.Sigmoid()
        )
    def forward(self,x): return self.net(x).view(-1)

class PairwiseVerifier(nn.Module):
    def __init__(self, dim, p_drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 64), nn.ReLU(inplace=True), nn.Dropout(p_drop),
            nn.Linear(64, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1), nn.Sigmoid(),
        )
    def forward(self, x): return self.net(x).view(-1)

# ---------------- Variants ----------------
def _perturb_weights_(m,scale):
    with torch.no_grad():
        for p in m.parameters(): p.add_(torch.randn_like(p)*scale)

def _finetune_steps_(m,loader,steps=3,lr=5e-4):
    m=m.to(DEVICE).train(); opt=torch.optim.Adam(m.parameters(),lr=lr)
    crit=nn.CrossEntropyLoss(); it=iter(loader)
    for _ in range(steps):
        try: b=next(it)
        except StopIteration: it=iter(loader); b=next(it)
        b=b.to(DEVICE); opt.zero_grad()
        out=m(b.x,b.edge_index,b)
        loss=crit(out,b.y); loss.backward(); opt.step()

def _magnitude_prune_(m,keep=0.9):
    with torch.no_grad():
        for p in m.parameters():
            flat=p.view(-1); k=max(1,int(len(flat)*keep))
            thr=torch.topk(flat.abs(),k).values.min()
            mask=(flat.abs()>=thr).float(); flat.mul_(mask)

# ---------------- FP generation (node/graph/link/match) ----------------
@torch.no_grad()
def fingerprints_node(model,data,n_pos=200,n_neg=200,pos_noise=0.01,neg_noise=0.1):
    fps,labels=[],[]
    for _ in range(n_pos):
        m=copy.deepcopy(model).to(DEVICE).eval(); _perturb_weights_(m,pos_noise)
        logits=m(data.x.to(DEVICE),data.edge_index.to(DEVICE))
        fps.append(logits.mean(0).cpu().numpy()); labels.append(1)
    for _ in range(n_neg):
        m=copy.deepcopy(model).to(DEVICE).eval(); _perturb_weights_(m,neg_noise)
        logits=m(data.x.to(DEVICE),data.edge_index.to(DEVICE))
        fps.append(logits.mean(0).cpu().numpy()); labels.append(0)
    return np.array(fps),np.array(labels)

@torch.no_grad()
def fingerprints_graph(model,ds,n_pos=200,n_neg=200,pos_noise=0.01,neg_noise=0.1,batch_size=32):
    fps,labels=[],[]
    loader=DataLoader(ds,batch_size=batch_size,shuffle=True,drop_last=True); it=iter(loader)
    def next_batch():
        nonlocal it
        try: return next(it)
        except StopIteration: it=iter(loader); return next(it)
    for _ in range(n_pos):
        b=next_batch().to(DEVICE); m=copy.deepcopy(model).to(DEVICE).eval()
        _perturb_weights_(m,pos_noise); emb=m.forward_features(b.x,b.edge_index,b)
        fps.append(emb.mean(0).cpu().numpy()); labels.append(1)
    for _ in range(n_neg):
        b=next_batch().to(DEVICE); m=copy.deepcopy(model).to(DEVICE).eval()
        _perturb_weights_(m,neg_noise); emb=m.forward_features(b.x,b.edge_index,b)
        fps.append(emb.mean(0).cpu().numpy()); labels.append(0)
    return np.array(fps),np.array(labels)

# ------ Link prediction helpers ------
def make_lp_splits(edge_index,num_val=500,num_test=1000,seed=42):
    edge_index=to_undirected(edge_index); E=edge_index.shape[1]
    rng=np.random.RandomState(seed); perm=rng.permutation(E)
    test_e=int(num_test); val_e=int(num_val)
    test_pos=edge_index[:,perm[:test_e]]
    val_pos=edge_index[:,perm[test_e:test_e+val_e]]
    train_pos=edge_index[:,perm[test_e+val_e:]]
    num_nodes=int(edge_index.max().item())+1
    def sample_negs(num,existing):
        return negative_sampling(edge_index=existing,num_nodes=num_nodes,num_neg_samples=num)
    val_neg=sample_negs(val_pos.shape[1],edge_index)
    test_neg=sample_negs(test_pos.shape[1],edge_index)
    train_neg=sample_negs(train_pos.shape[1],edge_index)
    return train_pos,val_pos,test_pos,train_neg,val_neg,test_neg

def train_linkpred(encoder,x,e,pos,neg,epochs=100,lr=0.01):
    encoder=encoder.to(DEVICE); opt=torch.optim.Adam(encoder.parameters(),lr=lr); bce=nn.BCELoss()
    x,e,pos,neg=x.to(DEVICE),e.to(DEVICE),pos.to(DEVICE),neg.to(DEVICE)
    for _ in range(epochs):
        encoder.train(); opt.zero_grad()
        z=encoder(x,e); pi,pj=z[pos[0]],z[pos[1]]; ni,nj=z[neg[0]],z[neg[1]]
        pos_prob=torch.sigmoid((pi*pj).sum(1)); neg_prob=torch.sigmoid((ni*nj).sum(1))
        y=torch.cat([torch.ones_like(pos_prob),torch.zeros_like(neg_prob)])
        pred=torch.cat([pos_prob,neg_prob]); loss=bce(pred,y); loss.backward(); opt.step()
    return encoder

@torch.no_grad()
def eval_link_auc(encoder,x,e,pos,neg):
    z=encoder(x.to(DEVICE),e.to(DEVICE))
    pi,pj=z[pos[0]],z[pos[1]]; ni,nj=z[neg[0]],z[neg[1]]
    pos_prob=torch.sigmoid((pi*pj).sum(1)).cpu().numpy()
    neg_prob=torch.sigmoid((ni*nj).sum(1)).cpu().numpy()
    y_true=np.concatenate([np.ones_like(pos_prob),np.zeros_like(neg_prob)])
    y_pred=np.concatenate([pos_prob,neg_prob])
    return roc_auc_score(y_true,y_pred), y_true, y_pred

# ------ Graph matching helpers ------
def ensure_graph_node_features(dataset, dim: int = 2):
    for d in dataset:
        if getattr(d, "x", None) is None:
            try:
                num_nodes = int(d.num_nodes)
                deg = degree(d.edge_index[0], num_nodes=num_nodes, dtype=torch.float32)
                if deg.max() > 0: deg = deg / deg.max()
                const = torch.ones_like(deg)
                d.x = torch.stack([deg, const], dim=1)
            except Exception:
                d.x = torch.ones((d.num_nodes, dim), dtype=torch.float32)

def embed_all_graphs(model: GCNGraph, dataset):
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    embs = []
    with torch.no_grad():
        model.eval()
        for batch in loader:
            batch = batch.to(DEVICE)
            e = model.forward_features(batch.x, batch.edge_index, batch)  # [B,H]
            e = F.normalize(e, p=2, dim=1)
            embs.append(e.detach().cpu())
    return torch.cat(embs, dim=0)

def build_matching_probe_pairs_by_label(dataset, n_pairs=256, seed=42):
    rng = np.random.RandomState(seed)
    labels = np.array([int(d.y.item()) for d in dataset])
    idx_by_label = {}
    for i, y in enumerate(labels):
        idx_by_label.setdefault(y, []).append(i)
    all_idx = np.arange(len(dataset))
    pos_pairs, neg_pairs = [], []
    half = n_pairs // 2
    # positives
    for _ in range(half):
        y = rng.choice(list(idx_by_label.keys()))
        if len(idx_by_label[y]) < 2: continue
        i, j = rng.choice(idx_by_label[y], size=2, replace=False)
        pos_pairs.append((i, j))
    # negatives
    for _ in range(half):
        i = rng.choice(all_idx)
        tries = 0
        while True:
            j = rng.choice(all_idx)
            if labels[j] != labels[i]: break
            tries += 1
            if tries > 20: j = (i + 1) % len(all_idx); break
        neg_pairs.append((i, j))
    pairs = pos_pairs + neg_pairs
    pair_labels = [1]*len(pos_pairs) + [0]*len(neg_pairs)
    return np.array(pairs, dtype=int), np.array(pair_labels, dtype=int)

def build_matching_probe_pairs_by_ged(ged_dataset: GEDDataset, n_pairs=256, pos_q=0.2, neg_q=0.8):
    N = len(ged_dataset)
    M = ged_dataset.norm_ged[:N, :N].copy()
    iu, ju = np.triu_indices(N, k=1)
    vals = M[iu, ju]
    lo = np.quantile(vals, pos_q); hi = np.quantile(vals, neg_q)
    pos_idx = np.where(vals <= lo)[0]; neg_idx = np.where(vals >= hi)[0]
    n_half = n_pairs // 2
    if len(pos_idx)==0 or len(neg_idx)==0:
        n_half = min(n_half, max(len(pos_idx), len(neg_idx)))
    rng = np.random.RandomState(42)
    pos_sel = rng.choice(pos_idx, size=min(n_half, len(pos_idx)), replace=False)
    neg_sel = rng.choice(neg_idx, size=min(n_half, len(neg_idx)), replace=False)
    pos_pairs = list(zip(iu[pos_sel], ju[pos_sel]))
    neg_pairs = list(zip(iu[neg_sel], ju[neg_sel]))
    pairs = pos_pairs + neg_pairs
    labels = [1]*len(pos_pairs) + [0]*len(neg_pairs)
    return np.array(pairs, dtype=int), np.array(labels, dtype=int)

@torch.no_grad()
def matching_fingerprint(model: GCNGraph, dataset, probe_pairs, verifier=None):
    G = embed_all_graphs(model, dataset)  # [N,H]
    dim = G.shape[1]
    if verifier is None:
        verifier = PairwiseVerifier(dim).to(DEVICE).eval()
    diffs = []
    for (i, j) in probe_pairs:
        d = torch.abs(G[i] - G[j]).unsqueeze(0).to(DEVICE)
        diffs.append(d)
    diffs = torch.cat(diffs, dim=0)  # [P, H]
    p = verifier(diffs).detach().cpu().numpy().reshape(-1)
    return p

def train_pairwise_verifier(G_emb: torch.Tensor, pairs: np.ndarray, labels: np.ndarray,
                            epochs=5, lr=1e-3, val_frac=0.2, seed=42):
    P = len(pairs)
    idx = np.arange(P)
    tr_idx, te_idx = train_test_split(idx, test_size=val_frac, stratify=labels, random_state=seed)

    def make_tensor(sel):
        diffs, ys = [], []
        for k in sel:
            i, j = pairs[k]
            diffs.append(torch.abs(G_emb[i] - G_emb[j]).unsqueeze(0))
            ys.append(labels[k])
        X = torch.cat(diffs, dim=0).to(DEVICE)
        y = torch.tensor(ys, dtype=torch.float32, device=DEVICE)
        return X, y

    Xtr, ytr = make_tensor(tr_idx)
    Xte, yte = make_tensor(te_idx)

    dim = G_emb.shape[1]
    net = PairwiseVerifier(dim).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    bce = nn.BCELoss()

    for _ in range(epochs):
        net.train(); opt.zero_grad()
        pred = net(Xtr); loss = bce(pred, ytr)
        loss.backward(); opt.step()

    net.eval()
    with torch.no_grad():
        p = net(Xte).detach().cpu().numpy()
    val_auc = roc_auc_score(yte.detach().cpu().numpy(), p)
    return net, float(val_auc)

# ---------------- Training wrappers ----------------
def train_node(model,data,epochs=200,lr=0.01,wd=5e-4):
    model=model.to(DEVICE).train()
    opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=wd)
    best_val,best_test=0.0,0.0
    for _ in range(epochs):
        opt.zero_grad()
        out=model(data.x.to(DEVICE),data.edge_index.to(DEVICE))
        loss=F.cross_entropy(out[data.train_mask],data.y[data.train_mask].to(DEVICE))
        loss.backward(); opt.step()
        model.eval(); pred=out.argmax(1).cpu()
        val=(pred[data.val_mask]==data.y[data.val_mask]).float().mean().item()
        test=(pred[data.test_mask]==data.y[data.test_mask]).float().mean().item()
        if val>best_val: best_val,best_test=val,test
        model.train()
    return best_val,best_test

def train_graph(model,loader,epochs=50):
    model.to(DEVICE).train(); opt=torch.optim.Adam(model.parameters(),lr=0.01)
    crit=nn.CrossEntropyLoss()
    for _ in range(epochs):
        for b in loader:
            b=b.to(DEVICE); opt.zero_grad(); out=model(b.x,b.edge_index,b)
            loss=crit(out,b.y); loss.backward(); opt.step()

@torch.no_grad()
def eval_graph(model,loader):
    model.eval(); correct,total=0,0
    for b in loader:
        b=b.to(DEVICE); out=model(b.x,b.edge_index,b)
        pred=out.argmax(1); correct+=(pred==b.y).sum().item(); total+=b.y.size(0)
    return correct/max(1,total)

# ---------------- Main Runner ----------------
def run_dataset(name, args):
    results = {}
    if name in ["Cora","CiteSeer","PubMed"]:
        ds=Planetoid(root=f"data/{name}", name=name); data=ds[0]
        model=GCNNode(ds.num_node_features, args.hidden_node, ds.num_classes)
        val_acc,test_acc=train_node(model,data,epochs=args.epochs_node)
        print(f"[{name}] Node Acc — Val: {val_acc:.3f}, Test: {test_acc:.3f}")
        fps,labels=fingerprints_node(model,data,n_pos=args.variants,n_neg=args.variants,
                                     pos_noise=args.pos_noise,neg_noise=args.neg_noise)
        results.update({"val_acc": float(val_acc), "test_acc": float(test_acc)})

    elif name in ["PROTEINS","NCI1","ENZYMES"]:
        ds=TUDataset(root=f"data/{name}", name=name).shuffle()
        in_dim=max(1,ds.num_node_features)
        model=GCNGraph(in_dim,args.hidden_graph,ds.num_classes).to(DEVICE)
        n=int(0.8*len(ds)); train_loader=DataLoader(ds[:n],batch_size=64,shuffle=True)
        test_loader=DataLoader(ds[n:],batch_size=64)
        train_graph(model,train_loader,epochs=args.epochs_graph)
        acc=eval_graph(model,test_loader); print(f"[{name}] Graph Acc — Test: {acc:.3f}")
        fps,labels=fingerprints_graph(model,ds,n_pos=args.variants,n_neg=args.variants,
                                      pos_noise=args.pos_noise,neg_noise=args.neg_noise)
        results.update({"test_acc": float(acc)})

    elif name in ["CoraLP","CiteSeerLP"]:
        base="Cora" if name=="CoraLP" else "CiteSeer"
        ds=Planetoid(root=f"data/{base}", name=base); data=ds[0]
        tr_pos,val_pos,te_pos,tr_neg,val_neg,te_neg=make_lp_splits(data.edge_index)
        encoder=GCNEncoder(ds.num_node_features,args.hidden_node)
        encoder=train_linkpred(encoder,data.x,data.edge_index,tr_pos,tr_neg,epochs=args.epochs_lp)
        lp_auc, y_true_te, y_pred_te = eval_link_auc(encoder,data.x,data.edge_index,te_pos,te_neg)
        print(f"[{name}] Link Prediction AUC — Test: {lp_auc:.3f}")
        # Fingerprints: use validation pairs as probe (scalar score per variant)
        fps,labels=[],[]
        for _ in range(args.variants):
            m=copy.deepcopy(encoder); _perturb_weights_(m,args.pos_noise)
            auc_val,_,_ = eval_link_auc(m,data.x,data.edge_index,val_pos,val_neg)
            fps.append(auc_val); labels.append(1)
        for _ in range(args.variants):
            m=copy.deepcopy(encoder); _perturb_weights_(m,args.neg_noise)
            auc_val,_,_ = eval_link_auc(m,data.x,data.edge_index,val_pos,val_neg)
            fps.append(auc_val); labels.append(0)
        fps=np.array(fps).reshape(-1,1); labels=np.array(labels)
        results.update({"lp_auc_test": float(lp_auc)})
    else:
        # ---- Graph matching datasets ----
        if name.upper()=="AIDS" or name.upper()=="AIDS700NEF":
            dsname = "AIDS700nef"
            try:
                ds_train = GEDDataset(root=f"data/{dsname}", name=dsname, train=True)
                ds_test  = GEDDataset(root=f"data/{dsname}", name=dsname, train=False)
            except Exception as e:
                print(f"[WARN] GEDDataset({dsname}) load failed: {e}")
                write_metrics(args.outdir, f"{name}_MATCH", {"error": f"Failed to load {dsname}."})
                return
            ds = ds_train
            in_dim = getattr(ds, "num_node_features", 1)
            model = GCNGraph(in_dim, args.hidden_graph, getattr(ds, "num_classes", 2)).to(DEVICE)
            loader = DataLoader(ds, batch_size=64, shuffle=True)
            train_graph(model, loader, epochs=args.epochs_graph)
            # probe pairs via GED
            probe_pairs, probe_labels = build_matching_probe_pairs_by_ged(ds, n_pairs=args.match_probe)
        else:
            # TU-based matching (IMDBMULTI, IMDBBINARY, MUTAG)
            canon = {"IMDBMULTI":"IMDB-MULTI", "IMDBBINARY":"IMDB-BINARY", "MUTAG":"MUTAG"}[name.upper()]
            try:
                ds = TUDataset(root=f"data/{canon}", name=canon).shuffle()
            except Exception as e:
                print(f"[WARN] TUDataset({canon}) load failed: {e}")
                write_metrics(args.outdir, f"{name}_MATCH", {"error": f"Failed to load {canon}."})
                return
            # ensure features for IMDB-*
            ensure_graph_node_features(ds)
            in_dim = max(1, getattr(ds, "num_node_features", 0))
            model = GCNGraph(in_dim, args.hidden_graph, ds.num_classes).to(DEVICE)
            n = len(ds); n_train = int(round(0.8 * n))
            train_ds = ds[:n_train]
            loader = DataLoader(train_ds, batch_size=64, shuffle=True)
            train_graph(model, loader, epochs=args.epochs_graph)
            probe_pairs, probe_labels = build_matching_probe_pairs_by_label(ds, n_pairs=args.match_probe)

        # (optional) train pairwise verifier
        trained_verifier = None
        matcher_auc = None
        with torch.no_grad():
            G_base = embed_all_graphs(model, ds)
        if args.train_matcher:
            trained_verifier, matcher_auc = train_pairwise_verifier(
                G_base, probe_pairs, probe_labels, epochs=args.matcher_epochs, lr=args.matcher_lr,
                val_frac=0.2, seed=args.seed
            )
            print(f"[{name}_MATCH] Trained verifier AUC (held-out pairs): {matcher_auc:.4f}")

        # Build fingerprints by evaluating fixed probe set through (noisy) variants
        fps, labels = [], []
        for _ in range(args.variants):
            m = copy.deepcopy(model).to(DEVICE).eval()
            _perturb_weights_(m, args.pos_noise)
            fp = matching_fingerprint(m, ds, probe_pairs, verifier=trained_verifier)
            fps.append(fp); labels.append(1)
        for _ in range(args.variants):
            m = copy.deepcopy(model).to(DEVICE).eval()
            _perturb_weights_(m, args.neg_noise)
            fp = matching_fingerprint(m, ds, probe_pairs, verifier=trained_verifier)
            fps.append(fp); labels.append(0)
        fps = np.array(fps); labels = np.array(labels)
        print(f"[{name}_MATCH] FPS shape: {fps.shape}, Labels: {np.bincount(labels)}")

        # Train fingerprint verifier (FingerprintNet) on the probe vectors
        X=torch.tensor(fps,dtype=torch.float32).to(DEVICE)
        y=torch.tensor(labels,dtype=torch.float32).to(DEVICE)
        tr,te=train_test_split(np.arange(len(y)),test_size=0.3,stratify=labels,random_state=args.seed)
        clf=FingerprintNet(X.shape[1]).to(DEVICE)
        opt=torch.optim.Adam(clf.parameters(),lr=1e-3); crit=nn.BCELoss()
        for _ in range(100):
            clf.train(); opt.zero_grad(); loss=crit(clf(X[tr]),y[tr]); loss.backward(); opt.step()
        clf.eval()
        with torch.no_grad(): pte=clf(X[te]).detach().cpu().numpy()
        yte=y[te].detach().cpu().numpy()
        fp_auc_test = roc_auc_score(yte,pte)
        print(f"[{name}_MATCH] Fingerprinting AUROC (test): {fp_auc_test:.4f}")
        # ARUC via ROC-based R–U: in this setup equals AUROC
        aruc = float(fp_auc_test)
        if args.save_plots:
            out=os.path.join(args.outdir,f"{name}_MATCH"); ensure_dir(out)
            plot_ru_from_scores(yte,pte,aruc,os.path.join(out,"robustness_uniqueness.png"))
        write_metrics(args.outdir, f"{name}_MATCH", {
            "fp_auc_test": float(fp_auc_test),
            "mean_test_acc": None,
            "aruc": aruc,
            "matcher_auc": matcher_auc
        })
        return  # matching branch returns after saving metrics

    # ---- Common verifier training & saving for node/graph/link ----
    print(f"[{name}] FPS shape: {fps.shape}, Labels: {np.bincount(labels)}")
    X=torch.tensor(fps,dtype=torch.float32).to(DEVICE)
    y=torch.tensor(labels,dtype=torch.float32).to(DEVICE)
    tr,te=train_test_split(np.arange(len(y)),test_size=0.3,stratify=labels,random_state=args.seed)
    clf=FingerprintNet(X.shape[1]).to(DEVICE)
    opt=torch.optim.Adam(clf.parameters(),lr=1e-3); crit=nn.BCELoss()
    for _ in range(100):
        clf.train(); opt.zero_grad(); loss=crit(clf(X[tr]),y[tr]); loss.backward(); opt.step()
    clf.eval()
    with torch.no_grad(): pte=clf(X[te]).detach().cpu().numpy()
    yte=y[te].detach().cpu().numpy()
    fp_auc_test = roc_auc_score(yte,pte)
    print(f"[{name}] Fingerprinting AUROC (test): {fp_auc_test:.4f}")
    aruc = float(fp_auc_test)
    if args.save_plots:
        out=os.path.join(args.outdir,name); ensure_dir(out)
        plot_ru_from_scores(yte,pte,aruc,os.path.join(out,"robustness_uniqueness.png"))

    payload = {
        "val_acc": results.get("val_acc"),
        "test_acc": results.get("test_acc"),
        "lp_auc_test": results.get("lp_auc_test"),
        "fp_auc_test": float(fp_auc_test),
        "mean_test_acc": None,
        "aruc": aruc,
        "matcher_auc": None
    }
    write_metrics(args.outdir, name, payload)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",default="all",
                   choices=["all","Cora","CiteSeer","PubMed",
                            "PROTEINS","NCI1","ENZYMES",
                            "CoraLP","CiteSeerLP",
                            "MUTAG","IMDBMULTI","IMDBBINARY","AIDS"])
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--hidden-graph",type=int,default=64)
    p.add_argument("--hidden-node",type=int,default=16)
    p.add_argument("--epochs-graph",type=int,default=50)
    p.add_argument("--epochs-node",type=int,default=200)
    p.add_argument("--epochs-lp",type=int,default=100)
    p.add_argument("--variants",type=int,default=200)
    p.add_argument("--pos-noise",type=float,default=0.01)
    p.add_argument("--neg-noise",type=float,default=0.10)
    p.add_argument("--match-probe",type=int,default=1024,
                   help="number of probe pairs for matching fingerprints")
    p.add_argument("--train-matcher",action="store_true",
                   help="train pairwise verifier on |g_i - g_j| for matching")
    p.add_argument("--matcher-epochs",type=int,default=30)
    p.add_argument("--matcher-lr",type=float,default=8e-4)
    p.add_argument("--save-plots",action="store_true")
    p.add_argument("--summarize",action="store_true")
    p.add_argument("--outdir",default="outputs")
    args=p.parse_args()
    set_seed(args.seed); ensure_dir(args.outdir)

    print("\n=== GNNFingers Full Unified Pipeline ===\n")

    if args.dataset=="all":
        datasets=["Cora","CiteSeer","PubMed",
                  "PROTEINS","NCI1","ENZYMES",
                  "CoraLP","CiteSeerLP",
                  "MUTAG","IMDBMULTI","IMDBBINARY","AIDS"]
        for d in datasets:
            run_dataset(d, args)
    else:
        run_dataset(args.dataset, args)

    if args.summarize:
        rows = collect_metrics(args.outdir)
        if len(rows)>0:
            write_summary_csv(rows, os.path.join(args.outdir, "summary_metrics.csv"))
            write_summary_markdown(rows, os.path.join(args.outdir, "summary_report.md"))
            print(f"\nWrote: {os.path.join(args.outdir, 'summary_metrics.csv')}")
            print(f"Wrote: {os.path.join(args.outdir, 'summary_report.md')}")
        else:
            print("No metrics found to summarize.")

if __name__=="__main__":
    main()
