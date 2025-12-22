import copy
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm
from datetime import datetime
from feature_net import FeatureExtractor
from torch.autograd import Function
import torch.nn.functional as F
# ============ 配置参数 ============
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


result_dir = r"..\result"

model_save_dir = r"..\_model_ori"

data_dir = r"E:\Cross_subject\data"

os.makedirs(result_dir, exist_ok=True)
os.makedirs(model_save_dir, exist_ok=True)

# 实验参数
num_subjects = 9
num_classes = 7
batch_size = 128
pretrain_epochs = 100
adapt_epochs = 100
feat_dim = 32


# ============ 辅助函数 ============
def mixup_data(x, y, ids, alpha=0.5):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    ids_a, ids_b = ids, ids[index]
    return mixed_x, y_a, y_b, ids_a, ids_b, lam


def mixup_criterion(pred, y_a, y_b, lam):
    criterion = nn.CrossEntropyLoss()
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x, alpha=1):
    return GradReverse.apply(x, alpha)


# ============ 训练/评估函数 ============
def pretrain_one_epoch(model_F, model_C, model_D, loader, optimizer, epoch):
    model_F.train()
    model_C.train()
    model_D.train()
    total_correct_cls = 0
    total_samples = 0
    total_loss = 0.0

    for batch in tqdm(loader, desc=f"Pretrain Epoch {epoch + 1:03d}", leave=False):
        data, label, domain = batch
        data, label, domain = data.to(device), label.to(device).long(), domain.to(device).long()

        data_mix, targets_a, targets_b, ids_a, ids_b, lam = mixup_data(data, label, domain)
        batch_size_curr = data.size(0)

        feat, feat2 = model_F(data_mix)
        out = model_C(feat2)
        feat_rev = grad_reverse(feat2)
        out2 = model_D(feat_rev)

        loss_d=nn.CrossEntropyLoss()(out2, domain)

        loss_c = mixup_criterion(out, targets_a, targets_b, 0)

        loss = loss_c+loss_d

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        with torch.no_grad():
            feat_ori, feat_ori2 = model_F(data)
            pred_labels = model_C(feat_ori2).argmax(dim=1)
            total_correct_cls += (pred_labels == label).sum().item()
            total_samples += batch_size_curr
            total_loss += loss.item() * batch_size_curr

    return total_correct_cls / total_samples, total_loss / total_samples


# def entropy_weighted_moment_matching(source, target, target_logits):
#
#     T = 2.0
#     probs = F.softmax(target_logits, dim=1)
#     entropy = -torch.sum(probs * torch.log(probs + 1e-5), dim=1)
#
#
#     w_t = 1.0 + torch.exp(-entropy / T)
#
#     w_t = w_t.detach()
#     w_t = w_t / torch.sum(w_t)
#
#
#     n_s = source.size(0)
#     w_s = torch.ones(n_s).to(device) / n_s
#
#     mu_s = torch.matmul(w_s.unsqueeze(0), source).squeeze(0)
#     mu_t = torch.matmul(w_t.unsqueeze(0), target).squeeze(0)
#
#     source_cent = source - mu_s
#     target_cent = target - mu_t
#
#     def _calc_cov(feat, w):
#         mu = torch.matmul(feat.t(), w.unsqueeze(1)).t()
#         feat_cent = feat - mu
#         sum_sq = torch.sum(w ** 2)
#         sum_sq = torch.clamp(sum_sq, max=0.9)
#         alpha = 1.0 / (1.0 - sum_sq + 1e-6)
#         alpha = torch.clamp(alpha, max=5.0)
#         feat_w = feat_cent * w.unsqueeze(1)
#         cov = alpha * torch.mm(feat_w.t(), feat_cent)
#         cov = cov + torch.eye(feat.size(1), device=device) * 1e-4
#         return cov
#
#     cov_s = _calc_cov(source_cent, w_s)
#     cov_t = _calc_cov(target_cent, w_t)
#
#
#     loss_coral = torch.sum((cov_s - cov_t) ** 2) /(4 * source.size(1) ** 2)+torch.mean((mu_s - mu_t) ** 2)
#
#     return loss_coral
def unified_weighted_coral_alignment(source, target, target_logits):

    feat_dim = source.size(1)

    # 1. 计算熵权重 (保持不变)
    T = 2.0
    probs = F.softmax(target_logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-5), dim=1)

    w_t = 1.0 + torch.exp(-entropy / T)
    w_t = w_t.detach() / torch.sum(w_t.detach())  # 归一化

    n_s = source.size(0)
    w_s = torch.ones(n_s).to(device) / n_s


    source_weighted = source * torch.sqrt(w_s).unsqueeze(1)
    target_weighted = target * torch.sqrt(w_t).unsqueeze(1)


    moment_s = torch.mm(source_weighted.t(), source_weighted)
    moment_t = torch.mm(target_weighted.t(), target_weighted)

    loss = torch.sum((moment_s - moment_t) ** 2) / (4 * feat_dim ** 2)

    return loss

# ============ 训练函数 ============
def adaption_one_epoch(model_F, model_C, model_D2, target_loader, source_loader, target_val_loader, optimizer,
                       epoch_idx,item_grl_a,item_coral_a,item_grl_w,coral_2_item):
    model_F.train()
    model_C.train()
    model_D2.train()


    infer_model_F = copy.deepcopy(model_F).eval()
    infer_model_C = copy.deepcopy(model_C).eval()

    criterion = nn.CrossEntropyLoss()


    src_iter = iter(source_loader)
    tgt_iter = iter(target_loader)
    max_iter = min(len(source_loader), len(target_loader))

    len_dataloader = min(len(source_loader), len(target_loader))


    for step in tqdm(range(max_iter), desc=f"Ep {epoch_idx}"):
        try:
            src_imgs, src_labels, _ = next(src_iter)
        except StopIteration:
            src_iter = iter(source_loader)
            src_imgs, src_labels, _ = next(src_iter)
        try:
            tgt_imgs, tgt_labels, _ = next(tgt_iter)
        except StopIteration:
            tgt_iter = iter(target_loader)
            tgt_imgs, tgt_labels, _ = next(tgt_iter)

        src_imgs, src_labels = src_imgs.to(device), src_labels.to(device)
        tgt_imgs = tgt_imgs.to(device)

        optimizer.zero_grad()

        # Forward
        src_feat, src_feat2 = model_F(src_imgs)
        target_feat, target_feat2 = model_F(tgt_imgs)


        _,infer_feat=infer_model_F(tgt_imgs)#构建伪标签
        target_logits = infer_model_C(infer_feat)
        tgt_pred_label=target_logits.argmax(dim=1)

        loss_coral = torch.tensor(0.0).to(device)
        valid_cnt = 0


        for c in range(num_classes):

            mask_s = (src_labels == c)

            mask_t = (tgt_pred_label == c)


            s_feat2_c = src_feat2[mask_s]
            t_feat2_c = target_feat2[mask_t]

            s_feat_c = src_feat[mask_s]
            t_feat_c = target_feat[mask_t]

            if s_feat2_c.size(0) < 8 or t_feat2_c.size(0) < 8:
                continue
            t_log=target_logits[mask_t]

            loss_coral += (1-coral_2_item) * unified_weighted_coral_alignment(s_feat2_c, t_feat2_c, t_log)
            loss_coral += coral_2_item * unified_weighted_coral_alignment(s_feat_c, t_feat_c,  t_log)

            valid_cnt += 1


        if valid_cnt > 0:
            loss_coral = loss_coral / valid_cnt



        ad_feat = torch.cat((src_feat2, target_feat2), dim=0)
        ad_out = model_D2(grad_reverse(ad_feat, alpha=1 ))
        ad_labels = torch.cat((torch.ones(src_imgs.size(0), dtype=torch.long, device=device),
                               torch.zeros(tgt_imgs.size(0), dtype=torch.long, device=device)), dim=0)
        loss_dann = criterion(ad_out, ad_labels)
        out_ori = model_C(src_feat2)
        loss_cls = criterion( out_ori, src_labels)
        total_loss = loss_cls +item_grl_a*loss_dann+item_coral_a*loss_coral
        total_loss.backward()
        optimizer.step()

    result = evaluate(model_F, model_C, target_val_loader)
    return result



def mixup_test_data(train_imgs, test_imgs, alpha=0.5):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    mixed_x = lam * train_imgs + (1 - lam) * test_imgs
    return mixed_x, lam


@torch.no_grad()
def evaluate(model_F, model_C, loader, num_classes=5):
    model_F.eval()
    model_C.eval()


    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    for batch in loader:
        data, label, _ = batch
        data, label = data.to(device), label.to(device).long()


        _, feat2 = model_F(data)
        preds = model_C(feat2).argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())


        total_correct += (preds == label).sum().item()
        total_samples += data.size(0)


    accuracy = total_correct / total_samples if total_samples > 0 else 0.0


    if total_samples == 0:
        precision = recall = f1 = 0.0
    else:

        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return accuracy, precision, recall, f1


# ============ 数据处理 ============
def load_subject_data(subj_idx, data_dir, idx):
    x_path = os.path.join(data_dir, f"data_{subj_idx}.npy")
    y_path = os.path.join(data_dir, f"label_{subj_idx}.npy")
    x = np.load(x_path)
    y = np.load(y_path)
    d = np.ones_like(y) * idx
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    d = torch.tensor(d, dtype=torch.long)
    return TensorDataset(x, y, d)


def split_source_target_data(num_subjects, target_subj, data_dir, batch_size):
    source_datasets = []
    idx = 0
    for subj in range(num_subjects):
        if subj != target_subj:
            source_datasets.append(load_subject_data(subj, data_dir, idx=idx))
            idx += 1

    source_x = torch.cat([ds.tensors[0] for ds in source_datasets])
    source_y = torch.cat([ds.tensors[1] for ds in source_datasets])
    source_d = torch.cat([ds.tensors[2] for ds in source_datasets])
    source_full = TensorDataset(source_x, source_y, source_d)

    train_len = int(len(source_full) * 0.8)
    val_len = len(source_full) - train_len
    src_train, src_val = random_split(source_full, [train_len, val_len], generator=torch.Generator().manual_seed(42))

    return (DataLoader(src_train, batch_size=batch_size, shuffle=True, drop_last=True),
            DataLoader(src_val, batch_size=batch_size, shuffle=False),
            load_subject_data(target_subj, data_dir, idx=9))


def mysplit(target_dataset, batch_size):
    total_len = len(target_dataset)
    train_len = int(total_len * 0.8)
    val_len = total_len - train_len
    tgt_train, tgt_val = random_split(target_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))
    return (DataLoader(tgt_train, batch_size=batch_size, shuffle=True, drop_last=True),
            DataLoader(tgt_val, batch_size=batch_size, shuffle=False))


# ============ main ============
if __name__ == "__main__":
    results = []
    save_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    for target_subj in range(num_subjects):

        print(f"\n=== cross-subject ,target: {target_subj} / total {num_subjects} ===")

        # 1. split
        source_train_loader, source_val_loader, target_dataset = split_source_target_data(
            num_subjects, target_subj, data_dir, batch_size
        )
        target_train_loader, target_val_loader = mysplit(target_dataset, batch_size)

        # 2. init
        model_F = FeatureExtractor.MCADNNFeatureExtractor(bottleneck_dim=feat_dim,in_channels=14).to(device)
        model_C = FeatureExtractor.Classifier(feat_dim=feat_dim, num_classes=num_classes).to(device)
        model_D = FeatureExtractor.Classifier(feat_dim=feat_dim, num_classes=num_subjects - 1).to(device)
        model_D2 = FeatureExtractor.Classifier(feat_dim=feat_dim, num_classes=2).to(device)

        # 3. pretrain
        print("\n--- pre-train ---")
        pretrain_optimizer = torch.optim.Adam([
            {'params': model_F.parameters(), 'lr': 1e-3},
            {'params': model_C.parameters(), 'lr': 1e-3},
            {'params': model_D.parameters(), 'lr': 1e-3}
        ])

        best_src_val_acc = 0.0
        best_src_precision = 0.0
        best_src_recall = 0.0
        best_src_f1 = 0.0
        best_pretrain_weights = None

        for epoch in range(pretrain_epochs):
            train_acc, train_loss = pretrain_one_epoch(
                model_F, model_C, model_D, source_train_loader, pretrain_optimizer, epoch
            )
            val_acc, val_precision, val_recall, val_f1 = evaluate(model_F, model_C, target_train_loader)

            if val_acc > best_src_val_acc:
                best_src_val_acc = val_acc
                best_src_precision= val_precision
                best_src_recall = val_recall
                best_src_f1 = val_f1

                best_pretrain_weights = {
                    'F': copy.deepcopy(model_F.state_dict()),
                    'C': copy.deepcopy(model_C.state_dict())
                }

                save_path_F = os.path.join(model_save_dir, f"pretrain_best_F_tgt{target_subj}.pth")
                save_path_C = os.path.join(model_save_dir, f"pretrain_best_C_tgt{target_subj}.pth")
                torch.save(model_F.state_dict(), save_path_F)
                torch.save(model_C.state_dict(), save_path_C)

            print(
                f"[Pretrain {epoch + 1:02d}] Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f} | Best: {best_src_val_acc:.3f}")

        print(f"best_val_Acc: {best_src_val_acc:.3f}")
        print(f"best_val_Precision: {best_src_precision:.3f}")
        print(f"best_val_Recall: {best_src_recall:.3f}")
        print(f"best_val_F1: {best_src_f1:.3f}")

        optimizer_ad = torch.optim.Adam([
            {'params': model_F.parameters(), 'lr': 1e-3},
            {'params': model_C.parameters(), 'lr': 1e-3},
            {'params': model_D2.parameters(), 'lr': 1e-3},
        ])
        best_test_acc = 0.0
        best_pre = 0.0
        best_recall = 0.0
        best_f1 = 0.0
        for epoch in range(adapt_epochs):

            metrix = adaption_one_epoch(
                model_F, model_C, model_D2,
                target_train_loader, source_train_loader, target_val_loader,
                optimizer_ad,adapt_epochs,item_grl_a=1, item_coral_a=1, item_grl_w=1,
                coral_2_item=0.3
            )
            test_acc = metrix[0]
            test_pre = metrix[1]
            test_recall = metrix[2]
            test_f1 = metrix[3]


            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_pre = test_pre
                best_recall = test_recall
                best_f1 = test_f1
                print(f"Ep {epoch + 1}: {test_acc:.3f} (Best: {best_test_acc:.3f})")
                print(f"Ep {epoch + 1}: {test_pre:.3f} (Best: {best_test_acc:.3f})")
                print(f"Ep {epoch + 1}: {test_recall:.3f} (Best: {best_test_acc:.3f})")
                print(f"Ep {epoch + 1}: {test_f1:.3f} (Best: {best_test_acc:.3f})")




