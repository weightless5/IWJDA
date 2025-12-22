import copy
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm
from datetime import datetime
from feature_net import FeatureExtractor
from torch.autograd import Function
import warnings
warnings.filterwarnings("ignore")
import torch
from sklearn.metrics import precision_score, recall_score, f1_score  # 也可手动实现，见下文
# ===================== 配置区域 =====================
pretrained_dir = r"E:\CODE\IWJDA\ODG\model_save_32"
# pretrained_dir = r"E:\HLW\CODE\1208IWJDA\目标域迁移测试\ori_model"
pretrained_timestamp = "2025_160946"
# data_dir = r"E:\002HLW\code\IWJDA\dataset"
data_dir = r"E:\Cross_subject\data"

torch.manual_seed(42)#42
np.random.seed(42)#42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_subjects = 9
num_classes = 7
batch_size = 128
adapt_epochs = 25
feat_dim = 32


buffer_size = 128


# ================= 工具函数 (保持不变) =================
def mixup_test_data(train_imgs, test_imgs, alpha=0.5):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    mixed_x = lam * train_imgs + (1 - lam) * test_imgs
    return mixed_x, lam


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


def grad_reverse(x, alpha=1.0):
    return GradReverse.apply(x, alpha)


# ================= Replay Buffer =================
class ClassAwareReplayBuffer:
    def __init__(self, num_classes, max_size=64):
        self.num_classes = num_classes
        self.max_size = max_size
        self.buffer = {c: {'source': [], 'target': []} for c in range(num_classes)}

    def add_source(self, imgs, labels):
        imgs = imgs.detach()
        labels = labels.detach()
        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.sum() == 0: continue
            imgs_c = imgs[mask]
            for i in range(imgs_c.size(0)):
                if len(self.buffer[c]['source']) < self.max_size:
                    self.buffer[c]['source'].append(imgs_c[i])

    def add_target(self, imgs, pred_labels, confs, threshold=0.75):
        imgs = imgs.detach()
        pred_labels = pred_labels.detach()
        confs = confs.detach()
        for c in range(self.num_classes):
            mask = (pred_labels == c) & (confs > threshold)
            if mask.sum() == 0: continue
            imgs_c = imgs[mask]
            for i in range(imgs_c.size(0)):
                if len(self.buffer[c]['target']) < self.max_size:
                    self.buffer[c]['target'].append(imgs_c[i])

    def is_ready(self, class_idx):
        # 只要达到一半就允许计算，保证初期能跑起来
        limit = int(self.max_size * 0.8)
        return (len(self.buffer[class_idx]['source']) >= limit) and \
            (len(self.buffer[class_idx]['target']) >= limit)

    def get_and_clear(self, class_idx):
        src_imgs = torch.stack(self.buffer[class_idx]['source'])
        tgt_imgs = torch.stack(self.buffer[class_idx]['target'])
        self.buffer[class_idx]['source'] = []
        self.buffer[class_idx]['target'] = []
        return src_imgs, tgt_imgs


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


#
#
# def entropy_weighted_moment_matching(source, target, target_logits):
#
#     T = 0.5
#     probs = F.softmax(target_logits, dim=1)
#     entropy = -torch.sum(probs * torch.log(probs + 1e-5), dim=1)
#
#     # 目标域权重
#     w_t = 1.0 + torch.exp(-entropy / T)#熵加权
#     # n_t = target.size(0)
#     # w_t = torch.ones(n_t).to(device) / n_t
#     w_t = w_t.detach()  # 记得detach
#     w_t = w_t / torch.sum(w_t)
#
#     # 源域权重 (均匀分布)
#     n_s = source.size(0)
#     w_s = torch.ones(n_s).to(device) / n_s
#
#     # 2. 【新增】计算加权均值 (Mean Alignment)
#     # source: [N, D], w_s: [N] -> mu_s: [D]
#     mu_s = torch.matmul(w_s.unsqueeze(0), source).squeeze(0)
#     mu_t = torch.matmul(w_t.unsqueeze(0), target).squeeze(0)
#
#     # 均值 Loss (一阶矩)
#     # loss_mean = torch.mean((mu_s - mu_t) ** 2)
#
#     # 3. 计算加权协方差 (CORAL, 二阶矩)
#     # 中心化数据 (利用刚才算出的 mu)
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
#     # loss_coral = torch.sum((cov_s - cov_t) ** 2) / (4 * source.size(1) ** 2)
#
#
#     return loss_coral



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
            # --- A. 筛选样本 (Masking) ---
            # 源域：取出真实标签为 c 的样本
            mask_s = (src_labels == c)
            # 目标域：取出伪标签为 c 的样本
            mask_t = (tgt_pred_label == c)

            # 获取对应的特征 (深层 feat2 和 浅层 feat)
            s_feat2_c = src_feat2[mask_s]
            t_feat2_c = target_feat2[mask_t]

            s_feat_c = src_feat[mask_s]
            t_feat_c = target_feat[mask_t]

            # 如果某一边样本少于2个，无法计算协方差，跳过该类
            if s_feat2_c.size(0) < 2 or t_feat2_c.size(0) < 2:
                continue
            t_log=target_logits[mask_t]


            loss_coral += (1-coral_2_item) * unified_weighted_coral_alignment(s_feat2_c, t_feat2_c, t_log)
            loss_coral += coral_2_item * unified_weighted_coral_alignment(s_feat_c, t_feat_c,  t_log)


            valid_cnt += 1

            # --- B. 样本数量检查 ---


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


# ============ 辅助函数: 数据集加载 ============
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
            idx = idx + 1
    source_x = torch.cat([ds.tensors[0] for ds in source_datasets])
    source_y = torch.cat([ds.tensors[1] for ds in source_datasets])
    source_d = torch.cat([ds.tensors[2] for ds in source_datasets])
    source_full_dataset = TensorDataset(source_x, source_y, source_d)

    src_train_len = int(len(source_full_dataset) * 0.8)
    src_val_len = len(source_full_dataset) - src_train_len
    src_train_dataset, src_val_dataset = random_split(
        source_full_dataset, [src_train_len, src_val_len],
        generator=torch.Generator().manual_seed(42)
    )
    # drop_last=True 保证 batch 不会太小
    source_train_loader = DataLoader(src_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    source_val_loader = DataLoader(src_val_dataset, batch_size=batch_size, shuffle=False)

    target_dataset = load_subject_data(target_subj, data_dir, idx=9)
    # drop_last=True
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return source_train_loader, source_val_loader, target_loader


def mysplit(target_loader):
    target_dataset = target_loader.dataset
    total_len = len(target_dataset)
    train_len = int(total_len * 0.8)
    val_len = total_len - train_len
    target_train_dataset, target_val_dataset = random_split(
        target_dataset, [train_len, val_len],
        generator=torch.Generator().manual_seed(42)
    )
    target_train_loader = DataLoader(target_train_dataset, batch_size=target_loader.batch_size, shuffle=True,
                                     drop_last=True)
    target_val_loader = DataLoader(target_val_dataset, batch_size=target_loader.batch_size, shuffle=True)
    return target_train_loader, target_val_loader


@torch.no_grad()
def evaluate(model_F, model_C, loader, num_classes=5):  # num_classes改为你的实际类别数（映射后是5类）
    model_F.eval()
    model_C.eval()

    # 初始化变量
    total_correct = 0
    total_samples = 0
    all_preds = []  # 存储所有预测标签
    all_labels = []  # 存储所有真实标签

    for batch in loader:
        data, label, _ = batch
        data, label = data.to(device), label.to(device).long()

        # 前向传播
        _, feat2 = model_F(data)
        preds = model_C(feat2).argmax(dim=1)

        # 收集结果（转到CPU避免CUDA内存累积）
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

        # 计算准确率
        total_correct += (preds == label).sum().item()
        total_samples += data.size(0)

    # 基础准确率
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    # 计算多分类宏平均Precision、Recall、F1（处理空类别，zero_division=0避免除零错误）
    if total_samples == 0:
        precision = recall = f1 = 0.0
    else:
        # macro：每个类别单独计算，再取算术平均；micro：全局计算TP/FP/FN
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return accuracy, precision, recall, f1


if __name__ == "__main__":
    results = []

    grl_a = [1]
    coral_a=[1]
    grl_w=[1]

    coral_2=[0.3]


    acc_a=[]


    for coral_2_item in coral_2:
        for item_grl_a in grl_a:#对抗损失权重
            for item_coral_a in coral_a:
                for item_grl_w in grl_w:#梯度反转系数
                    averge_acc = 0.0

                    for target_subj in range(num_subjects):
                        # if(target_subj!=4):continue
                        print(f"\n=== User {target_subj} ===")
                        # ... (加载数据逻辑) ...
                        source_train_loader, source_val_loader, target_loader = split_source_target_data(
                            num_subjects=num_subjects, target_subj=target_subj, data_dir=data_dir, batch_size=batch_size
                        )
                        target_train_loader, target_val_loader = mysplit(target_loader)

                        model_F = FeatureExtractor.MCADNNFeatureExtractor(bottleneck_dim=feat_dim).to(device)
                        model_C = FeatureExtractor.Classifier(feat_dim=feat_dim, num_classes=num_classes).to(device)
                        model_D2 = FeatureExtractor.Classifier(feat_dim=feat_dim, num_classes=2).to(device)

                        # 加载预训练
                        f_name = f"pretrain_best_F_tgt{target_subj}.pth"
                        c_name = f"pretrain_best_C_tgt{target_subj}.pth"
                        f_path = os.path.join(pretrained_dir, f_name)
                        c_path = os.path.join(pretrained_dir, c_name)


                        if os.path.exists(f_path):
                            model_F.load_state_dict(torch.load(f_path, map_location=device))
                            model_C.load_state_dict(torch.load(c_path, map_location=device))
                        else:
                            continue

                        optimizer = torch.optim.Adam([
                            {'params': model_F.parameters(), 'lr': 1e-3},
                            {'params': model_C.parameters(), 'lr': 1e-3},
                            {'params': model_D2.parameters(), 'lr': 1e-3},
                        ])

                        best_test_acc = 0.0
                        for epoch in range(adapt_epochs):
                            # 传入 epoch_idx 用于 warm-up
                            metrix = adaption_one_epoch(
                                model_F, model_C, model_D2,
                                target_train_loader, source_train_loader, target_val_loader,
                                optimizer, epoch,item_grl_a=item_grl_a,item_coral_a=item_coral_a, item_grl_w=item_grl_w,coral_2_item=coral_2_item
                            )
                            test_acc=metrix[0]
                            if test_acc > best_test_acc: best_test_acc = test_acc
                            print(f"Ep {epoch + 1}: {test_acc:.3f} (Best: {best_test_acc:.3f})")

                        results.append({"User": target_subj, "Best": best_test_acc,"f1": metrix[3]})
                        averge_acc+=best_test_acc

                        acc_a.append({"User": target_subj,"item_grl_w":item_grl_w,"item_grl":item_grl_a,"item_coral": item_coral_a, "Best": best_test_acc,"f1": metrix[3]})
                    print( acc_a)
                    # 保存...

                        # df = pd.DataFrame(results)
                        # save_path = os.path.join(pretrained_dir, f"IWJDA-local-{datetime.now().strftime('%H%M%S')}.xlsx")
                    acc_a.append({"User": target_subj,"item_grl_w":item_grl_w, "item_grl": item_grl_a, "item_coral": item_coral_a,
                                  "avg_Best": averge_acc / num_subjects})


    df2 = pd.DataFrame(acc_a)
    save_path = os.path.join(pretrained_dir, f"load_ori_iwjda{datetime.now().strftime('%H%M%S')}.xlsx")

    print(acc_a)

    df2.to_excel(save_path, index=False)