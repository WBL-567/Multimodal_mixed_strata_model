import os
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt

import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用 GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 在 main 函数开始时调用
set_seed(42)

# 设置中文字体，避免乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 超参数配置（已放松） =====================
EARLY_STOP_PATIENCE = 12      # 早停耐心值（适当增大）
MIN_DELTA = 0.001             # 验证损失显著下降的最小阈值（放宽）
MAX_EPOCHS = 100              # 最大训练轮次（增加）
LEARNING_RATE = 1e-4          # 学习率保持不变
WEIGHT_DECAY = 1e-4           # 权重衰减
GRAD_CLIP_NORM = 3.0          # 梯度裁剪（适度放宽）

# ===================== 路径配置 =====================
base_path = os.path.dirname(__file__)
prediction_dir = os.path.dirname(base_path)

DATA_DIR = os.path.join(base_path, 'data')
MODEL_SAVE_PATH = os.path.join(base_path, 'final_model.pth')
SCALER_PATH = os.path.join(base_path, 'final_scaler.pkl')
LOG_DIR = os.path.join(base_path, 'logs')
RESULTS_DIR = os.path.join(base_path, 'results')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ===================== 配置：Excel列名（6个特征）=====================
FEATURE_COLS = [
    "推进速度",
    "扭矩",
    "推力",
    "竖向振动加速度RMS",
    "横向振动加速度RMS",
    "轴向振动加速度RMS",
]
LABEL_COL = "地层分类"
IMAGE_SUBDIRS = ['11', '22', '33']
CLASS_NAMES = ['HSS', 'DMS', 'TMS']

# ===================== 温和版三分类配置 =====================
USE_ENSEMBLE = False          # 当前阶段默认关闭集成评估
BATCH_SIZE = 32

# ===================== 图像预处理函数 =====================
def build_train_transform():
    """训练集使用的轻度增强"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(
            degrees=3,
            translate=(0.02, 0.02),
            scale=(0.98, 1.02),
            fill=0
        ),
        transforms.ColorJitter(
            brightness=0.08,
            contrast=0.08
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

def build_eval_transform():
    """验证 / 测试 / 重要性分析：统一预处理"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


# ===================== 数据集 =====================
class TunnellingDataset(Dataset):
    def __init__(self, params, labels, indices, image_dirs, transform=None):
        self.params = params
        self.labels = labels
        self.indices = indices
        self.image_dirs = [os.path.abspath(p) for p in image_dirs]
        for path in self.image_dirs:
            if not os.path.exists(path):
                raise NotADirectoryError(f"图像目录不存在: {path}")

        self.image_files = [f"{idx + 1}.png" for idx in indices]
        self.transform = transform

        self.missing_count = 0
        for i, img_file in enumerate(self.image_files):
            for sub_dir in self.image_dirs:
                img_path = os.path.join(sub_dir, img_file)
                if not os.path.exists(img_path):
                    self.missing_count += 1
        if self.missing_count > 0:
            print(f"警告: 数据集共有 {self.missing_count}/{len(indices)*3} 个图像缺失")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        images = []
        for img_dir in self.image_dirs:
            img_path = os.path.join(img_dir, self.image_files[idx])
            try:
                img = Image.open(img_path).convert('L')
            except Exception:
                img = Image.new('L', (224, 224))
            if self.transform:
                img = self.transform(img)
            images.append(img)

        return (
            torch.tensor(self.params[idx], dtype=torch.float32),
            *images,
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

# ===================== 日志记录器 =====================
class EnhancedLogger:
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'lr': []
        }
        self.csv_path = os.path.join(log_dir, 'training_history.csv')

    def log_metrics(self, epoch, metrics: dict):
        self.writer.add_scalars('Loss', {'train': metrics['train_loss'], 'val': metrics['val_loss']}, epoch)
        self.writer.add_scalars('Accuracy', {'train': metrics['train_acc'], 'val': metrics['val_acc']}, epoch)
        self.writer.add_scalar('Learning Rate', metrics['lr'], epoch)

        for k in self.history:
            self.history[k].append(metrics.get(k, None))

        pd.DataFrame(self.history).to_csv(self.csv_path, index=False)

    def close(self):
        self.writer.close()

    def plot_curves(self, save_path):
        epochs = range(1, len(self.history['train_loss']) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(epochs, self.history['train_loss'], label='Train Loss')
        ax1.plot(epochs, self.history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.set_title('Loss Curves')
        ax1.grid(True)

        ax2.plot(epochs, self.history['train_acc'], label='Train Acc')
        ax2.plot(epochs, self.history['val_acc'], label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.set_title('Accuracy Curves')
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        print(f"学习曲线已保存为 PNG: {save_path}")
        base, _ = os.path.splitext(save_path)
        try:
            plt.savefig(base + '.emf', format='emf')
            print(f"学习曲线已保存为 EMF: {base}.emf")
        except Exception as e:
            print(f"无法保存 EMF 格式 (将保存为 SVG): {e}")
            plt.savefig(base + '.svg', format='svg')
            print(f"学习曲线已保存为 SVG: {base}.svg")
        plt.close()

    def save_history_to_excel(self, excel_path):
        df = pd.DataFrame(self.history)
        df.to_excel(excel_path, index=False)
        print(f"训练历史已保存为 Excel: {excel_path}")

# ===================== 模型（回退到初投稿版本，保持 Fig.4 一致） =====================
class MultimodalModel(nn.Module):
    def __init__(self, param_input_size: int, num_classes: int = 3):
        super().__init__()
        self.enable_param = True
        self.enable_image = True

        # ---------------- 参数分支：回到原稿版 ----------------
        # 原稿：单隐层 MLP，128 维，ReLU，Dropout=0.5
        self.param_fc = nn.Sequential(
            nn.Linear(param_input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # ---------------- 图像分支：回到原稿版 ----------------
        # 原稿：两层卷积 + 两次池化，共享卷积权重
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # [B,32,224,224]
            nn.ReLU(),
            nn.MaxPool2d(2),                              # [B,32,112,112]

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # [B,64,112,112]
            nn.ReLU(),
            nn.MaxPool2d(2)                               # [B,64,56,56]
        )

        # 每个图像分支编码为 128 维
        self.image_fc = nn.Sequential(
            nn.Linear(64 * 56 * 56, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # ---------------- 融合层：回到原稿版 ----------------
        # 参数分支 128 + 三个图像分支各 128 = 512
        self.fusion_input_dim = 128 * 4

        self.fusion = nn.Sequential(
            nn.Linear(self.fusion_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        # 保留初始化，但不再使用 BN / AdaptiveAvgPool
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def process_image(self, img):
        x = self.conv_block(img)
        x = x.flatten(1)
        x = self.image_fc(x)
        return x

    def forward(self, param_data, *images):
        if not (self.enable_param or self.enable_image):
            raise ValueError("至少需要启用一个模态（参数或图像）")

        features = []

        # 参数分支
        if self.enable_param:
            param_feature = self.param_fc(param_data)
        else:
            param_feature = torch.zeros(
                param_data.size(0), 128, device=param_data.device
            )
        features.append(param_feature)

        # 图像分支（三张图，共享卷积块）
        img1, img2, img3 = images
        if self.enable_image:
            img1_feature = self.process_image(img1)
            img2_feature = self.process_image(img2)
            img3_feature = self.process_image(img3)
        else:
            batch_size = param_data.size(0)
            zero_feature = torch.zeros(batch_size, 128, device=param_data.device)
            img1_feature = zero_feature
            img2_feature = zero_feature
            img3_feature = zero_feature

        features.extend([img1_feature, img2_feature, img3_feature])

        combined = torch.cat(features, dim=1)
        return self.fusion(combined)

# ===================== 基础工具函数（与之前相同） =====================
def save_dataframe_multi(df, base_path_no_ext):
    csv_path = base_path_no_ext + '.csv'
    xlsx_path = base_path_no_ext + '.xlsx'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    df.to_excel(xlsx_path, index=False)
    print(f"已保存: {csv_path}")
    print(f"已保存: {xlsx_path}")

def save_text(content, txt_path):
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"已保存文本: {txt_path}")

def save_confusion_matrix_table(cm, class_names, base_path_no_ext):
    df = pd.DataFrame(cm, index=[f"True_{c}" for c in class_names], columns=[f"Pred_{c}" for c in class_names])
    save_dataframe_multi(df.reset_index().rename(columns={'index': 'Label'}), base_path_no_ext)

def classification_report_to_dataframe(y_true, y_pred, class_names):
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    rows = []
    for label, metrics in report_dict.items():
        if isinstance(metrics, dict):
            row = {'label': label}
            row.update(metrics)
            rows.append(row)
        else:
            rows.append({'label': label, 'value': metrics})
    return pd.DataFrame(rows)

def save_prediction_details(y_true, y_pred, indices, class_names, base_path_no_ext):
    true_names = [class_names[int(i)] for i in y_true]
    pred_names = [class_names[int(i)] for i in y_pred]
    df = pd.DataFrame({
        'original_index': indices,
        'true_label_id': y_true,
        'pred_label_id': y_pred,
        'true_label_name': true_names,
        'pred_label_name': pred_names,
        'correct': [int(t == p) for t, p in zip(y_true, y_pred)]
    })
    save_dataframe_multi(df, base_path_no_ext)

def plot_and_save_bar(values, labels, title, ylabel, base_path_no_ext):
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30)
    plt.tight_layout()
    png_path = base_path_no_ext + '.png'
    plt.savefig(png_path, dpi=200)
    print(f"图已保存为 PNG: {png_path}")
    try:
        emf_path = base_path_no_ext + '.emf'
        plt.savefig(emf_path, format='emf')
        print(f"图已保存为 EMF: {emf_path}")
    except Exception as e:
        print(f"无法保存 EMF 格式 (将保存为 SVG): {e}")
        svg_path = base_path_no_ext + '.svg'
        plt.savefig(svg_path, format='svg')
        print(f"图已保存为 SVG: {svg_path}")
    plt.close()

# ===================== 训练/评估函数 =====================
def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip_norm=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for params, img1, img2, img3, labels in loader:
        params = params.to(device)
        img1 = img1.to(device)
        img2 = img2.to(device)
        img3 = img3.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(params, img1, img2, img3)
        loss = criterion(outputs, labels)
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_true = []
    all_pred = []

    for params, img1, img2, img3, labels in loader:
        params = params.to(device)
        img1 = img1.to(device)
        img2 = img2.to(device)
        img3 = img3.to(device)
        labels = labels.to(device)

        outputs = model(params, img1, img2, img3)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_true.extend(labels.cpu().numpy())
        all_pred.extend(predicted.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total

    from sklearn.metrics import confusion_matrix
    matrix = confusion_matrix(all_true, all_pred)
    return epoch_loss, epoch_acc, matrix, all_true, all_pred

def save_confusion_matrix(matrix, class_names, save_path):
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap='Blues', ax=ax, colorbar=False)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"混淆矩阵已保存为 PNG: {save_path}")
    base, _ = os.path.splitext(save_path)
    try:
        plt.savefig(base + '.emf', format='emf')
        print(f"混淆矩阵已保存为 EMF: {base}.emf")
    except Exception as e:
        print(f"无法保存 EMF 格式 (将保存为 SVG): {e}")
        plt.savefig(base + '.svg', format='svg')
        print(f"混淆矩阵已保存为 SVG: {base}.svg")
    plt.close()

# ===================== 消融实验与重要性分析 =====================
def evaluate_modalities(model, loader, device, enable_param: bool, enable_image: bool):
    prev_param = model.enable_param
    prev_image = model.enable_image
    model.enable_param = enable_param
    model.enable_image = enable_image

    criterion = nn.CrossEntropyLoss()
    _, acc, _, _, _ = evaluate(model, loader, criterion, device)

    model.enable_param = prev_param
    model.enable_image = prev_image
    return acc

def cross_modality_ablation(model, test_loader, device):
    results = {
        'both_modalities': evaluate_modalities(model, test_loader, device, True, True),
        'params_only': evaluate_modalities(model, test_loader, device, True, False),
        'images_only': evaluate_modalities(model, test_loader, device, False, True),
    }
    print("\n跨模态消融实验结果：")
    for k, v in results.items():
        print(f"{k}: {v:.2f}%")
    return results

def enhanced_feature_importance(model, params_test, labels_test, test_idx, image_dirs, device, scaler=None):
    # 重要性分析统一使用 eval_transform（带 Normalize）
    eval_transform = build_eval_transform()

    criterion = nn.CrossEntropyLoss()

    # 参数敏感性分析：只启用参数模态，关闭图像模态
    model.enable_param = True
    model.enable_image = False

    # 基线数据集：原始参数 + 标准预处理
    base_dataset = TunnellingDataset(params_test, labels_test, test_idx, image_dirs, eval_transform)
    base_loader = DataLoader(base_dataset, batch_size=32, shuffle=False, num_workers=0)
    _, base_acc, _, _, _ = evaluate(model, base_loader, criterion, device)

    print(f"参数模态基线准确率: {base_acc:.4f}%")

    sensitivity_scores = []

    for j in range(params_test.shape[1]):
        # 打乱第 j 个特征
        params_shuffled = params_test.copy()
        np.random.shuffle(params_shuffled[:, j])

        # 打乱后的数据集使用相同的预处理
        dataset = TunnellingDataset(params_shuffled, labels_test, test_idx, image_dirs, eval_transform)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
        _, acc, _, _, _ = evaluate(model, loader, criterion, device)

        # 取绝对变化量，保证敏感性得分为非负值
        sensitivity_score = abs(base_acc - acc)

        sensitivity_scores.append(sensitivity_score)

        print(
            f"特征 {FEATURE_COLS[j]} 打乱后准确率: {acc:.4f}%, "
            f"敏感性得分: {sensitivity_score:.4f}"
        )

    # 恢复完整多模态状态
    model.enable_param = True
    model.enable_image = True

    return sensitivity_scores

@torch.no_grad()
def calculate_image_importance_accurate(model, test_loader, device):
    criterion = nn.CrossEntropyLoss()
    base_loss, base_acc, _, _, _ = evaluate(model, test_loader, criterion, device)

    importance = []
    for img_zero_idx in range(3):
        model.eval()
        running_correct = 0
        total = 0
        for params, img1, img2, img3, labels in test_loader:
            params = params.to(device)
            img1 = img1.to(device)
            img2 = img2.to(device)
            img3 = img3.to(device)
            labels = labels.to(device)

            imgs = [img1, img2, img3]
            imgs[img_zero_idx] = torch.zeros_like(imgs[img_zero_idx])

            outputs = model(params, imgs[0], imgs[1], imgs[2])
            _, predicted = outputs.max(1)
            total += labels.size(0)
            running_correct += predicted.eq(labels).sum().item()

        acc = 100. * running_correct / total
        drop = base_acc - acc
        importance.append(drop)
        print(f"第 {img_zero_idx+1} 张图像置零后，精度下降: {drop:.4f}")

    return importance

def sensitivity_analysis_pipeline(model, params_test, labels_test, test_idx, image_dirs, test_loader, device):
    print("\n==================== 开始敏感性分析 ====================")

    feature_importance = enhanced_feature_importance(
        model, params_test, labels_test, test_idx, image_dirs, device
    )
    fi_df = pd.DataFrame({
        'feature': FEATURE_COLS,
        'importance_drop_acc': feature_importance
    })
    save_dataframe_multi(fi_df, os.path.join(RESULTS_DIR, 'feature_importance_final'))

    ablation_results = cross_modality_ablation(model, test_loader, device)
    ab_df = pd.DataFrame([
        {'modality': k, 'accuracy': v} for k, v in ablation_results.items()
    ])
    save_dataframe_multi(ab_df, os.path.join(RESULTS_DIR, 'modality_ablation_results'))
    plot_and_save_bar(
        values=[ablation_results['both_modalities'], ablation_results['params_only'], ablation_results['images_only']],
        labels=['Both', 'Params', 'Images'],
        title='Cross-modality Ablation',
        ylabel='Accuracy (%)',
        base_path_no_ext=os.path.join(RESULTS_DIR, 'modality_ablation_results')
    )

    model.enable_param = True
    model.enable_image = True
    importance_with_param = calculate_image_importance_accurate(model, test_loader, device)
    total_importance = sum(importance_with_param) if sum(importance_with_param) > 0 else 1e-12
    importance_with_param_pct = [v / total_importance * 100 for v in importance_with_param]
    img_df_with = pd.DataFrame({
        'image_channel': IMAGE_SUBDIRS,
        'importance_drop_acc': importance_with_param,
        'importance_percent': importance_with_param_pct
    })
    save_dataframe_multi(img_df_with, os.path.join(RESULTS_DIR, 'image_channel_importance_with_param'))
    plot_and_save_bar(
        values=importance_with_param_pct,
        labels=IMAGE_SUBDIRS,
        title='Image Channel Importance (with params)',
        ylabel='Contribution (%)',
        base_path_no_ext=os.path.join(RESULTS_DIR, 'image_channel_importance_with_param')
    )

    model.enable_param = False
    model.enable_image = True
    importance_without_param = calculate_image_importance_accurate(model, test_loader, device)
    total_importance2 = sum(importance_without_param) if sum(importance_without_param) > 0 else 1e-12
    importance_without_param_pct = [v / total_importance2 * 100 for v in importance_without_param]
    img_df_without = pd.DataFrame({
        'image_channel': IMAGE_SUBDIRS,
        'importance_drop_acc': importance_without_param,
        'importance_percent': importance_without_param_pct
    })
    save_dataframe_multi(img_df_without, os.path.join(RESULTS_DIR, 'image_channel_importance_without_param'))
    plot_and_save_bar(
        values=importance_without_param_pct,
        labels=IMAGE_SUBDIRS,
        title='Image Channel Importance (images only)',
        ylabel='Contribution (%)',
        base_path_no_ext=os.path.join(RESULTS_DIR, 'image_channel_importance_without_param')
    )

    model.enable_param = True
    model.enable_image = True

    print("==================== 敏感性分析完成 ====================\n")
    return {
        'feature_importance': feature_importance,
        'ablation_results': ablation_results,
        'image_importance_with_param': importance_with_param_pct,
        'image_importance_without_param': importance_without_param_pct,
    }

# ===================== 集成预测函数 =====================
def ensemble_predict(model_paths, test_loader, device, num_classes):
    """集成多个折的最佳模型，对测试集进行概率平均"""
    all_true = []
    for params, img1, img2, img3, labels in test_loader:
        all_true.extend(labels.numpy())

    total_samples = len(test_loader.dataset)
    cumulative_probs = torch.zeros((total_samples, num_classes))

    for path in model_paths:
        model = MultimodalModel(param_input_size=len(FEATURE_COLS), num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        start_idx = 0
        with torch.no_grad():
            for params, img1, img2, img3, labels in test_loader:
                batch_size = labels.size(0)
                params = params.to(device)
                img1 = img1.to(device)
                img2 = img2.to(device)
                img3 = img3.to(device)
                outputs = model(params, img1, img2, img3)
                probs = torch.softmax(outputs, dim=1).cpu()
                cumulative_probs[start_idx:start_idx + batch_size] += probs
                start_idx += batch_size
        print(f"已集成模型: {os.path.basename(path)}")

    avg_probs = cumulative_probs / len(model_paths)
    all_pred = avg_probs.argmax(dim=1).numpy()
    return all_true, all_pred, avg_probs

# ===================== 主流程 =====================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"当前配置 | USE_ENSEMBLE={USE_ENSEMBLE}, BATCH_SIZE={BATCH_SIZE}")
    print("当前策略 | 温和类权重(1/sqrt(freq)) + 无sampler + 无重增强 + 保留Normalize")

    # ---------- 数据读取 ----------
    excel_path = os.path.join(prediction_dir, "tunnelling_parameter_1.xlsx")
    if not os.path.exists(excel_path):
        print(f"错误: 未找到 Excel 文件 {excel_path}")
        return
    df = pd.read_excel(excel_path)
    required_cols = FEATURE_COLS + [LABEL_COL]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"错误: Excel 缺少列: {missing_cols}")
        return

    data = df[FEATURE_COLS].values
    labels = df[LABEL_COL].values.astype(int)
    original_indices = np.arange(len(df))

    print("\n[标签分布] 原始数据:", dict(Counter(labels)))

    # ---------- 图像路径检查 ----------
    image_dirs = []
    for sub in IMAGE_SUBDIRS:
        dir_path = os.path.join(prediction_dir, sub)
        if not os.path.isdir(dir_path):
            print(f"错误: 缺少图像子目录 {dir_path}")
            return
        image_dirs.append(dir_path)

    # ---------- 类别分布图 ----------
    plt.figure(figsize=(12, 4))
    plt.scatter(original_indices, labels, s=8)
    plt.yticks([0, 1, 2], CLASS_NAMES)
    plt.xlabel('Sample Index (Time Order)')
    plt.ylabel('Class')
    plt.title('Class Distribution Over Time')
    plt.grid(True)
    dist_png = os.path.join(RESULTS_DIR, 'class_distribution_over_time.png')
    plt.tight_layout()
    plt.savefig(dist_png, dpi=200)
    print(f"类别分布图已保存为 PNG: {dist_png}")
    base, _ = os.path.splitext(dist_png)
    try:
        plt.savefig(base + '.emf', format='emf')
        print(f"类别分布图已保存为 EMF: {base}.emf")
    except Exception as e:
        print(f"无法保存 EMF 格式 (将保存为 SVG): {e}")
        plt.savefig(base + '.svg', format='svg')
        print(f"类别分布图已保存为 SVG: {base}.svg")
    plt.close()

    dist_df = pd.DataFrame({
        'sample_index': original_indices,
        'label_id': labels,
        'label_name': [CLASS_NAMES[int(i)] for i in labels]
    })
    save_dataframe_multi(dist_df, os.path.join(RESULTS_DIR, 'class_distribution_over_time'))

    # ---------- 图像预处理：训练用轻度增强，验证/测试用统一预处理 ----------
    transform_train = build_train_transform()
    transform_val = build_eval_transform()

    # ---------- 连续时间块交叉验证 ----------
    K_FOLDS = 10
    valid_folds = range(1, K_FOLDS - 1)
    all_fold_acc = []
    fold_summaries = []
    ensemble_model_paths = []  # 收集每折最佳模型路径

    for fold in valid_folds:
        print("\n" + "="*50)
        print(f"开始第 {fold} 折训练...")
        print("="*50)

        train_idx, val_idx, test_idx = [], [], []
        for c in np.unique(labels):
            class_idx = original_indices[labels == c]
            blocks = np.array_split(class_idx, K_FOLDS)
            for b in blocks[:fold]:
                train_idx.extend(b)
            val_idx.extend(blocks[fold])
            test_idx.extend(blocks[fold + 1])

        train_idx = sorted(train_idx)
        val_idx = sorted(val_idx)
        test_idx = sorted(test_idx)

        train_set = set(train_idx)
        val_set = set(val_idx)
        test_set = set(test_idx)
        if train_set & val_set or train_set & test_set or val_set & test_set:
            raise ValueError(f"第 {fold} 折出现数据泄漏：train/val/test 存在重叠")

        total_unique = len(train_set | val_set | test_set)
        unused_count = len(data) - total_unique
        print(f"当前折未使用样本数: {unused_count}")

        params_train = data[train_idx]
        labels_train = labels[train_idx]
        params_val = data[val_idx]
        labels_val = labels[val_idx]
        params_test = data[test_idx]
        labels_test = labels[test_idx]

        print("训练集类别分布:", dict(Counter(labels_train)))
        print("验证集类别分布:", dict(Counter(labels_val)))
        print("测试集类别分布:", dict(Counter(labels_test)))

        scaler = StandardScaler()
        params_train = scaler.fit_transform(params_train)
        params_val = scaler.transform(params_val)
        params_test = scaler.transform(params_test)

        # ---------- 数据集（训练用增强，验证/测试用简单变换）----------
        train_dataset = TunnellingDataset(params_train, labels_train, train_idx, image_dirs, transform_train)
        val_dataset = TunnellingDataset(params_val, labels_val, val_idx, image_dirs, transform_val)
        test_dataset = TunnellingDataset(params_test, labels_test, test_idx, image_dirs, transform_val)

        print(f"训练集图像缺失数: {train_dataset.missing_count}/{len(train_idx)*3}")
        print(f"验证集图像缺失数: {val_dataset.missing_count}/{len(val_idx)*3}")
        print(f"测试集图像缺失数: {test_dataset.missing_count}/{len(test_idx)*3}")

        # ---------- 类别权重 ----------
        class_counts = np.bincount(labels_train)
        class_weights = 1.0 / np.sqrt(class_counts)
        class_weights = class_weights / class_weights.sum() * len(class_counts)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        # 模型、优化器、调度器
        model = MultimodalModel(param_input_size=len(FEATURE_COLS), num_classes=len(CLASS_NAMES)).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        # scheduler patience 增加到 4
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=4, factor=0.5)

        logger = EnhancedLogger(os.path.join(LOG_DIR, f'fold_{fold}'))

        best_val_acc = 0.0
        best_model_path = os.path.join(LOG_DIR, f'best_model_fold_{fold}.pth')
        early_stop_counter = 0
        best_val_loss = float('inf')
        best_epoch = 0

        for epoch in range(1, MAX_EPOCHS + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, grad_clip_norm=GRAD_CLIP_NORM
            )
            val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_loss)

            metrics = {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'lr': optimizer.param_groups[0]['lr']
            }
            logger.log_metrics(epoch, metrics)

            improved = val_loss < best_val_loss - MIN_DELTA
            if improved:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(model.state_dict(), best_model_path)
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            print(f"Fold {fold} | Epoch {epoch:03d} | "
                  f"Train Loss {train_loss:.4f} Acc {train_acc:.2f}% | "
                  f"Val Loss {val_loss:.4f} Acc {val_acc:.2f}% | "
                  f"LR {optimizer.param_groups[0]['lr']:.6f}")

            if early_stop_counter >= EARLY_STOP_PATIENCE:
                print(f"第 {fold} 折触发早停，停止于 epoch {epoch}")
                break

        logger.close()
        curve_png = os.path.join(RESULTS_DIR, f'learning_curve_fold{fold}.png')
        logger.plot_curves(curve_png)
        logger.save_history_to_excel(os.path.join(RESULTS_DIR, f'learning_curve_fold{fold}.xlsx'))

        # 加载最佳模型在测试集上评估
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        test_loss, test_acc, matrix, all_true, all_pred = evaluate(model, test_loader, criterion, device)
        all_fold_acc.append(test_acc)

        print(f"第 {fold} 折测试准确率: {test_acc:.2f}%")
        print("分类报告:")
        report_text = classification_report(all_true, all_pred, target_names=CLASS_NAMES, zero_division=0)
        print(report_text)

        save_text(report_text, os.path.join(RESULTS_DIR, f'classification_report_fold{fold}.txt'))
        report_df = classification_report_to_dataframe(all_true, all_pred, CLASS_NAMES)
        save_dataframe_multi(report_df, os.path.join(RESULTS_DIR, f'classification_report_fold{fold}'))
        save_prediction_details(all_true, all_pred, test_idx, CLASS_NAMES,
                                os.path.join(RESULTS_DIR, f'predictions_fold{fold}'))
        cm_path = os.path.join(RESULTS_DIR, f'confusion_matrix_fold{fold}.png')
        save_confusion_matrix(matrix, CLASS_NAMES, cm_path)
        save_confusion_matrix_table(matrix, CLASS_NAMES,
                                    os.path.join(RESULTS_DIR, f'confusion_matrix_fold{fold}'))

        report_dict = classification_report(all_true, all_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
        fold_summary = {
            'fold': fold,
            'best_epoch': best_epoch,
            'best_val_acc': best_val_acc,
            'best_val_loss': best_val_loss,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'test_size': len(test_idx),
            'unused_size': unused_count,
            'train_dist': str(dict(Counter(labels_train))),
            'val_dist': str(dict(Counter(labels_val))),
            'test_dist': str(dict(Counter(labels_test))),
            'macro_f1': report_dict['macro avg']['f1-score'],
            'weighted_f1': report_dict['weighted avg']['f1-score'],
        }
        for cname in CLASS_NAMES:
            fold_summary[f'{cname}_precision'] = report_dict[cname]['precision']
            fold_summary[f'{cname}_recall'] = report_dict[cname]['recall']
            fold_summary[f'{cname}_f1'] = report_dict[cname]['f1-score']
        fold_summaries.append(fold_summary)

        # 保存该折模型路径用于集成
        ensemble_model_paths.append(best_model_path)

    # ---------- K折汇总 ----------
    print("\n==================== K折交叉验证结果汇总 ====================")
    for i, acc in enumerate(all_fold_acc, start=1):
        print(f"Fold {i}: {acc:.2f}%")
    print(f"平均测试准确率: {np.mean(all_fold_acc):.2f}% ± {np.std(all_fold_acc):.2f}%")

    plt.figure(figsize=(8, 4))
    x = list(range(1, len(all_fold_acc)+1))
    plt.plot(x, all_fold_acc, marker='o')
    plt.xticks(x)
    plt.xlabel('Fold')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Accuracy across Folds')
    plt.grid(True)
    acc_curve_png = os.path.join(RESULTS_DIR, 'fold_accuracies.png')
    plt.tight_layout()
    plt.savefig(acc_curve_png, dpi=200)
    print(f"各折准确率曲线已保存为 PNG: {acc_curve_png}")
    base, _ = os.path.splitext(acc_curve_png)
    try:
        plt.savefig(base + '.emf', format='emf')
        print(f"各折准确率曲线已保存为 EMF: {base}.emf")
    except Exception as e:
        print(f"无法保存 EMF 格式 (将保存为 SVG): {e}")
        plt.savefig(base + '.svg', format='svg')
        print(f"各折准确率曲线已保存为 SVG: {base}.svg")
    plt.close()

    fold_df = pd.DataFrame(fold_summaries)
    overall_summary_df = pd.DataFrame([{
        'mean_test_acc': np.mean(all_fold_acc),
        'std_test_acc': np.std(all_fold_acc),
        'num_folds': len(all_fold_acc)
    }])

    kfold_xlsx = os.path.join(RESULTS_DIR, 'kfold_results.xlsx')
    with pd.ExcelWriter(kfold_xlsx) as writer:
        fold_df.to_excel(writer, index=False, sheet_name='fold_metrics')
        overall_summary_df.to_excel(writer, index=False, sheet_name='overall_summary')
    print(f"K折汇总结果已保存为 Excel: {kfold_xlsx}")
    fold_df.to_csv(os.path.join(RESULTS_DIR, 'kfold_fold_metrics.csv'), index=False, encoding='utf-8-sig')
    overall_summary_df.to_csv(os.path.join(RESULTS_DIR, 'kfold_overall_summary.csv'), index=False, encoding='utf-8-sig')

    # ---------- 最终模型训练（使用全部数据） ----------
    print("\n" + "=" * 50)
    print("开始训练最终模型（train / val / independent test）...")
    print("=" * 50)

    FINAL_TEST_RATIO = 0.20
    FINAL_VAL_RATIO_IN_TRAINVAL = 0.10

    train_indices_final = []
    val_indices_final = []
    test_indices_final = []

    for c in np.unique(labels):
        class_idx = original_indices[labels == c]
        split_test = int(len(class_idx) * (1 - FINAL_TEST_RATIO))
        trainval_idx = class_idx[:split_test]
        test_idx_c = class_idx[split_test:]

        split_val = int(len(trainval_idx) * (1 - FINAL_VAL_RATIO_IN_TRAINVAL))
        train_idx_c = trainval_idx[:split_val]
        val_idx_c = trainval_idx[split_val:]

        train_indices_final.extend(train_idx_c)
        val_indices_final.extend(val_idx_c)
        test_indices_final.extend(test_idx_c)

    train_indices_final.sort()
    val_indices_final.sort()
    test_indices_final.sort()

    params_train_final = data[train_indices_final]
    labels_train_final = labels[train_indices_final]
    params_val_final = data[val_indices_final]
    labels_val_final = labels[val_indices_final]
    params_test_final = data[test_indices_final]
    labels_test_final = labels[test_indices_final]

    print("最终训练集类别分布:", dict(Counter(labels_train_final)))
    print("最终验证集类别分布:", dict(Counter(labels_val_final)))
    print("最终测试集类别分布:", dict(Counter(labels_test_final)))

    final_scaler = StandardScaler()
    params_train_final = final_scaler.fit_transform(params_train_final)
    params_val_final = final_scaler.transform(params_val_final)
    params_test_final = final_scaler.transform(params_test_final)

    # ---------- 最终数据集（训练用增强）----------
    train_dataset_final = TunnellingDataset(params_train_final, labels_train_final, train_indices_final, image_dirs, transform_train)
    val_dataset_final = TunnellingDataset(params_val_final, labels_val_final, val_indices_final, image_dirs, transform_val)
    test_dataset_final = TunnellingDataset(params_test_final, labels_test_final, test_indices_final, image_dirs, transform_val)

    print(f"最终训练集图像缺失数: {train_dataset_final.missing_count}/{len(train_indices_final)*3}")
    print(f"最终验证集图像缺失数: {val_dataset_final.missing_count}/{len(val_indices_final)*3}")
    print(f"最终测试集图像缺失数: {test_dataset_final.missing_count}/{len(test_indices_final)*3}")

    # 最终模型权重和采样器
    class_counts_final = np.bincount(labels_train_final)
    class_weights_final = 1.0 / np.sqrt(class_counts_final)
    class_weights_final = class_weights_final / class_weights_final.sum() * len(class_counts_final)
    class_weights_final = torch.tensor(class_weights_final, dtype=torch.float).to(device)
    criterion_final = nn.CrossEntropyLoss(weight=class_weights_final)

    train_loader_final = DataLoader(train_dataset_final, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader_final = DataLoader(val_dataset_final, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader_final = DataLoader(test_dataset_final, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    final_split_df = pd.DataFrame({
        'split': (['train'] * len(train_indices_final)) + (['val'] * len(val_indices_final)) + (['test'] * len(test_indices_final)),
        'original_index': train_indices_final + val_indices_final + test_indices_final
    })
    save_dataframe_multi(final_split_df, os.path.join(RESULTS_DIR, 'final_split_indices'))

    final_model = MultimodalModel(param_input_size=len(FEATURE_COLS), num_classes=len(CLASS_NAMES)).to(device)
    optimizer_final = optim.AdamW(final_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler_final = optim.lr_scheduler.ReduceLROnPlateau(optimizer_final, mode='min', patience=4, factor=0.5)
    logger_final = EnhancedLogger(os.path.join(LOG_DIR, 'final_model'))

    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_epoch = 0
    early_stop_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            final_model, train_loader_final, criterion_final, optimizer_final, device, grad_clip_norm=GRAD_CLIP_NORM
        )
        val_loss, val_acc, _, _, _ = evaluate(final_model, val_loader_final, criterion_final, device)
        scheduler_final.step(val_loss)

        metrics = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': optimizer_final.param_groups[0]['lr']
        }
        logger_final.log_metrics(epoch, metrics)

        improved = val_loss < best_val_loss - MIN_DELTA
        if improved:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(final_model.state_dict(), MODEL_SAVE_PATH)
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        print(f"Final Model | Epoch {epoch:03d} | "
              f"Train Loss {train_loss:.4f} Acc {train_acc:.2f}% | "
              f"Val Loss {val_loss:.4f} Acc {val_acc:.2f}% | "
              f"LR {optimizer_final.param_groups[0]['lr']:.6f}")

        if early_stop_counter >= EARLY_STOP_PATIENCE:
            print(f"最终模型触发早停，停止于 epoch {epoch}")
            break

    logger_final.close()
    final_curve_png = os.path.join(RESULTS_DIR, 'final_learning_curve.png')
    logger_final.plot_curves(final_curve_png)
    logger_final.save_history_to_excel(os.path.join(RESULTS_DIR, 'final_learning_curve.xlsx'))

    joblib.dump(final_scaler, SCALER_PATH)
    print(f"最终标准化器已保存: {SCALER_PATH}")

    final_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))

    # ---------- 最终模型在验证集和测试集上的评估 ----------
    print("\n=== 最终模型在验证集上的评估 ===")
    val_loss, val_acc, val_matrix, val_true, val_pred = evaluate(final_model, val_loader_final, criterion_final, device)
    print(f"验证集准确率: {val_acc:.2f}%")
    val_report_text = classification_report(val_true, val_pred, target_names=CLASS_NAMES, zero_division=0)
    print("验证集分类报告:")
    print(val_report_text)
    save_text(val_report_text, os.path.join(RESULTS_DIR, 'classification_report_final_val.txt'))
    save_dataframe_multi(classification_report_to_dataframe(val_true, val_pred, CLASS_NAMES),
                         os.path.join(RESULTS_DIR, 'classification_report_final_val'))
    save_prediction_details(val_true, val_pred, val_indices_final, CLASS_NAMES,
                            os.path.join(RESULTS_DIR, 'predictions_final_val'))
    save_confusion_matrix(val_matrix, CLASS_NAMES, os.path.join(RESULTS_DIR, 'confusion_matrix_final_val.png'))
    save_confusion_matrix_table(val_matrix, CLASS_NAMES,
                                os.path.join(RESULTS_DIR, 'confusion_matrix_final_val'))

    print("\n=== 最终模型在独立测试集上的评估 ===")
    test_loss_final, test_acc_final, test_matrix_final, test_true_final, test_pred_final = evaluate(
        final_model, test_loader_final, criterion_final, device
    )
    print(f"独立测试集准确率: {test_acc_final:.2f}%")
    test_report_text = classification_report(test_true_final, test_pred_final, target_names=CLASS_NAMES, zero_division=0)
    print("独立测试集分类报告:")
    print(test_report_text)
    save_text(test_report_text, os.path.join(RESULTS_DIR, 'classification_report_final_test.txt'))
    save_dataframe_multi(classification_report_to_dataframe(test_true_final, test_pred_final, CLASS_NAMES),
                         os.path.join(RESULTS_DIR, 'classification_report_final_test'))
    save_prediction_details(test_true_final, test_pred_final, test_indices_final, CLASS_NAMES,
                            os.path.join(RESULTS_DIR, 'predictions_final_test'))
    save_confusion_matrix(test_matrix_final, CLASS_NAMES, os.path.join(RESULTS_DIR, 'confusion_matrix_final_test.png'))
    save_confusion_matrix_table(test_matrix_final, CLASS_NAMES,
                                os.path.join(RESULTS_DIR, 'confusion_matrix_final_test'))

    final_metrics_summary = pd.DataFrame([
        {'split': 'val', 'loss': val_loss, 'acc': val_acc, 'size': len(val_indices_final)},
        {'split': 'test', 'loss': test_loss_final, 'acc': test_acc_final, 'size': len(test_indices_final)}
    ])
    save_dataframe_multi(final_metrics_summary, os.path.join(RESULTS_DIR, 'final_model_metrics_summary'))

    # ---------- 敏感性分析 ----------
    sensitivity_analysis_pipeline(
        final_model,
        params_test=params_test_final,
        labels_test=labels_test_final,
        test_idx=test_indices_final,
        image_dirs=image_dirs,
        test_loader=test_loader_final,
        device=device
    )

    # ---------- 集成模型评估 ----------
    print("\n" + "=" * 50)
    print("集成模型评估（使用所有折的最佳模型在独立测试集上）")
    print("=" * 50)
    if len(ensemble_model_paths) > 0:
        all_true_ensemble, all_pred_ensemble, _ = ensemble_predict(
            ensemble_model_paths, test_loader_final, device, len(CLASS_NAMES)
        )
        ensemble_report = classification_report(all_true_ensemble, all_pred_ensemble, target_names=CLASS_NAMES, zero_division=0)
        print("集成模型分类报告:")
        print(ensemble_report)

        save_text(ensemble_report, os.path.join(RESULTS_DIR, 'classification_report_ensemble.txt'))
        ensemble_report_df = classification_report_to_dataframe(all_true_ensemble, all_pred_ensemble, CLASS_NAMES)
        save_dataframe_multi(ensemble_report_df, os.path.join(RESULTS_DIR, 'classification_report_ensemble'))

        from sklearn.metrics import confusion_matrix
        ensemble_matrix = confusion_matrix(all_true_ensemble, all_pred_ensemble)
        save_confusion_matrix(ensemble_matrix, CLASS_NAMES, os.path.join(RESULTS_DIR, 'confusion_matrix_ensemble.png'))
        save_confusion_matrix_table(ensemble_matrix, CLASS_NAMES,
                                    os.path.join(RESULTS_DIR, 'confusion_matrix_ensemble'))

        save_prediction_details(all_true_ensemble, all_pred_ensemble, test_indices_final, CLASS_NAMES,
                                os.path.join(RESULTS_DIR, 'predictions_ensemble'))
    else:
        print("警告：没有可用的折模型用于集成。")

    print("\n全部流程完成。")

if __name__ == '__main__':
    main()