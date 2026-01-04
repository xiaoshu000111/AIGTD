import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import json  # 新增：用于保存数据
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


BASE_DIR = '/home/scutnlp131/Tree' # 示例路径，请修改！


# === 配置区域 ===
# 请根据你的服务器实际路径修改以下三行
TRAIN_ROOT = os.path.join(BASE_DIR, 'dataset/detectRL-zh/train')
TEST_ROOT = os.path.join(BASE_DIR, 'dataset/detectRL-zh/test')
# 结果保存目录 (模型、图片、日志都存在这里)
RESULT_DIR = os.path.join(BASE_DIR, 'result/results_convnext_tiny') 

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 30
PATIENCE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = os.path.join(RESULT_DIR, 'best_model.pth')

def train_val_split(dataset, val_ratio=0.1, random_state=42):
    indices = list(range(len(dataset)))
    targets = [dataset.samples[i][1] for i in indices]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_state)
    train_idx, val_idx = next(sss.split(indices, targets))
    return Subset(dataset, train_idx), Subset(dataset, val_idx)

# === 修改绘图函数：增加 save_path 参数，使用 savefig 而非 show ===
def plot_roc_curve(y_true, y_probs, title='ROC Curve', save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        print(f"📄 ROC曲线已保存至: {save_path}")
    plt.close() # 关闭画布，释放内存

def plot_confusion_matrix(y_true, y_pred, class_names, title='Confusion Matrix', save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path)
        print(f"📄 混淆矩阵已保存至: {save_path}")
    plt.close()

    # 将分类准确率写入日志
    log_content = "\n📊 Class-wise Accuracy:\n"
    print(log_content.strip())
    for i, class_name in enumerate(class_names):
        if np.sum(y_true == i) > 0:
            class_acc = cm[i, i] / np.sum(y_true == i) * 100
            print(f"  {class_name}: {class_acc:.2f}% ({cm[i, i]}/{np.sum(y_true == i)})")

def plot_training_history(train_losses, val_losses, val_accuracies, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss
    ax1.plot(train_losses, label='Training Loss', color='blue', linewidth=2)
    ax1.plot(val_losses, label='Validation Loss', color='orange', linewidth=2, linestyle='--')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training vs Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(val_accuracies, label='Validation Accuracy', color='green', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Validation Accuracy Over Epochs')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"📄 训练曲线已保存至: {save_path}")
    plt.close()

def run_experiment():
    print(f"🚀 环境检查: Using device: {DEVICE}")
    print(f"📂 结果输出目录: {RESULT_DIR}")
    
    # 2. 数据预处理
    print("\n📊 设置数据预处理管道...")
    common_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. 加载数据集
    print(f"\n📂 正在加载训练数据: {TRAIN_ROOT} ...")
    try:
        full_train_dataset = datasets.ImageFolder(root=TRAIN_ROOT, transform=common_transform)
    except Exception as e:
        print(f"❌ 训练集加载错误: {e}")
        return

    print(f"📂 正在加载测试数据: {TEST_ROOT} ...")
    try:
        test_dataset = datasets.ImageFolder(root=TEST_ROOT, transform=common_transform)
    except Exception as e:
        print(f"❌ 测试集加载错误: {e}")
        return

    print(f"✅ 成功加载! 类别映射: {full_train_dataset.class_to_idx}")

    # 4. 划分验证集
    VAL_RATIO = 0.1
    train_subset, val_subset = train_val_split(full_train_dataset, val_ratio=VAL_RATIO)
    
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # 5. 模型初始化 (ConvNeXt-Tiny)
    print("\n🧠 初始化 ConvNeXt-Tiny 模型...")
    try:
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    except:
        model = models.convnext_tiny(pretrained=True)

    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 2)
    )
    model = model.to(DEVICE)

    # 6. 优化器与 Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6)

    # 7. 开始训练
    print("\n⚡ === 开始训练 ===")
    best_acc = 0.0
    best_epoch = 0
    no_improve = 0

    # 记录数据的列表
    history = {
        'train_losses': [],
        'val_losses': [],
        'val_accuracies': []
    }

    for epoch in range(EPOCHS):
        # === Training ===
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            pbar.set_postfix({'loss': loss.item(), 'acc': f'{100*correct_train/total_train:.2f}%'})

        epoch_loss = running_loss / len(train_subset)
        
        # === Validation ===
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_epoch_loss = val_running_loss / len(val_subset)
        val_acc = 100 * correct_val / total_val

        # === 记录数据 ===
        history['train_losses'].append(epoch_loss)
        history['val_losses'].append(val_epoch_loss)
        history['val_accuracies'].append(val_acc)

        print(f"📊 Epoch {epoch+1}: Train Loss {epoch_loss:.4f} | Val Loss {val_epoch_loss:.4f} | Val Acc {val_acc:.2f}%")

        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'class_to_idx': full_train_dataset.class_to_idx,
            }, SAVE_PATH)
            print(f"💾 最佳模型已保存! (Val Acc: {val_acc:.2f}%)")
        else:
            no_improve += 1
            print(f"⚠️ 未提升 ({no_improve}/{PATIENCE})")

        if no_improve >= PATIENCE:
            print(f"\n🛑 早停触发")
            break

    print(f"\n🏆 训练结束! 最佳验证集准确率: {best_acc:.2f}%")

    # === 保存训练历史数据 (JSON) ===
    history_path = os.path.join(RESULT_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f)
    print(f"📄 训练历史数据已保存至: {history_path}")

    # === 绘制并保存训练曲线 ===
    plot_training_history(history['train_losses'], history['val_losses'], history['val_accuracies'], 
                          save_path=os.path.join(RESULT_DIR, 'training_curve.png'))

    # ==========================================
    # 9. 最终测试 (Test & Save Report)
    # ==========================================
    print("\n" + "="*50)
    print("🔒 正在加载最佳模型并在【独立测试集】上进行评估...")
    print("="*50)

    checkpoint = torch.load(SAVE_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_preds = []
    test_labels = []
    test_probs = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())

    test_labels = np.array(test_labels)
    test_preds = np.array(test_preds)
    test_probs = np.array(test_probs)

    auroc_score = roc_auc_score(test_labels, test_probs[:, 1])
    final_acc = accuracy_score(test_labels, test_preds) * 100
    final_precision = precision_score(test_labels, test_preds, average='weighted') * 100
    final_recall = recall_score(test_labels, test_preds, average='weighted') * 100
    final_f1 = f1_score(test_labels, test_preds, average='weighted') * 100

    # === 生成并保存最终报告 ===
    report_text = f"""
    [FINAL TEST REPORT]
    ===================
    Model: ConvNeXt-Tiny
    Best Epoch: {best_epoch}
    
    Metrics:
    --------
    Accuracy : {final_acc:.2f}%
    Precision: {final_precision:.2f}%
    Recall   : {final_recall:.2f}%
    F1 Score : {final_f1:.2f}%
    AUROC    : {auroc_score:.4f}
    """
    print(report_text)
    
    # 保存报告到 txt
    with open(os.path.join(RESULT_DIR, 'final_report.txt'), 'w') as f:
        f.write(report_text)

    # 绘制并保存混淆矩阵
    class_names = list(full_train_dataset.class_to_idx.keys())
    plot_confusion_matrix(test_labels, test_preds, class_names, 
                          title='Test Set Confusion Matrix',
                          save_path=os.path.join(RESULT_DIR, 'confusion_matrix.png'))

    # 绘制并保存 ROC 曲线
    plot_roc_curve(test_labels, test_probs[:, 1], 
                   title='ROC Curve (Class: Human vs AI)',
                   save_path=os.path.join(RESULT_DIR, 'roc_curve.png'))

    print(f"\n✅ 所有结果已保存至文件夹: {RESULT_DIR}")

if __name__ == '__main__':
    run_experiment()