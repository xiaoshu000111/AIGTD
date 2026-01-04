import os
import json
import re
from PIL import Image, ImageDraw, ImageFont
import textwrap

'''
智能渲染函数，支持中文按字换行，英文按单词换行，重新生成了没有引入噪声的长图。
'''

# === 全局配置 ===
# 图片宽度 (Swin-Base 推荐 384 或 448)
IMG_WIDTH = 384 
# 字体大小 (适中即可，太小了虽然字多但看不清，太大了内容少)
FONT_SIZE = 18
# 边距
MARGIN = 20

def get_windows_font(size=20):
    """
    Windows 专用字体加载器
    优先加载微软雅黑 (msyh.ttc) 或 黑体 (simhei.ttf)
    """
    # Windows 字体目录
    windows_font_dir = r"D:\Downloads\SiYuanHeiTi-Regular\SiYuanHeiTi-Regular"
    
    # 优先级列表
    font_names = [
        "SourceHanSansSC-Regular-2.otf", # 思源黑体
        "msyh.ttc",   # 微软雅黑 (最清晰，首选)
        "simhei.ttf", # 黑体
        "simsun.ttc", # 宋体
        "arial.ttf"   # 英文保底
    ]
    
    for name in font_names:
        font_path = os.path.join(windows_font_dir, name)
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except Exception as e:
                continue
    
    print("⚠️ 未找到常用中文字体，尝试使用默认字体...")
    return ImageFont.load_default()

def render_text_smartly(text, save_path):
    """
    【核心修改】智能排版渲染函数
    1. 英文按单词换行，不切断。
    2. 中文按字换行。
    3. 生成长图 (Long Image)，宽度固定为 IMG_WIDTH。
    """
    # 1. 加载字体
    font = get_windows_font(FONT_SIZE)
    
    # 2. 准备画布参数
    max_text_width = IMG_WIDTH - 2 * MARGIN
    lines = []
    current_line = ""
    
    # 创建虚拟画布用于计算文字宽度
    dummy_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    
    # 3. 文本清洗与分段
    # 如果文本里原本就有换行符，先按换行符切分，保持段落结构
    # 这一步对于保留 AI 的“列表特征”很重要
    paragraphs = text.split('\n')
    
    for para in paragraphs:
        # 正则切分：中文按字，英文按单词
        # r'[\u4e00-\u9fa5]': 匹配单个汉字
        # r'[^\u4e00-\u9fa5\s]+': 匹配连续的非汉字非空格字符 (即英文单词、标点)
        # r'\s+': 匹配空格
        atoms = re.findall(r'[\u4e00-\u9fa5]|[^\u4e00-\u9fa5\s]+|\s+', para)
        
        for atom in atoms:
            test_line = current_line + atom
            
            # 计算宽度 (兼容不同版本的 Pillow)
            if hasattr(font, 'getlength'):
                width = font.getlength(test_line)
            else:
                width = dummy_draw.textlength(test_line, font=font)
                
            if width <= max_text_width:
                # 如果能放下，就加到当前行
                current_line = test_line
            else:
                # 放不下，先把当前行存入 lines
                if current_line:
                    lines.append(current_line)
                
                # 处理单个原子过长的情况 (极少见，比如超长网址)
                atom_width = font.getlength(atom) if hasattr(font, 'getlength') else dummy_draw.textlength(atom, font=font)
                if atom_width > max_text_width:
                    current_line = atom # 没办法只能强制换行
                else:
                    current_line = atom.lstrip() # 新起一行，去掉开头的空格
        
        # 段落结束，把缓冲区内容加入
        if current_line:
            lines.append(current_line)
            current_line = ""
            
        # (可选) 如果希望段落间有额外空行，可以取消下面注释
        # lines.append("") 

    # 4. 计算图片高度
    # 行高设置为字体的 1.5 倍，提供良好的阅读呼吸感
    line_height = int(FONT_SIZE * 1.5)
    # 总高度 = 行数 * 行高 + 上下边距
    image_height = len(lines) * line_height + 2 * MARGIN
    
    # 保证图片至少是正方形 (防止文字太少导致图片太扁，ResNet/Swin 喜欢正方形)
    if image_height < IMG_WIDTH:
        image_height = IMG_WIDTH

    # 5. 绘制图片
    img = Image.new('RGB', (IMG_WIDTH, image_height), (255, 255, 255)) # 白底
    draw = ImageDraw.Draw(img)
    
    y_text = MARGIN
    for line in lines:
        draw.text((MARGIN, y_text), line, font=font, fill=(0, 0, 0)) # 黑字
        y_text += line_height
        
    img.save(save_path)

# ==========================================
# 下面是你的数据处理逻辑 (已更新调用)
# ==========================================

def process_clean_hc3_qa(train_jsonl, test_jsonl, output_base_dir):
    """处理 clean_hc3_qa 数据"""
    
    # 创建目录结构
    train_human_dir = os.path.join(output_base_dir, "train_data", "human")
    train_ai_dir = os.path.join(output_base_dir, "train_data", "ai")
    test_human_dir = os.path.join(output_base_dir, "test_data", "human")
    test_ai_dir = os.path.join(output_base_dir, "test_data", "ai")
    
    for folder in [train_human_dir, train_ai_dir, test_human_dir, test_ai_dir]:
        os.makedirs(folder, exist_ok=True)
    
    print("正在处理训练集...")
    _process_dataset(train_jsonl, train_human_dir, train_ai_dir, "train")
    
    print("正在处理测试集...")
    _process_dataset(test_jsonl, test_human_dir, test_ai_dir, "test")
    
    print("\n✓ 所有数据处理完成！")

def _process_dataset(jsonl_path, human_folder, ai_folder, dataset_type):
    if not os.path.exists(jsonl_path):
        print(f"✗ 文件不存在: {jsonl_path}")
        return
    
    count = 0
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            try:
                data = json.loads(line.strip())
                label = data.get('label', '').strip().lower()
                text = data.get('article', '').strip()
                
                # 【重要】这里我去掉了 replace('\\n', '')
                # 因为新的渲染函数 render_text_smartly 可以很好地处理段落结构
                # 如果你想恢复成一行，可以取消下面注释
                # text = text.replace('\\n', ' ').replace('\n', ' ')
                
                if not text: continue
                
                if label == 'human':
                    output_path = os.path.join(human_folder, f"{line_num}.png")
                    render_text_smartly(text, output_path) # <--- 替换为新函数
                elif label == 'machine':
                    output_path = os.path.join(ai_folder, f"{line_num}.png")
                    render_text_smartly(text, output_path) # <--- 替换为新函数
                
                count += 1
                if count % 100 == 0:
                    print(f"Processing {dataset_type}: {count}...", end='\r')
                    
            except Exception as e:
                print(f"Error line {line_num}: {e}")

def process_nlpcc_train_data(json_path, output_base_dir):
    """处理 NLPCC 训练数据"""
    
    human_dir = os.path.join(output_base_dir, "human")
    ai_dir = os.path.join(output_base_dir, "ai")
    
    os.makedirs(human_dir, exist_ok=True)
    os.makedirs(ai_dir, exist_ok=True)
    
    if not os.path.exists(json_path):
        print(f"✗ 文件不存在: {json_path}")
        return
    
    print(f"🚀 开始处理: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    
    count = 0
    for item_id, item in enumerate(data_list, start=1):
        try:
            text = item.get('text', '').strip()
            label = item.get('label', -1)
            item_id_val = item.get('id', item_id)
            
            # 同样保留换行符，交给渲染函数处理
            if not text: continue
            
            if label == 0: # human
                output_path = os.path.join(human_dir, f"{item_id_val}.png")
                render_text_smartly(text, output_path) # <--- 替换为新函数
                count += 1
            elif label == 1: # ai
                output_path = os.path.join(ai_dir, f"{item_id_val}.png")
                render_text_smartly(text, output_path) # <--- 替换为新函数
                count += 1
                
            if count % 100 == 0:
                print(f"已生成 {count} 张...", end='\r')
                
        except Exception as e:
            print(f"Error ID {item_id_val}: {e}")

    print(f"\n🎉 处理完成! 共 {count} 张。")

    

# === 主程序入口 ===
if __name__ == "__main__":
    
    # 请根据你的实际情况取消注释运行
    
    # 任务 1: 处理 NLPCC 数据
    # ------------------------------------------------
    print(">>> 任务 1: 处理 NLPCC 数据")
    json_file = r"d:\Desktop\文本检测\NLPCC-2025-Task1-main\data\test_with_label.json"
    output_dir = r"/dataset/detectRL-zh/384/test"
    process_nlpcc_train_data(json_file, output_dir)
    
    
    # 任务 2: 处理 HC3 数据 (如果有需要)
    # ------------------------------------------------
    # print("\n>>> 任务 2: 处理 clean_hc3_qa 数据")
    # train_jsonl = r"d:\Desktop\文本检测\HC3-Chinese\clean_hc3_qa\train.jsonl"
    # test_jsonl = r"d:\Desktop\文本检测\HC3-Chinese\clean_hc3_qa\test.jsonl"
    # hc3_output_dir = r"d:\Desktop\文本检测\HC3-Chinese\dataset\clean"
    # process_clean_hc3_qa(train_jsonl, test_jsonl, hc3_output_dir)