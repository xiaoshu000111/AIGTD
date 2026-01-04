import os
import json
import re
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# === ⚙️ 全局配置 (保持与训练一致) ===
# ==========================================

# 1. 严格固定画布大小 (测试集专用)
CANVAS_SIZE = (224, 224) 

# 2. 字体大小 
# 建议与训练集保持一致 (例如 18 或 20)
FONT_SIZE = 18

# 4. 边距
MARGIN = 20

# ==========================================

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

def render_text_to_fixed_384(text, save_path):
    """
    【测试集专用】固定 384x384 画布的智能渲染函数
    1. 画布大小锁死为 384x384。
    2. 智能排版，不切断单词。
    3. 如果文字太长，超出 384 高度，则直接截断（不再绘制）。
    """
    # 1. 准备画布
    img = Image.new('RGB', CANVAS_SIZE, (255, 255, 255)) # 白底
    draw = ImageDraw.Draw(img)
    font = get_windows_font(FONT_SIZE)
    
    # 2. 计算排版参数
    max_text_width = CANVAS_SIZE[0] - 2 * MARGIN
    max_text_height = CANVAS_SIZE[1] - MARGIN # 留出下边距
    
    line_height = int(FONT_SIZE * 1.5)
    
    current_y = MARGIN
    current_line = ""
    
    # 3. 文本分段与原子化
    paragraphs = text.split('\n')
    
    stop_rendering = False # 标志位：如果画布满了就停止
    
    for para in paragraphs:
        if stop_rendering: break
        
        # 正则切分原子 (中文字符 / 英文单词 / 空格)
        atoms = re.findall(r'[\u4e00-\u9fa5]|[^\u4e00-\u9fa5\s]+|\s+', para)
        
        for atom in atoms:
            test_line = current_line + atom
            
            # 计算宽度
            if hasattr(font, 'getlength'):
                width = font.getlength(test_line)
            else:
                width = draw.textlength(test_line, font=font)
            
            if width <= max_text_width:
                current_line = test_line
            else:
                # --- 绘制当前行 ---
                # 检查是否超出高度
                if current_y + line_height > max_text_height:
                    stop_rendering = True
                    break
                
                if current_line:
                    draw.text((MARGIN, current_y), current_line, font=font, fill=(0, 0, 0))
                    current_y += line_height
                
                # --- 处理新的一行 ---
                atom_width = font.getlength(atom) if hasattr(font, 'getlength') else draw.textlength(atom, font=font)
                if atom_width > max_text_width:
                    current_line = atom # 强制换行
                else:
                    current_line = atom.lstrip() # 新起一行
        
        # 段落结束，绘制缓冲区剩余内容
        if not stop_rendering and current_line:
            if current_y + line_height > max_text_height:
                stop_rendering = True
            else:
                draw.text((MARGIN, current_y), current_line, font=font, fill=(0, 0, 0))
                current_y += line_height
                current_line = "" # 清空，准备下一段
                
        # (可选) 段落间空行逻辑，如果不需要紧凑排版可以开启
        # if not stop_rendering:
        #     current_y += int(line_height * 0.5) 

    # 4. 保存
    img.save(save_path)


# ==========================================
# === 批量处理入口 (针对测试集) ===
# ==========================================

def process_test_dataset(json_path, output_base_dir):
    """处理测试集数据 (NLPCC 或其他 JSON 格式)"""
    
    human_dir = os.path.join(output_base_dir, "human")
    ai_dir = os.path.join(output_base_dir, "ai")
    
    os.makedirs(human_dir, exist_ok=True)
    os.makedirs(ai_dir, exist_ok=True)
    
    if not os.path.exists(json_path):
        print(f"✗ 文件不存在: {json_path}")
        return
    
    print(f"🚀 开始生成固定测试集 (384x384): {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    
    count = 0
    for item_id, item in enumerate(data_list, start=1):
        try:
            text = item.get('text', '').strip()
            label = item.get('label', -1)
            item_id_val = item.get('id', item_id)
            
            if not text: continue
            
            # 只有这里不同：调用 render_text_to_fixed_384
            if label == 0: # human
                output_path = os.path.join(human_dir, f"{item_id_val}.png")
                render_text_to_fixed_384(text, output_path) 
                count += 1
            elif label == 1: # ai
                output_path = os.path.join(ai_dir, f"{item_id_val}.png")
                render_text_to_fixed_384(text, output_path) 
                count += 1
                
            if count % 100 == 0:
                print(f"已生成 {count} 张...", end='\r')
                
        except Exception as e:
            print(f"Error ID {item_id_val}: {e}")

    print(f"\n🎉 测试集生成完成! 共 {count} 张。")

if __name__ == "__main__":
    # 示例：生成 DetectRL-zh 的测试集
    test_json_file = r"d:\Desktop\文本检测\NLPCC-2025-Task1-main\data\test_with_label.json" 
    test_output_dir = r"/dataset/detectRL-zh/384/test"
    
    process_test_dataset(test_json_file, test_output_dir)