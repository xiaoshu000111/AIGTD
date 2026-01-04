from PIL import Image, ImageDraw, ImageFont
import textwrap
import os
import json

def text_to_long_image(text, output_path, img_width=224):
    """
    将文本转换为长图片
    img_width: 固定宽度，建议设为 300 或 400，既能体现段落结构，又方便后续裁剪
    """
    # 1. 设置字体 (Windows)
    try:
        font_size = 14
        # 优先使用微软雅黑 (msyh.ttf) 或 黑体 (simhei.ttf)
        font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", font_size)
    except:
        font = ImageFont.load_default()
        print("未找到中文字体，使用默认字体（可能乱码）")

    # 2. 文本换行计算
    chars_per_line = int(img_width / font_size) - 2 
    lines = textwrap.wrap(text, width=chars_per_line)
    
    # 3. 动态计算高度
    line_spacing = 6  # 行间距
    line_height = font_size + line_spacing
    
    # 计算需要的总高度
    content_height = len(lines) * line_height + 40 # 40是上下边距
    
    # 强制最小高度为 224 (为了后续方便喂给 ResNet)
    final_height = max(224, content_height)
    
    # 4. 创建画布
    img = Image.new('RGB', (img_width, final_height), color='white')
    d = ImageDraw.Draw(img)
    
    # 5. 绘制文本
    y_text = 20
    for line in lines:
        d.text((15, y_text), line, font=font, fill=(0, 0, 0)) # 15是左边距
        y_text += line_height
        
    img.save(output_path)

def text_to_fixed_image_sure(text, output_path, img_size=(224, 224), font_size=14, left_margin=15, right_margin=15):
    """
    将文本转换为固定大小的图片 (224, 224)
    
    参数：
    - text: 要转换的文本
    - output_path: 输出图片路径
    - img_size: 画布大小，固定为 (224, 224)
    - font_size: 字体大小，固定为 14px
    - left_margin: 左边距，默认 15px
    - right_margin: 右边距，默认 15px
    """
    # 1. 设置字体 (Windows)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", font_size)
    except:
        font = ImageFont.load_default()
        print("未找到中文字体，使用默认字体（可能乱码）")
    
    # 2. 计算排版参数
    img_width, img_height = img_size
    available_width = img_width - left_margin - right_margin  # 可用宽度：224-15-15=194
    chars_per_line = max(1, int(available_width / font_size))  # 每行字符数：约194/14=13-14个字
    
    line_spacing = 4  # 行间距
    line_height = font_size + line_spacing  # 每行高度：14+4=18px
    
    max_lines = int((img_height - 10) / line_height)  # 最多行数：(224-10)/18≈11-12行
    max_total_chars = chars_per_line * max_lines  # 最多字符数：13*11≈143个字
    
    # 3. 截断长文本
    if len(text) > max_total_chars:
        text = text[:max_total_chars]
    
    # 4. 文本换行处理
    lines = textwrap.wrap(text, width=chars_per_line)
    lines = lines[:max_lines]  # 只保留能显示的行数
    
    # 5. 创建画布
    img = Image.new('RGB', img_size, color='white')
    draw = ImageDraw.Draw(img)
    
    # 6. 绘制文本
    y_offset = 5  # 上边距
    for line in lines:
        draw.text((left_margin, y_offset), line, font=font, fill=(0, 0, 0))
        y_offset += line_height
        
        # 防止超出画布
        if y_offset >= img_height:
            break
    
    img.save(output_path)


def process_all_jsonl_sure(jsonl_path, human_folder, ai_folder, img_width=224):
    """
    处理jsonl文件，将human_answers和chatgpt_answers转换为固定大小(224x224)的图片
    
    参数：
    - jsonl_path: jsonl文件路径
    - human_folder: human答案输出文件夹路径
    - ai_folder: chatgpt答案输出文件夹路径
    - img_width: 图片宽度，默认224
    """
    # 创建输出文件夹（如果不存在）
    os.makedirs(human_folder, exist_ok=True)
    os.makedirs(ai_folder, exist_ok=True)
    
    # 读取和处理jsonl文件
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=0):  # 从0开始编号
            try:
                data = json.loads(line.strip())
                
                # 提取human_answers和chatgpt_answers
                human_answer = data.get('human_answers', [''])[0]  # 取第一条答案
                chatgpt_answer = data.get('chatgpt_answers', [''])[0]

                # 将所有换行符替换成空字符串，使文本成为单行
                human_answer = human_answer.replace('\\n', '').replace('\n', '').replace('\r', '')
                chatgpt_answer = chatgpt_answer.replace('\\n', '').replace('\n', '').replace('\r', '')
                
                # 生成图片
                if human_answer:
                    human_img_path = os.path.join(human_folder, f"{line_num+1}.png")
                    text_to_fixed_image_sure(human_answer, human_img_path)
                    print(f"✓ 已生成 human 图片: {line_num+1}.png")
                
                if chatgpt_answer:
                    ai_img_path = os.path.join(ai_folder, f"{line_num+1}.png")
                    text_to_fixed_image_sure(chatgpt_answer, ai_img_path)
                    print(f"✓ 已生成 ai 图片: {line_num+1}.png")
                    
            except json.JSONDecodeError as e:
                print(f"✗ 第 {line_num+1} 行JSON解析失败: {e}")
            except Exception as e:
                print(f"✗ 第 {line_num+1} 行处理失败: {e}")



def process_clean_hc3_qa(train_jsonl, test_jsonl, output_base_dir):
    """
    处理 clean_hc3_qa 数据，根据 label 字段和数据集类型将数据转换为图片
    
    参数：
    - train_jsonl: 训练集JSONL文件路径
    - test_jsonl: 测试集JSONL文件路径
    - output_base_dir: 输出基础目录（clean文件夹路径）
    """
    
    # 创建目录结构
    train_human_dir = os.path.join(output_base_dir, "train_data", "human")
    train_ai_dir = os.path.join(output_base_dir, "train_data", "ai")
    test_human_dir = os.path.join(output_base_dir, "test_data", "human")
    test_ai_dir = os.path.join(output_base_dir, "test_data", "ai")
    
    # 创建所有输出文件夹
    for folder in [train_human_dir, train_ai_dir, test_human_dir, test_ai_dir]:
        os.makedirs(folder, exist_ok=True)
    
    # 处理训练集
    print("=" * 50)
    print("处理训练集数据...")
    print("=" * 50)
    _process_dataset(train_jsonl, train_human_dir, train_ai_dir, "train")
    
    # 处理测试集
    print("\n" + "=" * 50)
    print("处理测试集数据...")
    print("=" * 50)
    _process_dataset(test_jsonl, test_human_dir, test_ai_dir, "test")
    
    print("\n✓ 所有数据处理完成！")


def _process_dataset(jsonl_path, human_folder, ai_folder, dataset_type):
    """
    处理单个数据集
    
    参数：
    - jsonl_path: JSONL文件路径
    - human_folder: human标签输出文件夹
    - ai_folder: ai标签输出文件夹
    - dataset_type: 数据集类型（train 或 test）
    """
    
    if not os.path.exists(jsonl_path):
        print(f"✗ 文件不存在: {jsonl_path}")
        return
    
    train_count = 0
    ai_count = 0
    error_count = 0
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            try:
                data = json.loads(line.strip())
                
                # 获取关键字段
                label = data.get('label', '').strip().lower()
                text = data.get('article', '').strip()
                
                # 清理文本中的特殊字符
                text = text.replace('\\n', '').replace('\n', '').replace('\r', '')
                
                if not text:
                    print(f"⚠ 第 {line_num} 行：文本为空，跳过")
                    continue
                
                # 根据label判断输出目录
                if label == 'human':
                    output_path = os.path.join(human_folder, f"{line_num}.png")
                    text_to_fixed_image_sure(text, output_path)
                    print(f"✓ [{dataset_type}] 已生成 human 图片: {line_num}.png")
                    train_count += 1
                    
                elif label == 'machine':
                    output_path = os.path.join(ai_folder, f"{line_num}.png")
                    text_to_fixed_image_sure(text, output_path)
                    print(f"✓ [{dataset_type}] 已生成 ai 图片: {line_num}.png")
                    ai_count += 1
                    
                else:
                    print(f"⚠ 第 {line_num} 行：标签值 '{label}' 不识别，跳过")
                    
            except json.JSONDecodeError as e:
                print(f"✗ 第 {line_num} 行JSON解析失败: {e}")
                error_count += 1
            except Exception as e:
                print(f"✗ 第 {line_num} 行处理失败: {e}")
                error_count += 1
    
    print(f"\n📊 [{dataset_type}] 统计结果:")
    print(f"   - human 图片: {train_count} 张")
    print(f"   - ai 图片: {ai_count} 张")
    print(f"   - 错误: {error_count} 条")
    print(f"   - 总计: {train_count + ai_count} 张")


def process_all_jsonl_sure(jsonl_path, human_folder, ai_folder, img_width=224):
    """
    处理jsonl文件，将human_answers和chatgpt_answers转换为固定大小(224x224)的图片
    
    参数：
    - jsonl_path: jsonl文件路径
    - human_folder: human答案输出文件夹路径
    - ai_folder: chatgpt答案输出文件夹路径
    - img_width: 图片宽度，默认224
    """
    # 创建输出文件夹（如果不存在）
    os.makedirs(human_folder, exist_ok=True)
    os.makedirs(ai_folder, exist_ok=True)
    
    # 读取和处理jsonl文件
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=0):  # 从0开始编号
            try:
                data = json.loads(line.strip())
                
                # 提取human_answers和chatgpt_answers
                human_answer = data.get('human_answers', [''])[0]  # 取第一条答案
                chatgpt_answer = data.get('chatgpt_answers', [''])[0]

                # 将所有换行符替换成空字符串，使文本成为单行
                human_answer = human_answer.replace('\\n', '').replace('\n', '').replace('\r', '')
                chatgpt_answer = chatgpt_answer.replace('\\n', '').replace('\n', '').replace('\r', '')
                
                # 生成图片
                if human_answer:
                    human_img_path = os.path.join(human_folder, f"{line_num+1}.png")
                    text_to_fixed_image_sure(human_answer, human_img_path)
                    print(f"✓ 已生成 human 图片: {line_num+1}.png")
                
                if chatgpt_answer:
                    ai_img_path = os.path.join(ai_folder, f"{line_num+1}.png")
                    text_to_fixed_image_sure(chatgpt_answer, ai_img_path)
                    print(f"✓ 已生成 ai 图片: {line_num+1}.png")
                    
            except json.JSONDecodeError as e:
                print(f"✗ 第 {line_num+1} 行JSON解析失败: {e}")
            except Exception as e:
                print(f"✗ 第 {line_num+1} 行处理失败: {e}")


def process_nlpcc_test_data(json_path, output_base_dir):
    """
    处理 NLPCC 测试数据，根据 label 字段将数据转换为图片
    
    参数：
    - json_path: JSON 文件路径 (train.json 或 test.json)
    - output_base_dir: 输出基础目录路径 (dataset/detectRL-zh/test)
    
    label = 0 -> human 文件夹
    label = 1 -> ai 文件夹
    """
    
    # 创建输出目录
    human_dir = os.path.join(output_base_dir, "human")
    ai_dir = os.path.join(output_base_dir, "ai")
    
    os.makedirs(human_dir, exist_ok=True)
    os.makedirs(ai_dir, exist_ok=True)
    
    if not os.path.exists(json_path):
        print(f"✗ 文件不存在: {json_path}")
        return
    
    human_count = 0
    ai_count = 0
    error_count = 0
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
        
        for item_id, item in enumerate(data_list, start=1):
            try:
                # 获取关键字段
                text = item.get('text', '').strip()
                label = item.get('label', -1)
                item_id_val = item.get('id', item_id)
                
                # 清理文本中的特殊字符
                text = text.replace('\\n', '').replace('\n', '').replace('\r', '')
                
                if not text:
                    print(f"⚠ ID {item_id_val}：文本为空，跳过")
                    continue
                
                # 根据 label 判断输出目录
                if label == 0:
                    # human 标签
                    output_path = os.path.join(human_dir, f"{item_id_val}.png")
                    text_to_fixed_image_sure(text, output_path)
                    print(f"✓ 已生成 human 图片: {item_id_val}.png")
                    human_count += 1
                    
                elif label == 1:
                    # ai 标签
                    output_path = os.path.join(ai_dir, f"{item_id_val}.png")
                    text_to_fixed_image_sure(text, output_path)
                    print(f"✓ 已生成 ai 图片: {item_id_val}.png")
                    ai_count += 1
                    
                else:
                    print(f"⚠ ID {item_id_val}：标签值 '{label}' 不识别，跳过")
                    
            except Exception as e:
                print(f"✗ ID {item_id_val} 处理失败: {e}")
                error_count += 1
        
        # 打印统计结果
        print(f"\n{'='*50}")
        print(f"📊 处理结果统计:")
        print(f"   - human 图片: {human_count} 张")
        print(f"   - ai 图片: {ai_count} 张")
        print(f"   - 错误: {error_count} 条")
        print(f"   - 总计: {human_count + ai_count} 张")
        print(f"{'='*50}\n")
        
    except json.JSONDecodeError as e:
        print(f"✗ JSON 解析失败: {e}")
    except Exception as e:
        print(f"✗ 处理过程出错: {e}")

def process_nlpcc_train_data(json_path, output_base_dir):
    """
    处理 NLPCC 测试数据，根据 label 字段将数据转换为图片
    
    参数：
    - json_path: JSON 文件路径 (train.json 或 test.json)
    - output_base_dir: 输出基础目录路径 (dataset/detectRL-zh/test)
    
    label = 0 -> human 文件夹
    label = 1 -> ai 文件夹
    """
    
    # 创建输出目录
    human_dir = os.path.join(output_base_dir, "human")
    ai_dir = os.path.join(output_base_dir, "ai")
    
    os.makedirs(human_dir, exist_ok=True)
    os.makedirs(ai_dir, exist_ok=True)
    
    if not os.path.exists(json_path):
        print(f"✗ 文件不存在: {json_path}")
        return
    
    human_count = 0
    ai_count = 0
    error_count = 0
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
        
        for item_id, item in enumerate(data_list, start=1):
            try:
                # 获取关键字段
                text = item.get('text', '').strip()
                label = item.get('label', -1)
                item_id_val = item.get('id', item_id)
                
                # 清理文本中的特殊字符
                text = text.replace('\\n', '').replace('\n', '').replace('\r', '')
                
                if not text:
                    print(f"⚠ ID {item_id_val}：文本为空，跳过")
                    continue
                
                # 根据 label 判断输出目录
                if label == 0:
                    # human 标签
                    output_path = os.path.join(human_dir, f"{item_id_val}.png")
                    text_to_fixed_image_sure(text, output_path)
                    print(f"✓ 已生成 human 图片: {item_id_val}.png")
                    human_count += 1
                    
                elif label == 1:
                    # ai 标签
                    output_path = os.path.join(ai_dir, f"{item_id_val}.png")
                    text_to_fixed_image_sure(text, output_path)
                    print(f"✓ 已生成 ai 图片: {item_id_val}.png")
                    ai_count += 1
                    
                else:
                    print(f"⚠ ID {item_id_val}：标签值 '{label}' 不识别，跳过")
                    
            except Exception as e:
                print(f"✗ ID {item_id_val} 处理失败: {e}")
                error_count += 1
        
        # 打印统计结果
        print(f"\n{'='*50}")
        print(f"📊 处理结果统计:")
        print(f"   - human 图片: {human_count} 张")
        print(f"   - ai 图片: {ai_count} 张")
        print(f"   - 错误: {error_count} 条")
        print(f"   - 总计: {human_count + ai_count} 张")
        print(f"{'='*50}\n")
        
    except json.JSONDecodeError as e:
        print(f"✗ JSON 解析失败: {e}")
    except Exception as e:
        print(f"✗ 处理过程出错: {e}")


# 使用示例
if __name__ == "__main__":
    # 处理 NLPCC 测试数据
    print("开始处理 NLPCC 测试数据...\n")
    
    json_file = r"d:\Desktop\文本检测\NLPCC-2025-Task1-main\data\train.json"
    output_dir = r"d:\Desktop\文本检测\HC3-Chinese\dataset\detectRL-zh\train"
    
    process_nlpcc_train_data(json_file, output_dir)

'''
# 使用示例
if __name__ == "__main__":
    # 方式1: 处理 clean_hc3_qa 数据
    print("开始处理 clean_hc3_qa 数据...\n")
    
    train_jsonl = r"d:\Desktop\文本检测\HC3-Chinese\clean_hc3_qa\train.jsonl"
    test_jsonl = r"d:\Desktop\文本检测\HC3-Chinese\clean_hc3_qa\test.jsonl"
    output_dir = r"d:\Desktop\文本检测\HC3-Chinese\dataset\clean"
    
    process_clean_hc3_qa(train_jsonl, test_jsonl, output_dir)

'''