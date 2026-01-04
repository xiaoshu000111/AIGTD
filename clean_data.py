import json
import os
'''
这个是从某个github中下载的关于清理jsonl文件中换行符的脚本，它产生了clean_hc3_qa文件夹相关文件。
'''
def clean_jsonl_newlines(input_file, output_file):
    """
    清除JSONL文件中的换行符
    
    参数：
    - input_file: 输入的JSONL文件路径
    - output_file: 输出的清洁后JSONL文件路径
    """
    count = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(infile, start=1):
                try:
                    # 解析JSON对象
                    data = json.loads(line.strip())
                    
                    # 清除各字段中的换行符和回车符
                    if 'question' in data:
                        data['question'] = data['question'].replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
                    
                    if 'human_answers' in data and isinstance(data['human_answers'], list):
                        data['human_answers'] = [
                            answer.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
                            for answer in data['human_answers']
                        ]
                    
                    if 'chatgpt_answers' in data and isinstance(data['chatgpt_answers'], list):
                        data['chatgpt_answers'] = [
                            answer.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
                            for answer in data['chatgpt_answers']
                        ]
                    
                    # 写入清洁后的JSON
                    outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                    count += 1
                    
                    if line_num % 100 == 0:
                        print(f"✓ 已处理 {line_num} 行")
                    
                except json.JSONDecodeError as e:
                    print(f"✗ 第 {line_num} 行JSON解析失败: {e}")
                except Exception as e:
                    print(f"✗ 第 {line_num} 行处理失败: {e}")
        
        print(f"\n✓ 清洁完成！共处理 {count} 条记录")
        print(f"✓ 输出文件: {output_file}")
        
    except FileNotFoundError:
        print(f"✗ 文件不存在: {input_file}")
    except Exception as e:
        print(f"✗ 处理过程出错: {e}")


# 使用示例
if __name__ == "__main__":
    input_jsonl = r"d:\Desktop\文本检测\HC3-Chinese\all.jsonl"
    output_jsonl = r"d:\Desktop\文本检测\HC3-Chinese\clean.jsonl"
    
    print("开始清除JSONL文件中的换行符...")
    clean_jsonl_newlines(input_jsonl, output_jsonl)