import torch
import torch.nn.functional as F
from sanguo_model import load_model_and_tokenizer

class SanguoChat:
    """三国演义风格的聊天模型"""
    
    def __init__(self, model_dir="../../sanguo_checkpoint"):
        print("正在加载三国演义模型...")
        self.model, self.tokenizer = load_model_and_tokenizer(model_dir)
        self.model.eval()
        print("模型加载完成！")
        print(f"词汇表大小: {self.tokenizer.vocab_size}")
        
    def generate_text(self, prompt, max_length=100, temperature=1.0, top_k=50):
        """生成文本"""
        # 编码输入
        input_ids = self.tokenizer.encode(prompt)
        if len(input_ids) == 0:
            input_ids = [self.tokenizer.char_to_idx.get('<start>', 0)]
        
        generated = input_ids.copy()
        
        with torch.no_grad():
            for _ in range(max_length):
                # 准备输入序列
                if len(generated) >= self.model.seq_length:
                    # 如果序列太长，取最后seq_length个token
                    input_sequence = generated[-self.model.seq_length:]
                else:
                    # 如果序列太短，用padding填充
                    input_sequence = generated + [0] * (self.model.seq_length - len(generated))
                
                input_tensor = torch.tensor([input_sequence])
                
                # 模型预测
                outputs = self.model(input_tensor)
                
                # 取最后一个位置的logits（对应下一个token的预测）
                next_token_logits = outputs[0, min(len(generated)-1, self.model.seq_length-1), :]
                
                # 应用temperature
                next_token_logits = next_token_logits / temperature
                
                # Top-k采样
                if top_k > 0:
                    # 保留top-k个最可能的token
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    # 将其他token的概率设为负无穷
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(0, top_k_indices, top_k_logits)
                
                # 计算概率分布
                probs = F.softmax(next_token_logits, dim=-1)
                
                # 采样下一个token
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                # 检查是否是结束符或特殊符号
                if next_token == self.tokenizer.char_to_idx.get('<end>', -1):
                    break
                if next_token == 0 or next_token >= self.tokenizer.vocab_size:
                    continue
                    
                generated.append(next_token)
                
                # 解码并检查是否应该停止
                current_text = self.tokenizer.decode(generated[len(input_ids):])
                if len(current_text) > 0 and current_text[-1] in ['。', '！', '？', '\n']:
                    break
        
        # 解码生成的文本
        generated_text = self.tokenizer.decode(generated[len(input_ids):])
        return generated_text
    
    def chat(self, user_input, max_length=50, temperature=0.8, top_k=30):
        """聊天对话"""
        # 生成回复
        response = self.generate_text(user_input, max_length, temperature, top_k)
        
        # 清理回复（移除可能的乱码或重复）
        response = response.strip()
        if not response:
            response = "..."
            
        return response
    
    def interactive_chat(self):
        """交互式聊天"""
        print("\n" + "="*50)
        print("🎭 欢迎与三国演义风格的AI对话！")
        print("💡 试试说：'却说天下大势' 或 '刘备如何？'")
        print("⚡ 输入 'quit' 退出对话")
        print("="*50)
        
        while True:
            try:
                user_input = input("\n你: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                    print("\n再会！愿你如关羽般义薄云天！🌟")
                    break
                
                if not user_input:
                    continue
                
                print("\n三国AI: ", end="", flush=True)
                
                # 生成回复
                response = self.chat(user_input, max_length=80, temperature=0.8, top_k=30)
                
                # 打字机效果显示
                import time
                for char in response:
                    print(char, end="", flush=True)
                    time.sleep(0.05)  # 调整打字速度
                
                print()  # 换行
                
            except KeyboardInterrupt:
                print("\n\n程序被中断，再见！")
                break
            except Exception as e:
                print(f"\n发生错误: {e}")
                continue

def quick_test(chat_bot):
    """快速测试生成效果"""
    print("\n🔍 快速测试模型生成效果:")
    print("-" * 40)
    
    test_prompts = [
        "却说天下大势",
        "刘备",
        "关羽",
        "诸葛亮",
        "三国"
    ]
    
    for prompt in test_prompts:
        response = chat_bot.chat(prompt, max_length=30, temperature=0.7)
        print(f"输入: '{prompt}' → AI: {response}")
    
    print("-" * 40)

if __name__ == "__main__":
    try:
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        model_dir = os.path.join(project_root, "sanguo_checkpoint")
        # 创建聊天机器人 - 正确的路径
        chat_bot = SanguoChat(model_dir)
        
        # 快速测试
        quick_test(chat_bot)
        
        # 开始交互式对话
        chat_bot.interactive_chat()
        
    except FileNotFoundError as e:
        print("❌ 找不到模型文件！")
        print(f"错误详情: {e}")
        print("请确保已经训练并保存了模型到 'sanguo_checkpoint' 目录")
        print("\n当前查找路径:")
        print("- 脚本位置: classic/transformer_practice/")
        print("- 模型路径: ../../sanguo_checkpoint (即项目根目录下的sanguo_checkpoint)")
    except Exception as e:
        print(f"❌ 启动聊天失败: {e}")
        print("请检查模型文件是否完整")
