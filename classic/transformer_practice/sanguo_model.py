import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import json

from torch.utils.data import Dataset
from char_tokenizer import get_dataset

class PositionalEncoding(nn.Module):
    """位置编码模块"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class SanguoModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=6, seq_length=50):
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.nhead = nhead
        self.num_layers = num_layers
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_length)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # 词嵌入 + 缩放
        x = self.embedding(x) * math.sqrt(self.d_model)
        
        # 位置编码
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)
        
        # Transformer编码
        x = self.transformer(x)
        
        # 输出投影
        output = self.output_projection(x)
        
        return output
    
    def get_config(self):
        """获取模型配置"""
        return {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_layers': self.num_layers,
            'seq_length': self.seq_length
        }
    
    @classmethod
    def from_config(cls, config):
        """从配置创建模型"""
        return cls(**config)

def save_model_and_tokenizer(model, tokenizer, save_dir):
    """保存模型和tokenizer"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存模型权重
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))
    
    # 保存模型配置
    with open(os.path.join(save_dir, "config.json"), 'w', encoding='utf-8') as f:
        json.dump(model.get_config(), f, indent=2, ensure_ascii=False)
    
    # 保存tokenizer
    tokenizer_data = {
        'char_to_idx': tokenizer.char_to_idx,
        'idx_to_char': tokenizer.idx_to_char,
        'vocab_size': tokenizer.vocab_size
    }
    with open(os.path.join(save_dir, "tokenizer.json"), 'w', encoding='utf-8') as f:
        json.dump(tokenizer_data, f, indent=2, ensure_ascii=False)

def load_model_and_tokenizer(save_dir):
    """加载模型和tokenizer"""
    # 加载配置
    with open(os.path.join(save_dir, "config.json"), 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 创建并加载模型
    model = SanguoModel.from_config(config)
    model.load_state_dict(torch.load(os.path.join(save_dir, "model.pth"), map_location='cuda' if torch.cuda.is_available() else 'cpu'))
    
    # 加载tokenizer
    from char_tokenizer import CharacterTokenizer
    tokenizer = CharacterTokenizer()
    
    with open(os.path.join(save_dir, "tokenizer.json"), 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)
    
    tokenizer.char_to_idx = tokenizer_data['char_to_idx']
    tokenizer.idx_to_char = {int(k): v for k, v in tokenizer_data['idx_to_char'].items()}
    tokenizer.vocab_size = tokenizer_data['vocab_size']
    
    return model, tokenizer

def train_model(config):
    """配置驱动的训练函数"""
    print("=== 三国Transformer训练开始 ===")
    print(f"配置: {json.dumps(config, indent=2, ensure_ascii=False)}")
    
    # 1. 准备数据
    from read_text import clean_text, extract_text
    text = extract_text(config['data_path'])
    text = clean_text(text)
    dataset, tokenizer = get_dataset(text, 
                                   seq_length=config['model']['seq_length'], 
                                   batch_size=config['training']['batch_size'])
    
    # 2. 创建模型
    model_config = config['model'].copy()
    model_config['vocab_size'] = tokenizer.vocab_size
    model = SanguoModel(**model_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # 3. 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # 4. 训练循环
    model.train()
    total_loss = 0
    step = 0
    
    print(f"\n开始训练，总步数: {config['training']['num_steps']}")
    print("-" * 50)
    
    for epoch in range(config['training']['num_epochs']):
        for batch_idx, (features, labels) in enumerate(dataset):
            if step >= config['training']['num_steps']:
                break
            
            # 前向传播
            outputs = model(features)
            logits = outputs[:, -1, :]  # 取最后一个位置
            loss = F.cross_entropy(logits, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            step += 1
            
            # 打印进度
            if step % config['training']['log_interval'] == 0:
                avg_loss = total_loss / config['training']['log_interval']
                print(f"步骤 {step}/{config['training']['num_steps']} - 平均损失: {avg_loss:.4f}")
                total_loss = 0
        
        if step >= config['training']['num_steps']:
            break
    
    # 5. 保存模型
    if config['training']['save_model']:
        save_model_and_tokenizer(model, tokenizer, config['training']['save_dir'])
        print(f"\n模型已保存到: {config['training']['save_dir']}")
    
    print("\n训练完成！")
    return model, tokenizer

def simple_generate_demo():
    """简单的生成演示"""
    try:
        model, tokenizer = load_model_and_tokenizer("sanguo_checkpoint")
        model.eval()
        
        print("=== 简单生成测试 ===")
        test_texts = ["却说天下大势", "刘备", "关羽义薄云天"]
        
        with torch.no_grad():
            for prompt in test_texts:
                input_ids = tokenizer.encode(prompt)
                if len(input_ids) < model.seq_length:
                    input_ids = input_ids + [0] * (model.seq_length - len(input_ids))
                else:
                    input_ids = input_ids[:model.seq_length]
                
                input_tensor = torch.tensor([input_ids])
                outputs = model(input_tensor)
                
                # 预测下一个字符
                next_logits = outputs[0, len(tokenizer.encode(prompt))-1, :]
                predicted_id = torch.argmax(next_logits).item()
                predicted_char = tokenizer.decode([predicted_id])
                
                print(f"'{prompt}' → '{predicted_char}'")
                
    except Exception as e:
        print(f"生成演示失败: {e}")

# 默认配置
DEFAULT_CONFIG = {
    'data_path': 'data/三國演義.epub',
    'model': {
        'd_model': 128,
        'nhead': 8,
        'num_layers': 6,
        'seq_length': 50
    },
    'training': {
        'batch_size': 32,
        'learning_rate': 0.001,
        'num_epochs': 10,
        'num_steps': 1000,
        'log_interval': 100,
        'save_model': True,
        'save_dir': 'sanguo_checkpoint'
    }
}

if __name__ == "__main__":
    train_model(DEFAULT_CONFIG)
    # 简单生成演示
    simple_generate_demo()
