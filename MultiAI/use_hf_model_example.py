"""
使用Hugging Face模型的示例 - 无需VPN
演示如何在项目中集成hf_model_downloader
"""

import os
import sys
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HF_Example")

# 导入下载工具
from hf_model_downloader import (
    setup_sentence_transformers, 
    setup_transformers, 
    download_sentence_transformer
)

def main():
    """主函数"""
    print("=" * 50)
    print("Hugging Face模型下载和使用示例 - 无需VPN")
    print("=" * 50)
    
    # 步骤1: 设置库使用镜像
    print("\n1. 设置库使用镜像")
    setup_transformers()
    setup_sentence_transformers()
    
    # 步骤2: 下载并使用sentence-transformers模型
    print("\n2. 下载并使用sentence-transformers模型")
    try:
        # 尝试导入sentence_transformers
        import sentence_transformers
        
        # 下载模型
        model_name = "paraphrase-MiniLM-L6-v2"
        print(f"正在下载模型: {model_name}")
        model_path = download_sentence_transformer(model_name)
        print(f"模型下载成功: {model_path}")
        
        # 加载模型
        print("正在加载模型...")
        model = sentence_transformers.SentenceTransformer(model_path)
        
        # 使用模型
        sentences = [
            "这是一个示例句子。",
            "这是另一个示例句子。",
            "这个句子与第一个句子相似。"
        ]
        
        print("计算句子嵌入...")
        embeddings = model.encode(sentences)
        
        print(f"生成的嵌入维度: {embeddings.shape}")
        
        # 计算相似度
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity([embeddings[0]], [embeddings[2]])[0][0]
        print(f"句子1和句子3的相似度: {similarity:.4f}")
        
    except ImportError:
        print("sentence-transformers未安装，请使用以下命令安装:")
        print("pip install sentence-transformers -i https://pypi.tuna.tsinghua.edu.cn/simple")
    except Exception as e:
        print(f"使用sentence-transformers时出错: {str(e)}")
    
    # 步骤3: 下载并使用transformers模型
    print("\n3. 下载并使用transformers模型")
    try:
        # 尝试导入transformers
        import transformers
        
        # 下载并加载模型
        print("正在下载并加载BERT模型...")
        tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-chinese")
        model = transformers.BertModel.from_pretrained("bert-base-chinese")
        
        # 使用模型
        text = "这是一个测试句子，用于演示如何使用BERT模型。"
        print(f"输入文本: {text}")
        
        # 编码文本
        inputs = tokenizer(text, return_tensors="pt")
        
        # 获取模型输出
        outputs = model(**inputs)
        
        # 打印结果
        print(f"BERT模型输出形状: {outputs.last_hidden_state.shape}")
        
    except ImportError:
        print("transformers未安装，请使用以下命令安装:")
        print("pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple")
    except Exception as e:
        print(f"使用transformers时出错: {str(e)}")
    
    print("\n示例运行完成!")

if __name__ == "__main__":
    main()
