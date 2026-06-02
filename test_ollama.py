import ollama

def test_ollama_connection():
    try:
        models = ollama.list()
        print("Ollama连接成功！")
        print("已安装的模型:")
        for model in models["models"]:
            print(f"  - {model['name']}")
        return True
    except Exception as e:
        print(f"Ollama连接失败: {e}")
        print("请确保Ollama已安装并运行")
        return False

def test_ollama_generate():
    try:
        response = ollama.generate(model="deepseek-r1:7b", prompt="你好，介绍一下你自己")
        print("\n模型响应:")
        print(response["response"])
        return True
    except Exception as e:
        print(f"生成响应失败: {e}")
        print("请确保已下载deepseek-r1:7b模型")
        return False

if __name__ == "__main__":
    print("=== 测试Ollama连接 ===")
    if test_ollama_connection():
        print("\n=== 测试Ollama生成 ===")
        test_ollama_generate()