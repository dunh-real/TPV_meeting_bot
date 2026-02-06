import ollama

model_name = "sailor2:1b"

try:
    content = str(input("Hỏi đê bạn êi: "))
    stream = ollama.chat(
        model = model_name,
        messages = [{"role": "user", "system_prompt": "Trả lời ngắn gọn, đúng trọng tâm. KHÔNG trả lời dài dòng, thừa thãi.", "content": content}],
        stream = True,
    )
    for chunk in stream:
        print(chunk["message"]["content"], end = "", flush = True)
    print()

except ollama.ResponseError as e:
    print(f"Dmm code kiểu dell gì có lỗi rồi đây này: {e}")
except ConnectionRefusedError:
    print("Mất mịe mạng rồi bạn trẻ êi")
    

class LLMService:
    def __init__(self):
        self.model_name = None
    
    def stream(self, content: str) -> str:
        try:
            stream = ollama.chat(
                model = self.model_name,
                messages = [{"role": "user", "system_prompt": "Trả lời ngắn gọn, đúng trọng tâm.", "content": content}],
                stream = True,
            )
            for chunk in stream:
                print(chunk["message"]["content"], end = "", flush = True)
            print()
        except ollama.ResponseError as e:
            print(f"Dmm code kiểu dell gì có lỗi rồi đây này: {e}")
        except ConnectionRefusedError:
            print("Mất mịe mạng rồi bạn trẻ êi")