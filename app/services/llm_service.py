import ollama

model_name = "qwen2.5:latest"

with open("../../transcript.txt", "r", encoding = "utf-8") as f:
    content = f.read()

try:
    # content = str(input("Hỏi đê bạn êi: "))
    content = str(content)
    stream = ollama.chat(
        model = model_name,
        messages = [{"role": "user", "system_prompt": "Trả lời ngắn gọn, đúng trọng tâm. KHÔNG trả lời dài dòng, thừa thãi. Ngôn ngữ: tiếng Việt", "content": f"Đây là nội dung của một bổi nói chuyện giữa nhiều người với nhau. Bạn hãy tóm tắt nội dung của buổi nói chuyện này giúp tôi: {content}"}],
        stream = True,
    )
    with open("output.txt", "w", encoding = "utf-8") as f:
        for chunk in stream:
            print(chunk["message"]["content"], end = "", flush = True)
            # print(type(chunk["message"]["content"]))
            f.write(str(chunk["message"]["content"]))
            
    print()
    # print(type(stream['message']['content']))

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