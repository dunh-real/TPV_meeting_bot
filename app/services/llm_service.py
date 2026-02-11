import ollama

MODEL_LLM_NAME = "qwen2.5:latest"
OUTPUT_PATH = "../../data/output"
    
class LLMService:
    def __init__(self, model_name = MODEL_LLM_NAME):
        self.model_name = model_name

    def llm_summarize(self, content: str, id):
        prompt = f"""
        # ROLE:
            Bạn là một Thư ký điều hành chuyên nghiệp, có khả năng phân tích hội thoại sắc bén và bóc tách dữ liệu nhân sự chính xác từ các bản ghi chép cuộc họp (CONTENT).        
        
        # TASK: 
            Phân tích nội dung <CONTENT> để thực hiện các yêu cầu sau:
            1. Xác định danh tính tất cả Speaker (Người nói). Quét kỹ TOÀN BỘ nội dung để lập danh sách Speaker không trùng lặp.
            2. Tổng hợp nội dung theo từng speaker về: Những quan điểm, báo cáo hoặc đề xuất quan trọng trong cuộc họp.
            3. Nội dung tổng hợp yêu cầu rõ ràng, đúng trọng tâm.

        # RULES (TUÂN THỦ TUYỆT ĐỐI):
            1. NGÔN NGỮ: Phản hồi HOÀN TOÀN bằng Tiếng Việt, văn phong chuyên nghiệp, súc tích.
            2. NGUYÊN TẮC TRUNG THỰC: Chỉ trích xuất thông tin có trong nội dung <CONTENT>. Tuyệt đối không tự suy diễn hoặc bổ sung thông tin ngoài luồng.
            3. OUTPUT FORMAT: Không thêm các ký tự Markdown, chỉ trình bày dạng văn bản thuần. Không trình bày lại các trích dẫn có trong dấu (). Không cần chỉ định rõ thời gian đưa ra hành động.
        
        # OUTPUT FORMAT: 
            I. TỔNG QUAN CUỘC HỌP
            - Nội dung chính: (Tóm tắt các nội dung được trình bày trong CONTENT).
            - Mục tiêu, kết quả: (Trình bày mục tiêu/kết quả đạt được trong cuộc họp).
            II. CHI TIẾT THEO NHÂN SỰ
            a. Speaker - (Tên Người Nói): (Tóm tắt nội dung liên quan đến người nói - Yêu cầu rõ ràng, chi tiết, đúng trọng tâm).
            (Lặp lại cấu trúc trên cho đến Speaker cuối cùng).
        
        # CONTENT:
        {content}        
        """

        try:
            stream = ollama.chat(
                model = self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temparature': 0.3},
            )

            response = stream['message']['content'].strip()

            output_path = f"{OUTPUT_PATH}/{id}_summarize.txt"

            with open(output_path, "w", encoding = "utf-8") as f:
                f.write(response)

        except ollama.ResponseError as e:
            print(f"Lỗi nè sếp: {e}")

        except ConnectionRefusedError:
            print("Lỗi nè sếp")