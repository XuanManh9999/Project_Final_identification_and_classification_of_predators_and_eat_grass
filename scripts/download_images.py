from icrawler.builtin import GoogleImageCrawler
import os

# Danh sách động vật ăn thịt
carnivores = [
    'lion', 'tiger', 'wolf', 'leopard', 'cheetah', 'hyena', 'fox', 'panther', 
    'jaguar', 'cougar', 'lynx', 'bobcat', 'crocodile', 'alligator', 'shark',
    'eagle', 'hawk', 'falcon', 'vulture', 'owl', 'python', 'cobra', 'orca',
    'polar bear', 'grizzly bear', 'puma', 'komodo dragon'
]

# Danh sách động vật ăn cỏ
herbivores = [
    'elephant', 'deer', 'cow', 'giraffe', 'rabbit', 'zebra', 'goat', 'sheep',
    'horse', 'buffalo', 'camel', 'alpaca', 'donkey', 'antelope', 'bison', 
    'kangaroo', 'koala', 'panda', 'manatee', 'sloth', 'rhinoceros', 'hippopotamus',
    'yak', 'moose', 'llama', 'guinea pig'
]

def download_images(keywords, output_dir, max_num=50):
    for keyword in keywords:
        # Tạo thư mục cho từ khóa nếu chưa có
        keyword_dir = os.path.join(output_dir, keyword.replace(" ", "_"))
        os.makedirs(keyword_dir, exist_ok=True)
        
        # Cấu hình GoogleImageCrawler để lưu ảnh vào thư mục tương ứng
        crawler = GoogleImageCrawler(storage={"root_dir": keyword_dir})
        
        # Tải ảnh và gán nhãn dựa trên từ khóa
        print(f"Downloading images for: {keyword}")
        crawler.crawl(keyword=keyword, max_num=max_num)

if __name__ == "__main__":
    # Kết hợp danh sách động vật ăn thịt và ăn cỏ
    keywords = carnivores + herbivores  # Các từ khóa động vật cần thu thập thêm
    
    output_dir = "data/raw"  # Thư mục lưu dữ liệu
    max_num = 5            # Số lượng ảnh mỗi từ khóa

    download_images(keywords, output_dir, max_num)
