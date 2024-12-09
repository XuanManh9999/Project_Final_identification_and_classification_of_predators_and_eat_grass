from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Load mô hình ResNet50 đã huấn luyện sẵn trên ImageNet
model = ResNet50(weights='imagenet')

# Danh sách động vật ăn thịt (carnivores) mở rộng
carnivores = [
    'lion', 'tiger', 'wolf', 'leopard', 'cheetah', 'hyena', 'fox', 'panther', 
    'jaguar', 'cougar', 'lynx', 'bobcat', 'crocodile', 'alligator', 'shark',
    'eagle', 'hawk', 'falcon', 'vulture', 'owl', 'python', 'cobra', 'orca',
    'polar bear', 'grizzly bear', 'puma', 'komodo dragon', 'hyena', 'jackal',
    'wolverine', 'otter', 'coyote', 'cheetah', 'wildcat', 'kingfisher',
    'saber-toothed tiger', 'harpy eagle', 'snow leopard', 'barbary lion', 'panthera',
    'sea lion', 'dingo', 'manta ray', 'piranha', 'falcon', 'golden eagle', 'wolf spider', 
    'swan', 'geese', 'sparrowhawk', 'vulture', 'kestrel', 'great white shark', 'puma', 'bald_eagle',
    'African_crocodile', 'quoll', 'coati', 'meerkat', 'stoat', 'fossa', 'caracal', 'ocelot', 'kookaburra',
    'mole', 'harpy eagle', 'badger', 'golden jackal', 'lynx', 'ring-tailed lemur', 'European wildcat',
    'crocodile', 'sea snake', 'albatross', 'vulture', 'black mamba', 'snow leopard', 'jaguarundi', 'shark',
    'seal', 'killer whale', 'cuttlefish'
]

# Danh sách động vật ăn cỏ (herbivores) mở rộng
herbivores = [
    'elephant', 'deer', 'cow', 'giraffe', 'rabbit', 'zebra', 'goat', 'sheep',
    'horse', 'buffalo', 'camel', 'alpaca', 'donkey', 'antelope', 'bison', 
    'kangaroo', 'koala', 'panda', 'manatee', 'sloth', 'rhinoceros', 'hippopotamus',
    'yak', 'moose', 'llama', 'guinea pig', 'capybara', 'ibex', 'bighorn', 'platypus',
    'tortoise', 'armadillo', 'peacock', 'flamingo', 'water buffalo', 'okapi',
    'wallaby', 'reindeer', 'gazelle', 'buffalo', 'elk', 'ostrich', 'gibbon', 'camel', 
    'alpaca', 'aardvark', 'tapir', 'chinchilla', 'bison', 'squirrel', 'porcupine', 'red panda',
    'giraffe', 'pronghorn', 'sable antelope', 'wild boar', 'elk', 'moose', 'beetle', 'clownfish',
    'koala', 'groundhog', 'platypus', 'mandrill', "water_buffalo", 'swan', 'elephant seal', 'bison',
    'meerkat', 'antelope', 'hippopotamus', 'camel', 'cow', 'ox', 'donkey', 'mule', 'bison', 'bighorn sheep',
    'manatee', 'kangaroo', 'llama', 'hippopotamus', 'rheas', 'yaks', 'guinea fowl', 'pigeon', 'butterfly',
    'elk', 'pronghorn antelope', 'llama', 'giraffe', 'pudu', 'okapi', 'gazelle', 'quokka', 'pika', 'caribou'
]

# Hàm tiền xử lý ảnh
def preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Hàm dự đoán loại động vật và phân loại ăn thịt/ăn cỏ
def predict_animal_and_type(model, image_path):
    processed_img = preprocess_image(image_path)
    prediction = model.predict(processed_img)
    decoded = decode_predictions(prediction, top=3)[0]  # Giải mã nhãn dự đoán

    # Phân loại dựa trên Top-3 predictions
    carnivore_count = sum(1 for _, label, _ in decoded if label in carnivores)
    herbivore_count = sum(1 for _, label, _ in decoded if label in herbivores)

    # Quyết định chế độ ăn
    if carnivore_count > herbivore_count:
        animal_type = 'Carnivore (Động vật ăn thịt)'
    elif herbivore_count > carnivore_count:
        animal_type = 'Herbivore (Động vật ăn cỏ)'
    else:
        animal_type = 'Unknown (Không xác định)'

    return decoded, animal_type

# Hàm hiển thị ảnh và xử lý điều hướng
def display_images_with_navigation(model, image_dir):
    img_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
    img_paths = [os.path.join(image_dir, img) for img in img_files]
    current_index = [0]  # Sử dụng danh sách để giữ giá trị tham chiếu

    def update_plot():
        img_path = img_paths[current_index[0]]
        decoded, animal_type = predict_animal_and_type(model, img_path)

        # Đọc ảnh
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Vẽ lại hình ảnh
        ax.clear()
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"{decoded[0][1]} - {animal_type}\nTop-3 Predictions:\n" +
                     "\n".join([f"{i+1}. {label}: {score*100:.2f}%" 
                                for i, (_, label, score) in enumerate(decoded)]),
                     fontsize=10, loc='center', pad=20)
        fig.canvas.draw()

    def on_click(event):
        if event.key == 'right':  # Phím 'Next'
            current_index[0] = (current_index[0] + 1) % len(img_paths)
        elif event.key == 'left':  # Phím 'Previous'
            current_index[0] = (current_index[0] - 1) % len(img_paths)
        update_plot()

    # Tạo giao diện Matplotlib
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.canvas.mpl_connect('key_press_event', on_click)  # Gắn sự kiện phím
    update_plot()
    plt.show()

# Đường dẫn thư mục chứa ảnh
image_dir = "data/animal"

# Hiển thị với điều hướng
display_images_with_navigation(model, image_dir)