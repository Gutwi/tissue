import os
import random
from PIL import Image, ImageDraw

# 保存先フォルダ
output_dir = "dummy_images"
good_dir = "good"
defective_dir = "defective"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir+os.sep+good_dir, exist_ok=True)
os.makedirs(output_dir+os.sep+defective_dir, exist_ok=True)

# 画像サイズ
image_size = (224, 224)

# 図形を描画する関数
def draw_shape(draw, shape, color, size, position):
    x, y = position
    if shape == good_dir:
        draw.ellipse([x, y, x + size, y + size], fill=color, outline=color)
    elif shape == defective_dir:
        draw.line([x, y, x + size, y + size], fill=color, width=5)
        draw.line([x + size, y, x, y + size], fill=color, width=5)

# 図形ごとに画像を生成
def generate_images(shape, count):
    for i in range(count):
        img = Image.new("RGB", image_size, "white")
        draw = ImageDraw.Draw(img)

        # ランダムな大きさ、色、位置を設定
        size = random.randint(30, 80)
        x = random.randint(0, image_size[0] - size)
        y = random.randint(0, image_size[1] - size)
        color = tuple(random.randint(0, 255) for _ in range(3))

        draw_shape(draw, shape, color, size, (x, y))

        # ファイル名を設定して保存
        filename = f"{shape}_{i + 1}.png"
        # img.save(os.path.join(output_dir, filename))  #250124
        img.save(os.path.join(output_dir+os.sep+shape, filename))


# 画像生成
generate_images(good_dir, 100)
generate_images(defective_dir, 100)

print(f"Images saved in '{output_dir}'")
