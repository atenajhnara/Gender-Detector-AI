# ==========================================
# 🤖 اجرای مدل تشخیص جنسیت با DeepFace
# ==========================================
import os
import cv2    #برای پردازش تصویر
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# -----------------------
# 1️⃣ بارگذاری دیتاست
# -----------------------
folder = r"C:\Users\Asus\.cache\kagglehub\datasets\jangedoo\utkface-new\versions\1\crop_part1"
files = [f for f in os.listdir(folder) if f.lower().endswith(".jpg")]

images = []
labels = []
filenames = []

for f in files:
    parts = f.split('_')  # نام فایل UTKFace: age_gender_race_date.jpg
    if len(parts) < 2:
        continue
    try:
        gender = int(parts[1])  # 0=male, 1=female
    except:
        continue
    if gender not in[0,1]: #به غیر از 1 و 0 بقیه رو رها میکنه
        continue
    path = os.path.join(folder, f)
    img = cv2.imread(path)
    if img is None:
        continue
    img = cv2.resize(img, (64,64))
    img = img / 255.0
    images.append(img)
    labels.append(gender)
    filenames.append(f)

images = np.array(images)
labels = np.array(labels)
labels_cat = to_categorical(labels, num_classes=2)

# -----------------------
# 2️⃣ تقسیم train/test
# -----------------------
X_train, X_test, y_train, y_test, files_train, files_test = train_test_split(
    images, labels_cat, filenames, test_size=0.2, random_state=42
)

# -----------------------
# 3️⃣ تعریف مدل CNN
# -----------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# -----------------------
# 4️⃣ آموزش مدل
# -----------------------
history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.1)

# -----------------------
# 5️⃣ ارزیابی مدل
# -----------------------
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {acc*100:.2f}%")

# -----------------------
# 6️⃣ پیش‌بینی و نمایش جنسیت واقعی و پیش‌بینی شده
# -----------------------
for i in range(len(X_test)):
    img = X_test[i]
    real_gender = "male" if np.argmax(y_test[i])==0 else "female"
    pred = model.predict(np.expand_dims(img, axis=0))
    pred_gender = "male" if np.argmax(pred)==0 else "female"

    img_unit8=(img*255).astype(np.uint8)
    img_rgb=cv2.cvtColor(img_unit8,cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title(f"Real: {real_gender} | Pred: {pred_gender}")
    plt.show()