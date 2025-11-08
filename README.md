# 👩‍💻 Gender Detector using CNN | تشخیص جنسیت با شبکه عصبی

A deep learning-based gender detection project using Convolutional Neural Networks (CNNs) to classify human faces as male or female.  
The model is trained on the UTKFace dataset, which contains diverse human face images labeled with gender and age.

پروژه‌ای برای تشخیص جنسیت چهره‌ها با استفاده از شبکه عصبی کانولوشنی (CNN).  
مدل روی دیتاست UTKFace آموزش دیده و قادر است تصاویر چهره را به دو دسته "زن" و "مرد" تقسیم کند.

---

## 🧠 Technologies Used | تکنولوژی‌های استفاده‌شده

- Python 3.10+  
- TensorFlow / Keras (برای ساخت مدل CNN)  
- OpenCV (برای پردازش تصویر)  
- NumPy & Matplotlib (برای پردازش داده و نمایش تصاویر)  
- UTKFace dataset (تصاویر چهره انسان‌ها با برچسب جنسیت)

---

## ⚙️ How It Works | نحوه کار

1. Load face images and labels from the UTKFace dataset.  
2. Preprocess images: resize to 64x64 and normalize pixel values.  
3. Split dataset into train and test sets.  
4. Build a CNN model with Conv2D, MaxPooling, Flatten, Dense, and Dropout layers.  
5. Train the model and validate performance.  
6. Evaluate test accuracy and visualize predictions vs. real gender labels.

مراحل کار:  
1. بارگذاری تصاویر و برچسب‌ها از دیتاست UTKFace  
2. پیش‌پردازش تصاویر: تغییر اندازه به 64x64 و نرمال‌سازی  
3. تقسیم داده‌ها به مجموعه‌های آموزش و تست  
4. تعریف مدل CNN با لایه‌های Conv2D، MaxPooling، Flatten، Dense و Dropout  
5. آموزش مدل و بررسی عملکرد روی داده‌های اعتبارسنجی  
6. ارزیابی دقت و نمایش تصاویر با جنسیت واقعی و پیش‌بینی شده

---

## 🧩 Key Code Structure | ساختار اصلی کد

```python
# Load dataset & preprocess
# - Read images from UTKFace
# - Resize & normalize
# - Convert gender labels to categorical
...

# Train-test split
...

# Define CNN model
# - Conv2D + MaxPooling + Flatten + Dense + Dropout
...

# Train the model
# model.fit(...)

# Evaluate model
# loss, acc = model.evaluate(...)

# Predict & visualize
# for each test image: show real vs predicted
...
