# Báo cáo Machine Learning: Nhận dạng Kanji bằng HOG và các mô hình học máy cổ điển

## 2. Dataset Description

### 2.1. Nguồn dữ liệu

Dataset được sử dụng trong notebook `Kanji_HOG_ML_Colab.ipynb` được đọc từ Google Drive với đường dẫn:

```python
DATASET_DIR = '/content/drive/MyDrive/dataset_output_N5'
```

Theo cấu trúc xử lý trong code, dataset được tổ chức theo dạng thư mục, trong đó mỗi thư mục con tương ứng với một lớp Kanji. Tên thư mục con được dùng trực tiếp làm nhãn của ảnh.

Nguồn gốc ban đầu của dataset, ví dụ dataset được thu thập từ đâu, do ai cung cấp, hoặc có giấy phép sử dụng như thế nào, không được mô tả trong source code nên không thể xác định chắc chắn từ mã nguồn.

### 2.2. Số lượng mẫu

Notebook thống kê dataset và cho kết quả như sau:

| Thông tin | Giá trị |
|---|---:|
| Tổng số lớp Kanji | 80 |
| Tổng số ảnh | 16.079 |
| Số ảnh trung bình mỗi lớp | 201,0 |
| Số ảnh ít nhất trong một lớp | 200 |
| Số ảnh nhiều nhất trong một lớp | 201 |
| Ảnh bị bỏ qua do lỗi đọc ảnh | 0 |

Dataset tương đối cân bằng vì mỗi lớp có khoảng 200 hoặc 201 ảnh.

### 2.3. Classes/labels

Nhãn được lấy từ tên thư mục con trong `DATASET_DIR`. Code không khai báo thủ công danh sách nhãn mà tự động duyệt thư mục:

```python
for class_entry in sorted(os.scandir(DATASET_DIR), key=lambda e: e.name):
    if not class_entry.is_dir():
        continue
    label = class_entry.name
```

Sau đó, nhãn dạng chuỗi được mã hóa bằng `LabelEncoder`:

```python
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
```

Notebook hiển thị mapping giữa `Class ID` và Kanji, nhưng toàn bộ danh sách 80 ký tự không được xuất đầy đủ trong output dạng văn bản. Vì vậy, có thể xác định rằng có 80 nhãn Kanji, nhưng không thể liệt kê đầy đủ tất cả nhãn chỉ dựa trên phần output đã lưu trong source.

### 2.4. Các bước tiền xử lý dữ liệu

Các bước tiền xử lý chính trong code gồm:

| Bước | Mô tả |
|---|---|
| Đọc ảnh | Đọc ảnh bằng OpenCV ở chế độ grayscale. |
| Hỗ trợ đường dẫn Unicode | Dùng `np.fromfile` và `cv2.imdecode` để đọc file có tên hoặc đường dẫn Unicode. |
| Resize | Không resize ảnh trong hàm tiền xử lý, dù biến `IMG_SIZE = (64, 64)` được khai báo. |
| Nhị phân hóa/Otsu | Không sử dụng Otsu hoặc thresholding. |
| Trích xuất đặc trưng | Dùng HOG để chuyển ảnh thành vector đặc trưng. |
| Chuẩn hóa | Dùng `StandardScaler` trong pipeline học máy. |
| Giảm chiều | Dùng `PCA(n_components=0.95)` để giữ 95% phương sai. |

Hàm tiền xử lý:

```python
def preprocess_image(path):
    img = imread_unicode(path)
    if img is None:
        return None
    return img
```

Như vậy, tiền xử lý ảnh ở mức tối thiểu: chỉ đọc ảnh grayscale, không thay đổi kích thước, không lọc nhiễu, không tăng cường dữ liệu.

### 2.5. Chiến lược chia train/validation/test

Code chia dữ liệu thành train và test theo tỷ lệ 80/20:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=SEED, stratify=y_encoded
)
```

Kết quả:

| Tập dữ liệu | Số mẫu |
|---|---:|
| Train | 12.863 |
| Test | 3.216 |

Tham số `stratify=y_encoded` đảm bảo phân phối lớp trong tập train và test gần giống với phân phối lớp ban đầu.

Trong code không có tập validation riêng. Việc đánh giá bổ sung được thực hiện bằng `StratifiedKFold` 5-fold cross-validation trên toàn bộ dữ liệu cho top 3 mô hình.

## 3. Code Analysis

### 3.1. Tổng quan file nguồn

Trong workspace chỉ có một file source chính:

| File | Vai trò |
|---|---|
| `Kanji_HOG_ML_Colab.ipynb` | Notebook Google Colab thực hiện toàn bộ pipeline: cài thư viện, đọc dữ liệu, phân tích dataset, trích xuất HOG, huấn luyện nhiều mô hình ML, tinh chỉnh mô hình tốt nhất, đánh giá và trực quan hóa kết quả. |

File `kanji_hog_ml_colab.py` đang mở trong IDE nhưng không tồn tại trong thư mục dự án tại thời điểm phân tích. Vì vậy báo cáo chỉ dựa trên file thực tế có trong project là `Kanji_HOG_ML_Colab.ipynb`.

### 3.2. `Kanji_HOG_ML_Colab.ipynb`

#### Mục đích của file

Notebook xây dựng hệ thống nhận dạng ký tự Kanji N5 bằng phương pháp:

1. Đọc ảnh Kanji từ thư mục dataset.
2. Trích xuất đặc trưng HOG từ ảnh grayscale.
3. Chuẩn hóa đặc trưng bằng `StandardScaler`.
4. Giảm chiều bằng PCA.
5. Huấn luyện và so sánh nhiều mô hình học máy cổ điển.
6. Chọn mô hình tốt nhất dựa trên Accuracy, Macro Precision, Macro Recall và Macro F1.
7. Lưu mô hình đã huấn luyện bằng `joblib`.

#### Các thư viện/framework được sử dụng

| Thư viện | Mục đích sử dụng |
|---|---|
| `os`, `random`, `warnings` | Duyệt thư mục, lấy mẫu ngẫu nhiên, ẩn cảnh báo. |
| `cv2` | Đọc và giải mã ảnh grayscale. |
| `numpy` | Xử lý mảng số, đọc file ảnh bằng `np.fromfile`. |
| `pandas` | Tạo bảng thống kê dataset và kết quả mô hình. |
| `matplotlib`, `seaborn` | Vẽ biểu đồ phân phối, PCA, so sánh metric, confusion matrix. |
| `skimage.feature.hog` | Trích xuất đặc trưng HOG. |
| `skimage.exposure` | Điều chỉnh cường độ ảnh HOG khi trực quan hóa. |
| `sklearn.pipeline.Pipeline` | Đóng gói scaler, PCA và model thành một pipeline. |
| `StandardScaler` | Chuẩn hóa đặc trưng trước PCA và mô hình. |
| `PCA` | Giảm chiều đặc trưng HOG. |
| `train_test_split` | Chia dữ liệu train/test. |
| `StratifiedKFold`, `cross_val_score` | Đánh giá cross-validation có giữ phân phối lớp. |
| `accuracy_score`, `classification_report`, `confusion_matrix` | Tính metric đánh giá. |
| `SVC`, `LinearSVC` | Mô hình SVM kernel RBF và tuyến tính. |
| `RandomForestClassifier` | Mô hình rừng ngẫu nhiên. |
| `LogisticRegression` | Mô hình tuyến tính cho phân loại đa lớp. |
| `KNeighborsClassifier` | Mô hình KNN. |
| `GaussianNB` | Mô hình Naive Bayes Gaussian. |
| `XGBClassifier` | Mô hình gradient boosting từ XGBoost. |
| `joblib` | Lưu pipeline mô hình ra file `.pkl`. |
| `google.colab.drive` | Mount Google Drive để đọc dataset và lưu model. |

`LGBMClassifier` được import nhưng không được đưa vào dictionary `models`, vì vậy LightGBM không thực sự được huấn luyện trong code.

#### Cài đặt và cấu hình

Notebook cài các thư viện:

```python
!pip install xgboost scikit-image scikit-learn joblib opencv-python-headless matplotlib seaborn pandas -q
```

Sau đó khai báo:

```python
SEED = 42
DATASET_DIR = '/content/drive/MyDrive/dataset_output_N5'
SAVE_DIR = '/content/drive/MyDrive/trained_models'
IMG_SIZE = (64, 64)
```

`SEED = 42` giúp kết quả chia dữ liệu và một số thuật toán có tính tái lập. `IMG_SIZE` được lưu vào model metadata nhưng không được dùng để resize ảnh trong pipeline hiện tại.

#### Phân tích dataset

Code duyệt các thư mục con trong `DATASET_DIR`, đếm số ảnh của từng lớp và tạo `df_info`.

Output chính:

| Metric | Giá trị |
|---|---:|
| Số lớp | 80 |
| Số ảnh | 16.079 |
| Mean | 200,9875 |
| Std | 0,1118 |
| Min | 200 |
| Max | 201 |

Notebook cũng vẽ histogram phân phối số ảnh trên từng class. Biểu đồ cho thấy dataset gần như cân bằng hoàn toàn.

#### Các hàm chính

| Hàm | Input | Output | Chức năng |
|---|---|---|---|
| `imread_unicode(path, flags=cv2.IMREAD_GRAYSCALE)` | Đường dẫn ảnh, chế độ đọc ảnh | Ảnh OpenCV hoặc `None` | Đọc ảnh bằng `np.fromfile` và `cv2.imdecode`, phù hợp với đường dẫn Unicode. |
| `preprocess_image(path)` | Đường dẫn ảnh | Ảnh grayscale hoặc `None` | Gọi `imread_unicode`; không resize, không threshold. |
| `extract_hog(img)` | Ảnh grayscale | `features`, `hog_img` | Trích xuất vector HOG và ảnh HOG để trực quan hóa. |
| `make_pipeline(clf)` | Classifier scikit-learn | `Pipeline` | Tạo pipeline gồm `StandardScaler`, `PCA(0.95)` và mô hình phân loại. |

#### Trích xuất đặc trưng HOG

Hàm HOG được cấu hình như sau:

```python
hog(
    img,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm='L2-Hys',
    feature_vector=True,
    visualize=True
)
```

Kết quả sau khi load toàn bộ dataset:

| Thông tin | Giá trị |
|---|---:|
| Tổng ảnh đã load | 16.079 |
| Ảnh bỏ qua | 0 |
| Số class | 80 |
| Kích thước vector HOG | 1.512 chiều |

#### Chuẩn hóa và PCA

Trước khi phân tích PCA, code chuẩn hóa dữ liệu:

```python
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
```

Sau đó fit PCA trên tập train:

```python
pca_full = PCA(random_state=SEED).fit(X_train_sc)
```

Kết quả:

| Mức phương sai giữ lại | Số thành phần PCA cần thiết |
|---|---:|
| 95% | 197 / 1.512 |
| 99% | 404 / 1.512 |

Trong pipeline huấn luyện, PCA dùng `n_components=0.95`, nghĩa là số chiều được chọn tự động để giữ 95% phương sai.

#### Các mô hình được huấn luyện

| Mô hình | Hyperparameters chính |
|---|---|
| SVM RBF | `kernel='rbf'`, `C=10`, `gamma='scale'`, `probability=True`, `class_weight='balanced'` |
| SVM Linear | `C=1.0`, `class_weight='balanced'`, `max_iter=5000` |
| Random Forest | `n_estimators=500`, `n_jobs=-1`, `class_weight='balanced'`, `random_state=42` |
| Logistic Regression | `max_iter=3000`, `n_jobs=-1`, `class_weight='balanced'` |
| KNN | `n_neighbors=5`, `n_jobs=-1` |
| Naive Bayes | `GaussianNB()` |
| XGBoost | `n_estimators=500`, `max_depth=6`, `learning_rate=0.05`, `objective='multi:softprob'`, `eval_metric='mlogloss'`, `tree_method='hist'` |

Mỗi mô hình được huấn luyện trong cùng một pipeline:

```python
Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95, random_state=SEED)),
    ('model', clf)
])
```

#### Input và output

| Thành phần | Input | Output |
|---|---|---|
| Dataset loader | Thư mục ảnh theo class | `X`, `y` |
| HOG extractor | Ảnh grayscale | Vector đặc trưng 1.512 chiều |
| Label encoder | Nhãn dạng chuỗi | Nhãn số nguyên |
| Train/test split | `X`, `y_encoded` | `X_train`, `X_test`, `y_train`, `y_test` |
| Model pipeline | Vector HOG | Nhãn dự đoán |
| Evaluation | `y_test`, `y_pred` | Accuracy, Precision, Recall, F1, Confusion Matrix |
| Model saving | Pipeline đã fit | File `.pkl` trong `SAVE_DIR` |

#### Kết quả thực thi

Kết quả huấn luyện và đánh giá trên tập test:

| Hạng | Model | Accuracy | Macro F1 | Macro Precision | Macro Recall |
|---:|---|---:|---:|---:|---:|
| 1 | SVM (RBF) | 0,9801 | 0,9801 | 0,9808 | 0,9801 |
| 2 | Logistic Regression | 0,9571 | 0,9573 | 0,9587 | 0,9571 |
| 3 | KNN | 0,9316 | 0,9317 | 0,9347 | 0,9317 |
| 4 | Random Forest | 0,9192 | 0,9190 | 0,9217 | 0,9193 |
| 5 | SVM (Linear) | 0,9179 | 0,9178 | 0,9197 | 0,9180 |
| 6 | Naive Bayes | 0,8955 | 0,8974 | 0,9037 | 0,8957 |
| 7 | XGBoost | 0,8803 | 0,8805 | 0,8833 | 0,8805 |

Mô hình tốt nhất là `SVM (RBF)`. Model được lưu tại:

```text
/content/drive/MyDrive/trained_models/SVM_(RBF)_hog_pca.pkl
```

Notebook cũng thực hiện GridSearchCV cho SVM RBF và tìm được:

```text
Best params: {'model__C': 5, 'model__gamma': 'scale'}
Accuracy sau tinh chỉnh: 0,9801
Accuracy trước tinh chỉnh: 0,9801
```

Do accuracy không cải thiện, code giữ nguyên mô hình ban đầu.

### 3.3. Giải thích chi tiết các hàm, class và khối code quan trọng

Phần này giải thích từng thành phần quan trọng trong notebook theo đúng vai trò của nó trong pipeline Machine Learning. Notebook không định nghĩa class tự viết; các class xuất hiện trong code đều là class từ thư viện như `Pipeline`, `StandardScaler`, `PCA`, `SVC`, `RandomForestClassifier`, `LogisticRegression`, `KNeighborsClassifier`, `GaussianNB`, `XGBClassifier`, `LabelEncoder`, `GridSearchCV` và các class/utility trực quan hóa của scikit-learn hoặc matplotlib.

#### 3.3.1. Hàm `imread_unicode(path, flags=cv2.IMREAD_GRAYSCALE)`

**Tên hàm:** `imread_unicode`

**Mục đích trong project:** Đọc ảnh Kanji từ đường dẫn file, kể cả khi tên thư mục hoặc tên file có ký tự Unicode. Đây là lớp đọc dữ liệu đầu tiên của pipeline.

**Tham số đầu vào:**

| Tham số | Kiểu dữ liệu | Ý nghĩa |
|---|---|---|
| `path` | `str` | Đường dẫn ảnh cần đọc. |
| `flags` | `int` | Chế độ đọc ảnh của OpenCV, mặc định là `cv2.IMREAD_GRAYSCALE` để đọc ảnh xám. |

**Giá trị trả về:** Mảng ảnh dạng `numpy.ndarray` sau khi OpenCV giải mã thành công, hoặc `None` nếu ảnh không đọc được.

**Logic xử lý từng bước:**

1. Gọi `np.fromfile(path, dtype=np.uint8)` để đọc toàn bộ byte của file ảnh thành mảng số nguyên 8-bit.
2. Truyền mảng byte này vào `cv2.imdecode(data, flags)`.
3. OpenCV giải mã byte thành ảnh theo chế độ `flags`.
4. Trả về ảnh đã giải mã.

**Vì sao cần trong pipeline ML:** Dữ liệu đầu vào là ảnh Kanji nằm trong các thư mục có thể chứa ký tự Nhật hoặc Unicode. Cách đọc thông thường bằng `cv2.imread` đôi khi lỗi với đường dẫn Unicode, trong khi `np.fromfile` kết hợp `cv2.imdecode` xử lý ổn định hơn. Nếu bước đọc ảnh sai, toàn bộ các bước trích xuất HOG, huấn luyện và đánh giá phía sau đều không có dữ liệu đúng.

**Nơi được gọi/sử dụng:** Được gọi trong `preprocess_image(path)` và trong khối minh họa pipeline tiền xử lý để hiển thị ảnh gốc.

**Vai trò trong pipeline:** Tiền xử lý dữ liệu, cụ thể là data loading.

#### 3.3.2. Hàm `preprocess_image(path)`

**Tên hàm:** `preprocess_image`

**Mục đích trong project:** Cung cấp điểm vào thống nhất cho bước tiền xử lý ảnh trước khi trích xuất đặc trưng. Ở phiên bản hiện tại, hàm chỉ đọc ảnh grayscale và kiểm tra lỗi đọc ảnh.

**Tham số đầu vào:**

| Tham số | Kiểu dữ liệu | Ý nghĩa |
|---|---|---|
| `path` | `str` | Đường dẫn ảnh Kanji cần xử lý. |

**Giá trị trả về:** Ảnh grayscale dạng `numpy.ndarray` nếu đọc thành công; `None` nếu ảnh lỗi hoặc không giải mã được.

**Logic xử lý từng bước:**

1. Gọi `imread_unicode(path)` để đọc ảnh.
2. Kiểm tra `img is None`.
3. Nếu ảnh lỗi, trả về `None` để vòng lặp load dataset bỏ qua ảnh đó.
4. Nếu ảnh hợp lệ, trả về ảnh gốc ở dạng grayscale.

**Vì sao cần trong pipeline ML:** Hàm này tách riêng phần chuẩn bị ảnh khỏi phần duyệt dataset. Dù hiện tại chưa resize, threshold, denoise hoặc augmentation, nó tạo một điểm mở rộng rõ ràng nếu sau này cần chuẩn hóa kích thước, tăng tương phản, làm sạch nhiễu hoặc nhị phân hóa ảnh.

**Nơi được gọi/sử dụng:** Được gọi trong vòng lặp load toàn bộ dataset và trong khối trực quan hóa pipeline ảnh gốc sang HOG.

**Vai trò trong pipeline:** Tiền xử lý và kiểm soát chất lượng dữ liệu đầu vào.

#### 3.3.3. Hàm `extract_hog(img)`

**Tên hàm:** `extract_hog`

**Mục đích trong project:** Chuyển ảnh Kanji grayscale thành vector đặc trưng HOG để các mô hình học máy cổ điển có thể học được hình dạng nét chữ.

**Tham số đầu vào:**

| Tham số | Kiểu dữ liệu | Ý nghĩa |
|---|---|---|
| `img` | `numpy.ndarray` | Ảnh grayscale đã đọc từ file. |

**Giá trị trả về:**

| Output | Kiểu dữ liệu | Ý nghĩa |
|---|---|---|
| `features` | `numpy.ndarray` | Vector đặc trưng HOG dùng làm `X` cho mô hình ML. |
| `hog_img` | `numpy.ndarray` | Ảnh trực quan hóa HOG dùng để hiển thị, không dùng trực tiếp để train. |

**Logic xử lý từng bước:**

1. Gọi `hog(img, ...)` từ `skimage.feature`.
2. Chia ảnh thành các cell kích thước `8x8` pixel.
3. Tính histogram hướng gradient với `orientations=9`.
4. Gom các cell thành block `2x2`.
5. Chuẩn hóa block bằng `L2-Hys` để giảm ảnh hưởng của độ sáng và tương phản.
6. Ghép toàn bộ histogram thành một vector duy nhất vì `feature_vector=True`.
7. Tạo thêm ảnh HOG trực quan vì `visualize=True`.
8. Trả về vector đặc trưng và ảnh minh họa HOG.

**Vì sao cần trong pipeline ML:** Các mô hình như SVM, Logistic Regression, KNN, Random Forest hoặc Naive Bayes không tự học trực tiếp đặc trưng không gian phức tạp từ pixel như CNN. HOG biến hình dạng nét Kanji thành vector số mô tả hướng cạnh và gradient, giúp mô hình học được cấu trúc chữ.

**Nơi được gọi/sử dụng:** Được gọi trong vòng lặp load dataset để tạo `X`, và trong khối hiển thị pipeline để tạo ảnh HOG minh họa.

**Vai trò trong pipeline:** Feature extraction, là cầu nối giữa ảnh thô và mô hình phân loại.

#### 3.3.4. Hàm `make_pipeline(clf)`

**Tên hàm:** `make_pipeline`

**Mục đích trong project:** Tạo pipeline scikit-learn thống nhất cho mọi mô hình, gồm chuẩn hóa đặc trưng, giảm chiều PCA và classifier.

**Tham số đầu vào:**

| Tham số | Kiểu dữ liệu | Ý nghĩa |
|---|---|---|
| `clf` | Estimator scikit-learn hoặc XGBoost | Mô hình phân loại cần đưa vào cuối pipeline. |

**Giá trị trả về:** Một object `Pipeline` gồm ba bước: `scaler`, `pca`, `model`.

**Logic xử lý từng bước:**

1. Tạo bước `StandardScaler()` để chuẩn hóa các chiều HOG về phân phối có trung bình gần 0 và độ lệch chuẩn gần 1.
2. Tạo bước `PCA(n_components=0.95, random_state=SEED)` để giữ lại số thành phần chính đủ giải thích 95% phương sai.
3. Đưa classifier `clf` vào bước cuối với tên `model`.
4. Trả về `Pipeline`.

**Vì sao cần trong pipeline ML:** Hàm này đảm bảo tất cả mô hình được so sánh trên cùng quy trình tiền xử lý đặc trưng. Việc đóng gói scaler và PCA trong `Pipeline` cũng tránh rò rỉ dữ liệu vì các bước này chỉ `fit` trên tập train trong từng lần huấn luyện hoặc cross-validation.

**Nơi được gọi/sử dụng:** Được gọi trong vòng lặp train tất cả mô hình, trong `GridSearchCV` khi tinh chỉnh mô hình tốt nhất, và trong cross-validation cho top 3 mô hình.

**Vai trò trong pipeline:** Tổ chức training pipeline, chuẩn hóa feature engineering và hỗ trợ đánh giá công bằng.

#### 3.3.5. Khối import thư viện và cấu hình môi trường

**Mục đích:** Cài đặt và import toàn bộ thư viện cần cho đọc ảnh, xử lý số, trực quan hóa, trích xuất HOG, chia dữ liệu, huấn luyện, đánh giá và lưu model.

**Input:** Không có input từ người dùng trong hàm; notebook chạy lệnh `pip install` và các câu lệnh `import`.

**Output:** Các module, class và hàm được nạp vào runtime Colab.

**Logic xử lý:** Cài thư viện cần thiết, import OpenCV, NumPy, pandas, matplotlib, seaborn, HOG, PCA, scaler, các mô hình ML, metric đánh giá và `joblib`.

**Vì sao cần trong pipeline ML:** Đây là lớp phụ trợ để toàn bộ pipeline có đủ công cụ: OpenCV đọc ảnh, HOG tạo đặc trưng, scikit-learn train/evaluate, XGBoost train boosting model, joblib lưu pipeline.

**Nơi sử dụng:** Toàn bộ notebook.

**Vai trò:** Hỗ trợ project, không trực tiếp train model nhưng là điều kiện để pipeline chạy.

#### 3.3.6. Khối mount Google Drive và khai báo đường dẫn

**Mục đích:** Kết nối Google Drive, khai báo vị trí dataset và thư mục lưu model.

**Input:** Tài khoản Google Drive trong Colab, `DATASET_DIR`, `SAVE_DIR`, `IMG_SIZE`.

**Output:** Drive được mount; thư mục `SAVE_DIR` được tạo nếu chưa có.

**Logic xử lý:** Gọi `drive.mount('/content/drive')`, khai báo đường dẫn dataset, khai báo nơi lưu model, tạo thư mục lưu bằng `os.makedirs(SAVE_DIR, exist_ok=True)`.

**Vì sao cần trong pipeline ML:** Pipeline cần đọc ảnh train/test từ Drive và lưu các pipeline đã train để tái sử dụng trong prediction hoặc triển khai.

**Nơi sử dụng:** `DATASET_DIR` được dùng trong thống kê dataset và load ảnh; `SAVE_DIR` được dùng khi `joblib.dump`; `IMG_SIZE` được lưu vào metadata của model.

**Vai trò:** Hỗ trợ data loading, model persistence và triển khai sau huấn luyện.

#### 3.3.7. Khối thống kê dataset `df_info`

**Mục đích:** Đếm số class Kanji và số ảnh trong từng class để kiểm tra dữ liệu trước khi train.

**Input:** Cấu trúc thư mục trong `DATASET_DIR`, mỗi thư mục con là một class.

**Output:** DataFrame `df_info` gồm tên class và số ảnh; các thống kê tổng số class, tổng số ảnh, mean, min, max.

**Logic xử lý từng bước:**

1. Duyệt các entry trong `DATASET_DIR`.
2. Chỉ giữ entry là thư mục.
3. Đếm số file ảnh trong từng thư mục class.
4. Lưu từng dòng vào `class_info`.
5. Chuyển `class_info` thành `pd.DataFrame`.
6. In tổng số class, tổng ảnh và thống kê mô tả.

**Vì sao cần trong pipeline ML:** Giúp phát hiện dataset thiếu lớp, mất cân bằng lớp hoặc thư mục rỗng trước khi huấn luyện. Đây là bước kiểm tra chất lượng dữ liệu.

**Nơi sử dụng:** `df_info` được dùng để hiển thị thống kê, vẽ histogram phân bố ảnh/class và chọn `demo_classes` cho minh họa pipeline.

**Vai trò:** Data understanding và kiểm tra preprocessing đầu vào.

#### 3.3.8. Khối histogram phân bố số ảnh mỗi class

**Mục đích:** Trực quan hóa độ cân bằng của dataset.

**Input:** Cột `count` trong `df_info`.

**Output:** Biểu đồ histogram và đường trung bình số ảnh/class.

**Logic xử lý:** Tạo `fig, ax`, vẽ histogram số ảnh từng class, vẽ đường mean bằng `ax.axvline`, đặt tiêu đề, nhãn trục, legend và grid.

**Vì sao cần trong pipeline ML:** Nếu dataset lệch lớp mạnh, accuracy có thể gây hiểu nhầm và mô hình có xu hướng ưu tiên class nhiều mẫu. Biểu đồ này cho thấy dataset gần cân bằng, giúp củng cố lựa chọn metric và chiến lược train.

**Nơi sử dụng:** Chỉ dùng trong phần phân tích dữ liệu ban đầu.

**Vai trò:** Evaluation hỗ trợ dữ liệu, không train/predict trực tiếp.

#### 3.3.9. Khối hiển thị pipeline ảnh gốc sang HOG

**Mục đích:** Minh họa trực quan quá trình biến ảnh Kanji grayscale thành biểu diễn HOG.

**Input:** `df_info`, `DATASET_DIR`, ba class được chọn ngẫu nhiên, ảnh mẫu trong từng class.

**Output:** Figure gồm ảnh gốc và ảnh HOG visualization cho từng class mẫu.

**Logic xử lý từng bước:**

1. Chọn ngẫu nhiên ba class từ `df_info['class']`.
2. Tạo lưới subplot 3 hàng, 2 cột.
3. Với mỗi class, lấy một ảnh mẫu trong thư mục class.
4. Đọc ảnh gốc bằng `imread_unicode`.
5. Đọc/tiền xử lý bằng `preprocess_image`.
6. Trích xuất HOG bằng `extract_hog`.
7. Điều chỉnh cường độ ảnh HOG bằng `exposure.rescale_intensity`.
8. Hiển thị ảnh gốc và ảnh HOG cạnh nhau.

**Vì sao cần trong pipeline ML:** Giúp kiểm chứng bằng mắt rằng HOG đang bắt được hướng nét chữ thay vì tạo đặc trưng rỗng hoặc nhiễu. Đây là bước debug quan trọng trước khi train hàng loạt.

**Nơi sử dụng:** Chỉ trong phần trực quan hóa notebook.

**Vai trò:** Hỗ trợ preprocessing và feature extraction.

#### 3.3.10. Khối load dataset và tạo `X`, `y`

**Mục đích:** Chuyển toàn bộ ảnh trong dataset thành ma trận đặc trưng `X` và vector nhãn `y`.

**Input:** `DATASET_DIR`, các file ảnh trong từng thư mục class, hàm `preprocess_image`, hàm `extract_hog`.

**Output:** `X` là `numpy.ndarray` chứa vector HOG của tất cả ảnh; `y` là `numpy.ndarray` chứa nhãn class dạng chuỗi; `skipped` là số ảnh lỗi bị bỏ qua.

**Logic xử lý từng bước:**

1. Khởi tạo `X = []`, `y = []`, `skipped = 0`.
2. Duyệt từng thư mục class trong `DATASET_DIR`.
3. Bỏ qua entry không phải thư mục.
4. Lấy tên thư mục làm nhãn `label`.
5. Duyệt từng file ảnh trong thư mục class.
6. Gọi `preprocess_image(img_entry.path)`.
7. Nếu ảnh lỗi, tăng `skipped` và bỏ qua.
8. Nếu ảnh hợp lệ, gọi `extract_hog(img)`.
9. Thêm vector HOG vào `X` và nhãn vào `y`.
10. Chuyển `X`, `y` từ list sang `numpy.ndarray`.

**Vì sao cần trong pipeline ML:** Đây là bước tạo dataset học máy thực tế. Model không train trực tiếp trên file ảnh; nó train trên ma trận số `X` và nhãn `y`.

**Nơi sử dụng:** `X` và `y` được dùng trong mã hóa nhãn, chia train/test, PCA analysis, training, cross-validation và báo cáo tổng kết.

**Vai trò:** Feature extraction và chuẩn bị dữ liệu cho training/evaluation.

#### 3.3.11. Class `LabelEncoder`

**Tên class:** `sklearn.preprocessing.LabelEncoder`

**Mục đích trong project:** Chuyển nhãn Kanji dạng chuỗi/ký tự thành số nguyên để các mô hình scikit-learn xử lý.

**Input:** Vector nhãn `y` dạng chuỗi.

**Output:** `y_encoded`, trong đó mỗi class được ánh xạ thành một ID số; `label_encoder.classes_` lưu danh sách class gốc.

**Logic xử lý:** Khởi tạo `LabelEncoder`, gọi `fit_transform(y)`, sinh mapping class sang số nguyên theo thứ tự class.

**Vì sao cần trong pipeline ML:** Nhiều thuật toán phân loại cần nhãn dạng số. Đồng thời `label_encoder` giúp chuyển ID dự đoán ngược lại thành ký tự Kanji khi hiển thị hoặc triển khai.

**Nơi sử dụng:** Dùng trước `train_test_split`, dùng trong `classification_report`, tạo mapping `Class ID` sang Kanji, và được lưu cùng model bằng `joblib.dump`.

**Vai trò:** Preprocessing nhãn và hỗ trợ prediction.

#### 3.3.12. Khối chia train/test bằng `train_test_split`

**Mục đích:** Chia dataset thành tập huấn luyện và tập kiểm tra.

**Input:** `X`, `y_encoded`, `test_size=0.2`, `random_state=SEED`, `stratify=y_encoded`.

**Output:** `X_train`, `X_test`, `y_train`, `y_test`.

**Logic xử lý:** Chia 80% dữ liệu cho train và 20% cho test; dùng `stratify` để giữ phân phối class trong train/test gần giống dataset gốc.

**Vì sao cần trong pipeline ML:** Tập test độc lập cho phép đánh giá khả năng tổng quát hóa của mô hình trên dữ liệu chưa thấy.

**Nơi sử dụng:** Các biến train/test được dùng trong PCA analysis, fit model, predict, tính metric, confusion matrix và per-class F1.

**Vai trò:** Training/evaluation split.

#### 3.3.13. Class `StandardScaler`

**Tên class:** `sklearn.preprocessing.StandardScaler`

**Mục đích trong project:** Chuẩn hóa các chiều đặc trưng HOG trước khi PCA và classifier.

**Input:** Ma trận đặc trưng `X_train` hoặc dữ liệu trong pipeline.

**Output:** Ma trận đặc trưng đã chuẩn hóa.

**Logic xử lý:** Khi `fit`, scaler học mean và standard deviation từ tập train. Khi `transform`, mỗi đặc trưng được biến đổi theo công thức `(x - mean) / std`.

**Vì sao cần trong pipeline ML:** HOG có nhiều chiều; nếu scale khác nhau, PCA và các mô hình nhạy khoảng cách/biên như SVM, Logistic Regression, KNN có thể bị ảnh hưởng. Chuẩn hóa giúp các chiều đóng góp công bằng hơn.

**Nơi sử dụng:** Dùng thủ công trong khối PCA analysis (`scaler.fit_transform`, `scaler.transform`) và dùng trong `make_pipeline`.

**Vai trò:** Preprocessing đặc trưng.

#### 3.3.14. Class `PCA`

**Tên class:** `sklearn.decomposition.PCA`

**Mục đích trong project:** Giảm số chiều vector HOG nhưng vẫn giữ phần lớn thông tin.

**Input:** Đặc trưng đã chuẩn hóa.

**Output:** Đặc trưng sau giảm chiều; trong phân tích còn có `explained_variance_ratio_`, `cum_var`, `n95`, `n99`.

**Logic xử lý:** PCA tìm các trục thành phần chính có phương sai lớn nhất, sắp xếp chúng theo lượng thông tin giữ được, rồi chiếu dữ liệu sang không gian ít chiều hơn. Trong pipeline, `n_components=0.95` nghĩa là giữ đủ số thành phần để giải thích 95% phương sai.

**Vì sao cần trong pipeline ML:** Vector HOG có thể nhiều chiều. PCA giúp giảm nhiễu, giảm thời gian train/inference và có thể cải thiện khả năng tổng quát hóa.

**Nơi sử dụng:** Dùng trong khối phân tích phương sai PCA và trong `make_pipeline`.

**Vai trò:** Feature preprocessing và dimensionality reduction.

#### 3.3.15. Dictionary `models`

**Mục đích:** Khai báo danh sách mô hình cần train và so sánh.

**Input:** Không nhận input trực tiếp; mỗi key là tên mô hình, mỗi value là classifier đã cấu hình.

**Output:** Dictionary gồm 7 mô hình: `SVM (RBF)`, `SVM (Linear)`, `Random Forest`, `Logistic Regression`, `KNN`, `Naive Bayes`, `XGBoost`.

**Logic xử lý:** Tạo từng classifier với hyperparameter ban đầu, ví dụ `SVC(kernel='rbf', C=10)`, `RandomForestClassifier(n_estimators=500)`, `XGBClassifier(n_estimators=500, max_depth=6)`.

**Vì sao cần trong pipeline ML:** Cho phép benchmark nhiều thuật toán trên cùng đặc trưng HOG + PCA để chọn mô hình tốt nhất dựa trên metric thực nghiệm thay vì giả định.

**Nơi sử dụng:** Vòng lặp train tất cả mô hình, `GridSearchCV` chọn base model, và cross-validation top 3.

**Vai trò:** Training và model selection.

#### 3.3.16. Các class mô hình phân loại

| Class | Mục đích | Input khi train | Output khi predict | Vai trò trong pipeline |
|---|---|---|---|---|
| `SVC` | SVM kernel RBF học ranh giới phi tuyến giữa các class Kanji. | Feature HOG sau scaler/PCA và `y_train`. | ID class dự đoán; có thể xuất xác suất vì `probability=True`. | Training/prediction, là mô hình tốt nhất trong kết quả notebook. |
| `LinearSVC` | SVM tuyến tính, nhanh hơn RBF nhưng ít linh hoạt hơn. | Feature HOG sau scaler/PCA và `y_train`. | ID class dự đoán. | Baseline mạnh cho dữ liệu tuyến tính hoặc gần tuyến tính. |
| `RandomForestClassifier` | Tập hợp nhiều cây quyết định để phân loại. | Feature HOG/PCA và nhãn. | ID class dự đoán. | Benchmark mô hình ensemble không tuyến tính. |
| `LogisticRegression` | Mô hình tuyến tính đa lớp dựa trên xác suất. | Feature HOG/PCA và nhãn. | ID class dự đoán. | Baseline dễ diễn giải, thường ổn với đặc trưng đã chuẩn hóa. |
| `KNeighborsClassifier` | Dự đoán theo các mẫu gần nhất trong không gian feature. | Feature train đã lưu trong model. | ID class theo đa số láng giềng. | Baseline dựa trên khoảng cách, nhạy với scale nên cần scaler. |
| `GaussianNB` | Naive Bayes giả định feature theo phân phối Gaussian. | Feature HOG/PCA và nhãn. | ID class dự đoán. | Baseline nhanh, đơn giản. |
| `XGBClassifier` | Gradient boosting dùng nhiều cây yếu để tạo mô hình mạnh. | Feature HOG/PCA và nhãn. | ID class/xác suất class. | Benchmark boosting hiện đại cho dữ liệu đặc trưng số. |

**Vì sao cần nhiều class mô hình:** Không có mô hình nào luôn tốt nhất cho mọi dataset. Việc so sánh nhiều class giúp chọn thuật toán phù hợp với đặc trưng HOG của ảnh Kanji.

#### 3.3.17. Class `Pipeline`

**Tên class:** `sklearn.pipeline.Pipeline`

**Mục đích trong project:** Đóng gói các bước `StandardScaler`, `PCA` và classifier thành một đối tượng thống nhất.

**Input:** Danh sách các bước dạng `(tên_bước, estimator)`.

**Output:** Pipeline có thể gọi `.fit`, `.predict`, dùng trong `GridSearchCV`, `cross_val_score` và lưu bằng `joblib`.

**Logic xử lý:** Khi `.fit`, pipeline fit scaler, transform dữ liệu, fit PCA, transform dữ liệu, rồi fit model. Khi `.predict`, pipeline áp dụng đúng scaler và PCA đã học trước khi gọi model dự đoán.

**Vì sao cần trong pipeline ML:** Giúp quy trình train và inference nhất quán, tránh quên bước chuẩn hóa/giảm chiều khi dự đoán ảnh mới.

**Nơi sử dụng:** Được tạo trong `make_pipeline`; sau đó dùng trong training, tuning, cross-validation và lưu model.

**Vai trò:** Training, evaluation và prediction.

#### 3.3.18. Khối train, predict, đánh giá và lưu từng model

**Mục đích:** Huấn luyện tất cả mô hình, dự đoán trên tập test, tính metric, lưu pipeline và tổng hợp kết quả.

**Input:** `models`, `X_train`, `y_train`, `X_test`, `y_test`, `label_encoder`, `SAVE_DIR`, `IMG_SIZE`.

**Output:** `df_results`, `pipelines`, `predictions`, các file `.pkl` đã lưu trong Drive.

**Logic xử lý từng bước:**

1. Khởi tạo `results`, `pipelines`, `predictions`.
2. Duyệt từng cặp `name, clf` trong `models`.
3. Tạo pipeline bằng `make_pipeline(clf)`.
4. Gọi `pipe.fit(X_train, y_train)` để huấn luyện scaler, PCA và model.
5. Gọi `pipe.predict(X_test)` để dự đoán tập test.
6. Tính `accuracy_score`.
7. Gọi `classification_report(..., output_dict=True)` để lấy precision, recall, F1 theo class và macro average.
8. Lưu pipeline, label encoder, image size, tên model và accuracy bằng `joblib.dump`.
9. Lưu pipeline đã train vào `pipelines[name]`.
10. Lưu dự đoán vào `predictions[name]`.
11. Thêm metric và đường dẫn model vào `results`.
12. Tạo `df_results` và sắp xếp giảm dần theo Accuracy.

**Vì sao cần trong pipeline ML:** Đây là trung tâm của quá trình training/evaluation. Nó tạo mô hình thực tế, đo chất lượng, lưu lại artifact và chọn ứng viên tốt nhất.

**Nơi sử dụng:** `df_results` được dùng ở hầu hết các khối phía sau: tuning, bảng kết quả, biểu đồ metric, confusion matrix, per-class F1, cross-validation và tổng kết.

**Vai trò:** Training, evaluation, model selection và model persistence.

#### 3.3.19. Hàm/class đánh giá `accuracy_score`, `classification_report`, `confusion_matrix`

**Mục đích:** Đo hiệu năng mô hình ở nhiều góc nhìn.

**Input:** `y_test` và `y_pred`.

**Output:** Accuracy, precision, recall, F1-score, confusion matrix.

**Logic xử lý:** `accuracy_score` tính tỷ lệ dự đoán đúng; `classification_report` tính precision/recall/F1 cho từng class và trung bình macro; `confusion_matrix` đếm số mẫu đúng/sai giữa từng cặp class thật và class dự đoán.

**Vì sao cần trong pipeline ML:** Accuracy cho cái nhìn tổng quát, macro metric quan trọng khi có nhiều class, confusion matrix chỉ ra class nào hay bị nhầm.

**Nơi sử dụng:** Trong vòng lặp train, confusion matrix best/worst model, per-class F1 và tổng kết.

**Vai trò:** Evaluation.

#### 3.3.20. Class `GridSearchCV`

**Tên class:** `sklearn.model_selection.GridSearchCV`

**Mục đích trong project:** Tinh chỉnh hyperparameter cho mô hình tốt nhất ban đầu.

**Input:** `tune_pipe`, `param_grid`, `cv=3`, `scoring='accuracy'`, `X_train`, `y_train`.

**Output:** Object `gs` chứa `best_params_`, `best_estimator_` và mô hình đã fit tốt nhất theo cross-validation nội bộ.

**Logic xử lý từng bước:**

1. Lấy `best_name` từ dòng đầu của `df_results`.
2. Chọn `param_grid` phù hợp nếu model tốt nhất là SVM RBF, Logistic Regression hoặc XGBoost.
3. Tạo pipeline bằng `make_pipeline(base_clf)`.
4. Chạy `GridSearchCV` với 3-fold CV trên tập train.
5. Fit `gs` bằng `X_train`, `y_train`.
6. Dự đoán `X_test` bằng `gs.predict`.
7. So sánh accuracy sau tinh chỉnh với accuracy ban đầu.
8. Nếu tốt hơn, lưu model tuned vào `pipelines` và `predictions`; nếu không, giữ model gốc.

**Vì sao cần trong pipeline ML:** Hyperparameter như `C`, `gamma`, số cây hoặc độ sâu cây ảnh hưởng mạnh đến chất lượng model. Grid search kiểm tra có hệ thống một tập giá trị nhỏ để tránh chọn cấu hình thủ công thiếu căn cứ.

**Nơi sử dụng:** Chạy sau khi đã có `df_results`.

**Vai trò:** Model tuning và validation.

#### 3.3.21. Khối bảng kết quả `styled`

**Mục đích:** Trình bày bảng so sánh các mô hình trên tập test.

**Input:** `df_results` và các cột `Model`, `Accuracy`, `Macro F1`, `Macro Precision`, `Macro Recall`.

**Output:** Bảng pandas styled có format số và màu nền theo metric.

**Logic xử lý:** Chọn cột cần hiển thị, format metric 4 chữ số thập phân, dùng `background_gradient` để tô màu theo Accuracy và Macro F1, thêm caption.

**Vì sao cần trong pipeline ML:** Giúp đọc nhanh mô hình nào tốt hơn và hỗ trợ quyết định chọn model.

**Nơi sử dụng:** Sau khi train xong tất cả mô hình và ở phần tổng kết.

**Vai trò:** Evaluation reporting.

#### 3.3.22. Khối biểu đồ so sánh metric và radar chart

**Mục đích:** Trực quan hóa so sánh giữa các mô hình theo nhiều metric.

**Input:** `df_results`, danh sách `metrics`.

**Output:** Biểu đồ thanh ngang theo từng metric và radar chart.

**Logic xử lý:** Tạo `df_plot`, vẽ từng metric bằng `barh`, thêm giá trị lên thanh; radar chart chuyển metric thành các trục góc, vẽ đường và vùng cho từng model.

**Vì sao cần trong pipeline ML:** Một mô hình có thể accuracy cao nhưng macro recall thấp. Biểu đồ giúp nhìn cân bằng tổng thể giữa các metric.

**Nơi sử dụng:** Sau bảng kết quả.

**Vai trò:** Evaluation visualization.

#### 3.3.23. Class `ConfusionMatrixDisplay` và khối confusion matrix best/worst

**Mục đích:** Phân tích lỗi dự đoán của mô hình tốt nhất và tệ nhất.

**Input:** `y_test`, `predictions[name]`, `df_results`, `label_encoder`.

**Output:** Confusion matrix chuẩn hóa cho top class bị nhầm nhiều nhất và bảng mapping `Class ID` sang Kanji.

**Logic xử lý từng bước:**

1. Chọn model tốt nhất và tệ nhất theo Accuracy.
2. Tính `confusion_matrix`.
3. Tính số lỗi mỗi class bằng tổng hàng trừ đường chéo.
4. Chọn top `TOP_N = 10` class lỗi nhiều nhất.
5. Cắt ma trận confusion cho các class này.
6. Chuẩn hóa theo hàng để mỗi hàng thể hiện tỷ lệ nhầm.
7. Hiển thị bằng `ConfusionMatrixDisplay`.
8. Tạo `mapping_df` để người đọc biết `C0`, `C1`, ... tương ứng Kanji nào.

**Vì sao cần trong pipeline ML:** Confusion matrix cho biết mô hình hay nhầm cặp Kanji nào. Đây là thông tin quan trọng để cải thiện dữ liệu, augmentation hoặc thiết kế feature tốt hơn.

**Nơi sử dụng:** Sau khi đã có `predictions` của từng model.

**Vai trò:** Error analysis và evaluation.

#### 3.3.24. Khối per-class F1

**Mục đích:** Đánh giá chi tiết chất lượng mô hình tốt nhất trên từng class Kanji.

**Input:** `best_name`, `y_test`, `predictions[best_name]`, `label_encoder.classes_`.

**Output:** DataFrame `per_class`, biểu đồ top 10 F1 cao nhất/thấp nhất, histogram phân phối F1 và danh sách class có F1 < 0.6.

**Logic xử lý từng bước:**

1. Tạo `classification_report` dạng dictionary.
2. Chuyển metric từng class thành DataFrame.
3. Sắp xếp theo `f1-score`.
4. Lấy top 10 class tốt nhất và top 10 class thấp nhất.
5. Vẽ bar chart F1 cho các class tiêu biểu.
6. Vẽ histogram phân phối F1 toàn bộ class.
7. Lọc các class có F1 < 0.6 để khuyến nghị cải thiện.

**Vì sao cần trong pipeline ML:** Với bài toán 80 class, metric trung bình có thể che giấu các class khó. Per-class F1 giúp xác định chính xác class nào cần thêm dữ liệu hoặc xử lý riêng.

**Nơi sử dụng:** Sau confusion matrix và trước cross-validation.

**Vai trò:** Evaluation chi tiết và định hướng cải thiện dataset/model.

#### 3.3.25. Class `StratifiedKFold` và hàm `cross_val_score`

**Mục đích:** Đánh giá độ ổn định của top 3 mô hình bằng 5-fold cross-validation.

**Input:** `X`, `y_encoded`, top 3 model theo Accuracy, `cv=StratifiedKFold(...)`.

**Output:** `cv_results`, mean accuracy và standard deviation của từng model, boxplot CV Accuracy.

**Logic xử lý từng bước:**

1. Tạo `StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)`.
2. Lấy top 3 model từ `df_results`.
3. Với từng model, tạo pipeline mới bằng `make_pipeline(models[name])`.
4. Chạy `cross_val_score` trên toàn bộ `X`, `y_encoded`.
5. Lưu mảng accuracy của 5 fold vào `cv_results`.
6. In mean và standard deviation.
7. Vẽ boxplot để so sánh phân phối accuracy.

**Vì sao cần trong pipeline ML:** Một lần chia train/test có thể may rủi. Cross-validation kiểm tra mô hình có ổn định trên nhiều cách chia dữ liệu hay không.

**Nơi sử dụng:** Sau khi đã xếp hạng mô hình.

**Vai trò:** Evaluation và model validation.

#### 3.3.26. Khối prediction sample

**Mục đích:** Lấy ngẫu nhiên một số mẫu test để kiểm tra dự đoán đúng/sai của model tốt nhất.

**Input:** `X_test`, `y_test`, `predictions[best_name]`, `pipelines[best_name]`.

**Output:** `sample_table` gồm sample index, true class, predicted class và trạng thái đúng/sai.

**Logic xử lý:** Chọn tối đa 10 index ngẫu nhiên từ tập test, lấy nhãn thật và nhãn dự đoán, tạo DataFrame hiển thị kết quả.

**Vì sao cần trong pipeline ML:** Đây là kiểm tra định tính đơn giản giúp xem model hoạt động trên từng mẫu cụ thể, bổ sung cho metric tổng hợp.

**Nơi sử dụng:** Sau cross-validation.

**Vai trò:** Prediction và kiểm tra kết quả dự đoán.

#### 3.3.27. Khối lưu model bằng `joblib.dump`

**Mục đích:** Lưu pipeline đã train và thông tin phụ trợ ra file `.pkl`.

**Input:** Dictionary gồm `model`, `label_encoder`, `img_size`, `model_name`, `accuracy`, cùng `save_path`.

**Output:** File model trong `SAVE_DIR`, ví dụ `/content/drive/MyDrive/trained_models/SVM_(RBF)_hog_pca.pkl`.

**Logic xử lý:** Tạo đường dẫn file theo tên model, gọi `joblib.dump(...)` để serialize pipeline và metadata.

**Vì sao cần trong pipeline ML:** Sau khi train, model cần được lưu để inference lại mà không phải huấn luyện từ đầu. Việc lưu cả `label_encoder` giúp giải mã class ID thành Kanji khi dự đoán.

**Nơi sử dụng:** Bên trong vòng lặp train từng model.

**Vai trò:** Model persistence và hỗ trợ prediction/triển khai.

#### 3.3.28. Khối tổng kết và khuyến nghị

**Mục đích:** Tóm tắt mô hình tốt nhất, mô hình thứ hai, mô hình yếu nhất và đưa ra hướng cải thiện.

**Input:** `df_results`, `display_cols`, `X.shape[1]`, `n95`.

**Output:** Nội dung in ra màn hình và bảng kết quả cuối.

**Logic xử lý:** Lấy dòng đầu, dòng thứ hai và dòng cuối của `df_results`; in accuracy và Macro F1; nêu nhận xét về HOG + PCA, class weight, augmentation, CNN và khả năng tích hợp SRS.

**Vì sao cần trong pipeline ML:** Giúp biến kết quả thực nghiệm thành quyết định: nên dùng model nào, khi nào cân nhắc model khác, và cải thiện tiếp theo ở đâu.

**Nơi sử dụng:** Phần cuối notebook.

**Vai trò:** Reporting, model selection và định hướng phát triển.

## 4. Machine Learning Techniques

### 4.1. Data preprocessing

Pipeline tiền xử lý gồm đọc ảnh grayscale và kiểm tra ảnh lỗi. Code không thực hiện resize, crop, thresholding, denoising hoặc cân bằng histogram trước khi trích xuất đặc trưng.

Ưu điểm của cách tiền xử lý tối giản là giữ nguyên dữ liệu gốc, giảm rủi ro làm mất nét chữ. Tuy nhiên, nếu ảnh đầu vào có kích thước hoặc độ tương phản không đồng nhất, việc không chuẩn hóa ảnh có thể làm mô hình nhạy với khác biệt về cách chụp hoặc cách render ảnh.

### 4.2. Feature extraction: HOG

HOG, viết tắt của Histogram of Oriented Gradients, là kỹ thuật mô tả hình dạng dựa trên hướng gradient cục bộ. Với ảnh chữ Kanji, HOG phù hợp vì ký tự được nhận biết nhiều qua nét, cạnh, hướng stroke và cấu trúc hình học.

Các tham số HOG trong code:

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `orientations` | 9 | Số hướng gradient được lượng tử hóa. |
| `pixels_per_cell` | `(8, 8)` | Kích thước mỗi cell tính histogram. |
| `cells_per_block` | `(2, 2)` | Số cell trong mỗi block chuẩn hóa. |
| `block_norm` | `L2-Hys` | Cách chuẩn hóa block, giúp ổn định trước thay đổi độ sáng. |
| `feature_vector` | `True` | Trả về vector 1 chiều. |
| `visualize` | `True` | Trả thêm ảnh HOG để vẽ minh họa. |

Vector HOG đầu ra có 1.512 chiều.

### 4.3. Data augmentation

Không có bước data augmentation trong code. Các phép như xoay, dịch chuyển, co giãn, dilation, erosion hoặc thêm nhiễu không được áp dụng.

Trong phần khuyến nghị cuối notebook, tác giả đề xuất có thể tăng cường dữ liệu bằng rotation và dilation cho các lớp có F1 thấp, nhưng đây chỉ là đề xuất, chưa được triển khai trong mã nguồn.

### 4.4. Normalization/standardization

`StandardScaler` được dùng để chuẩn hóa đặc trưng:

```python
('scaler', StandardScaler())
```

Chuẩn hóa giúp mỗi chiều đặc trưng có trung bình gần 0 và độ lệch chuẩn gần 1. Điều này quan trọng với SVM, Logistic Regression, KNN và PCA vì các phương pháp này nhạy với thang đo của đặc trưng.

### 4.5. PCA giảm chiều

PCA được dùng sau chuẩn hóa:

```python
('pca', PCA(n_components=0.95, random_state=SEED))
```

Mục tiêu là giảm số chiều từ 1.512 xuống số thành phần đủ giữ 95% phương sai. Phân tích trong notebook cho thấy cần 197 thành phần để giữ 95% phương sai.

Lợi ích:

- Giảm nhiễu và thông tin dư thừa.
- Tăng tốc huấn luyện.
- Giảm nguy cơ overfitting với không gian đặc trưng quá lớn.

### 4.6. Model architecture

Code không sử dụng mô hình Deep Learning nên không có kiến trúc neural network, layer convolution, activation hay pooling. Các mô hình là học máy cổ điển.

Kiến trúc tổng quát:

```text
Ảnh grayscale -> HOG -> StandardScaler -> PCA -> Classifier -> Nhãn Kanji
```

Các classifier được thử nghiệm gồm SVM, Random Forest, Logistic Regression, KNN, Naive Bayes và XGBoost.

### 4.7. Loss functions

Loss function không được định nghĩa thủ công. Mỗi thư viện sử dụng loss nội bộ:

| Mô hình | Loss/tiêu chí tối ưu |
|---|---|
| SVM RBF / LinearSVC | Tối ưu margin; LinearSVC thường dùng squared hinge loss mặc định. |
| Logistic Regression | Log loss/cross-entropy đa lớp theo mặc định của scikit-learn. |
| Random Forest | Tiêu chí chia node mặc định của scikit-learn, thường là Gini impurity. |
| KNN | Không huấn luyện bằng loss; dự đoán dựa trên láng giềng gần nhất. |
| GaussianNB | Dựa trên xác suất Bayes với giả định Gaussian. |
| XGBoost | `mlogloss` được khai báo trong `eval_metric`; objective là `multi:softprob`. |

### 4.8. Optimizers

Không có optimizer kiểu Deep Learning như SGD, Adam hoặc RMSProp. Việc tối ưu được thực hiện bên trong từng thuật toán:

- SVM tối ưu bài toán margin.
- Logistic Regression dùng solver mặc định của scikit-learn.
- Random Forest xây dựng nhiều cây quyết định.
- XGBoost dùng gradient boosting trên cây.
- PCA dùng phân rã tuyến tính để tìm thành phần chính.

### 4.9. Training procedure

Quy trình huấn luyện:

1. Duyệt dataset và đọc ảnh grayscale.
2. Trích xuất HOG cho từng ảnh.
3. Mã hóa nhãn bằng `LabelEncoder`.
4. Chia train/test theo tỷ lệ 80/20 có stratify.
5. Với từng mô hình:
   - Tạo pipeline gồm scaler, PCA và classifier.
   - Fit pipeline trên tập train.
   - Dự đoán trên tập test.
   - Tính Accuracy, Macro Precision, Macro Recall, Macro F1.
   - Lưu model bằng `joblib`.
6. Sắp xếp kết quả theo Accuracy.
7. Tinh chỉnh mô hình tốt nhất bằng GridSearchCV.
8. Vẽ biểu đồ, confusion matrix, per-class F1 và cross-validation.

### 4.10. Evaluation methods và metrics

Các phương pháp đánh giá:

| Phương pháp | Mục đích |
|---|---|
| Test set 20% | Đánh giá hiệu năng cuối trên dữ liệu chưa dùng khi train. |
| Classification report | Tính precision, recall, F1 theo từng lớp và trung bình macro. |
| Confusion matrix | Quan sát các lớp bị nhầm lẫn. |
| Per-class F1 | Phân tích hiệu quả theo từng class. |
| 5-fold Stratified Cross-validation | Kiểm tra độ ổn định của top 3 mô hình. |

Metrics được sử dụng:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Confusion Matrix
- Per-class Precision/Recall/F1
- Cross-validation Accuracy mean và standard deviation

## 5. Training Process

### 5.1. Training workflow

Workflow huấn luyện trong notebook có thể tóm tắt như sau:

```text
Mount Google Drive
        |
Đọc dataset theo thư mục class
        |
Đọc ảnh grayscale
        |
Trích xuất HOG
        |
Encode nhãn
        |
Train/test split 80/20
        |
StandardScaler + PCA + Model
        |
Đánh giá trên test set
        |
Tinh chỉnh mô hình tốt nhất
        |
Cross-validation top 3
        |
Lưu mô hình
```

### 5.2. Hyperparameters

#### Hyperparameters chung

| Hyperparameter | Giá trị |
|---|---|
| `SEED` | 42 |
| `test_size` | 0,2 |
| `stratify` | Có |
| HOG orientations | 9 |
| HOG pixels per cell | `(8, 8)` |
| HOG cells per block | `(2, 2)` |
| HOG block norm | `L2-Hys` |
| PCA | Giữ 95% phương sai |
| Cross-validation | StratifiedKFold 5 folds |

#### Epochs

Không có `epochs` trong code vì notebook sử dụng các mô hình học máy cổ điển, không sử dụng neural network/deep learning.

#### Batch size

Không có `batch_size` trong code. Các mô hình được huấn luyện bằng API `.fit()` của scikit-learn/XGBoost trên toàn bộ tập train hoặc theo cơ chế nội bộ của thuật toán.

#### Learning rate

Learning rate chỉ được khai báo cho XGBoost:

| Model | Learning rate |
|---|---:|
| XGBoost | 0,05 |

Các mô hình còn lại không khai báo learning rate trong code.

#### Hyperparameters theo mô hình

| Model | Hyperparameters |
|---|---|
| SVM RBF | `C=10`, `gamma='scale'`, `kernel='rbf'`, `probability=True`, `class_weight='balanced'` |
| SVM Linear | `C=1.0`, `max_iter=5000`, `class_weight='balanced'` |
| Random Forest | `n_estimators=500`, `n_jobs=-1`, `class_weight='balanced'` |
| Logistic Regression | `max_iter=3000`, `n_jobs=-1`, `class_weight='balanced'` |
| KNN | `n_neighbors=5`, `n_jobs=-1` |
| Naive Bayes | Mặc định `GaussianNB()` |
| XGBoost | `n_estimators=500`, `max_depth=6`, `learning_rate=0.05`, `tree_method='hist'` |

### 5.3. Training results per epoch

Không có kết quả theo epoch vì code không huấn luyện mô hình deep learning. Kết quả chỉ được ghi nhận sau khi mỗi mô hình hoàn tất quá trình fit.

### 5.4. Phân tích hội tụ và hành vi mô hình

Do không có log loss theo epoch hoặc validation curve, không thể phân tích hội tụ theo từng vòng lặp một cách trực tiếp.

Tuy nhiên, có thể rút ra một số nhận xét từ kết quả:

- SVM RBF đạt Accuracy và Macro F1 khoảng 0,9801, cho thấy HOG kết hợp kernel phi tuyến phù hợp tốt với bài toán nhận dạng Kanji.
- Logistic Regression đạt 0,9571 Accuracy, chứng tỏ đặc trưng HOG sau chuẩn hóa và PCA có tính phân tách tuyến tính khá tốt.
- KNN đạt 0,9316, cho thấy khoảng cách trong không gian đặc trưng HOG-PCA có ý nghĩa phân loại.
- XGBoost có kết quả thấp nhất trong các mô hình thử nghiệm, có thể do đặc trưng PCA liên tục không phải dạng tối ưu nhất cho cây boosting trong cấu hình hiện tại.
- GridSearchCV cho SVM RBF không cải thiện accuracy, nghĩa là cấu hình ban đầu `C=10`, `gamma='scale'` đã đủ tốt hoặc hiệu năng đã gần bão hòa trên tập test.

## 6. Experimental Results

### 6.1. Bảng kết quả cuối cùng

| Hạng | Model | Accuracy | Macro F1 | Macro Precision | Macro Recall |
|---:|---|---:|---:|---:|---:|
| 1 | SVM (RBF) | 0,9801 | 0,9801 | 0,9808 | 0,9801 |
| 2 | Logistic Regression | 0,9571 | 0,9573 | 0,9587 | 0,9571 |
| 3 | KNN | 0,9316 | 0,9317 | 0,9347 | 0,9317 |
| 4 | Random Forest | 0,9192 | 0,9190 | 0,9217 | 0,9193 |
| 5 | SVM (Linear) | 0,9179 | 0,9178 | 0,9197 | 0,9180 |
| 6 | Naive Bayes | 0,8955 | 0,8974 | 0,9037 | 0,8957 |
| 7 | XGBoost | 0,8803 | 0,8805 | 0,8833 | 0,8805 |

Mô hình tốt nhất: `SVM (RBF)`.

Mô hình yếu nhất trong thử nghiệm: `XGBoost`.

### 6.2. Kết quả GridSearchCV

GridSearchCV được áp dụng cho SVM RBF với:

```python
param_grid = {
    'model__C': [1, 5, 10, 20],
    'model__gamma': ['scale', 'auto']
}
```

Kết quả:

| Nội dung | Giá trị |
|---|---|
| Best params | `C=5`, `gamma='scale'` |
| Accuracy trước tinh chỉnh | 0,9801 |
| Accuracy sau tinh chỉnh | 0,9801 |
| Kết luận | Không cải thiện đáng kể |

### 6.3. Cross-validation top 3 mô hình

Notebook chạy 5-fold Stratified Cross-validation cho ba mô hình tốt nhất theo Accuracy:

| Model | CV Accuracy mean | CV Accuracy std |
|---|---:|---:|
| SVM (RBF) | 0,9740 | 0,0027 |
| Logistic Regression | 0,9537 | 0,0040 |
| KNN | 0,9338 | 0,0024 |

SVM RBF không chỉ tốt nhất trên test set mà còn có kết quả cross-validation cao và độ lệch chuẩn nhỏ, cho thấy mô hình tương đối ổn định.

### 6.4. Charts/plots được tạo bởi code

Notebook tạo các biểu đồ sau:

| Biểu đồ | Mục đích |
|---|---|
| Histogram số ảnh/class | Kiểm tra phân phối dữ liệu giữa các lớp. |
| Ảnh gốc và HOG visualization | Minh họa pipeline từ ảnh grayscale sang đặc trưng HOG. |
| PCA cumulative explained variance | Xác định số thành phần PCA cần để giữ 95% và 99% phương sai. |
| Bar chart so sánh metric | So sánh Accuracy, Macro F1, Macro Precision, Macro Recall giữa các mô hình. |
| Radar chart | So sánh đa chiều các mô hình trên nhiều metric. |
| Confusion matrix | So sánh mô hình tốt nhất và yếu nhất trên các lớp có nhiều lỗi. |
| Per-class F1 chart | Quan sát các lớp có F1 cao nhất và thấp nhất của mô hình tốt nhất. |
| Histogram phân phối F1 | Đánh giá phân phối hiệu năng theo từng lớp. |
| Boxplot cross-validation | So sánh độ ổn định của top 3 mô hình. |

Các biểu đồ được hiển thị trong notebook bằng `plt.show()`, nhưng code không lưu biểu đồ thành file ảnh riêng. Vì vậy báo cáo không thể nhúng trực tiếp các ảnh biểu đồ từ workspace.

### 6.5. Phân tích điểm mạnh và điểm yếu của mô hình

#### Điểm mạnh

- SVM RBF đạt Accuracy rất cao, 0,9801, phù hợp với bài toán phân loại ký tự dựa trên hình dạng.
- Macro F1 gần bằng Accuracy, cho thấy mô hình hoạt động đồng đều trên các lớp, không chỉ tốt ở lớp đông mẫu.
- Dataset cân bằng nên đánh giá bằng Accuracy có ý nghĩa hơn so với trường hợp dữ liệu lệch lớp nặng.
- Cross-validation của SVM RBF đạt 0,9740 ± 0,0027, thể hiện độ ổn định tốt.
- Không có lớp nào có F1 dưới 0,6 trong kết quả per-class F1 của mô hình tốt nhất.

#### Điểm yếu

- Không có validation set riêng để theo dõi mô hình trong quá trình tinh chỉnh.
- Không có augmentation nên mô hình có thể nhạy với biến đổi chữ viết như xoay nhẹ, lệch vị trí, độ dày nét hoặc nhiễu.
- Không resize trong `preprocess_image`, trong khi `IMG_SIZE` được khai báo nhưng không dùng. Nếu dataset sau này có kích thước ảnh không đồng nhất, HOG có thể tạo vector khác chiều và gây lỗi.
- Không có đánh giá trên dữ liệu ngoài thực tế, ví dụ ảnh viết tay từ người dùng hoặc ảnh chụp bằng camera.
- Không lưu biểu đồ thành file, gây khó khăn khi tái sử dụng kết quả trong báo cáo hoặc thuyết trình.
- Không có so sánh với mô hình deep learning như CNN, ResNet hoặc EfficientNet.

## 7. Advantages and Limitations

### 7.1. Advantages

Phương pháp HOG kết hợp học máy cổ điển có nhiều ưu điểm trong bài toán nhận dạng Kanji:

| Ưu điểm | Phân tích |
|---|---|
| Dễ hiểu và dễ giải thích | HOG mô tả hướng cạnh và cấu trúc nét chữ, phù hợp với ký tự Kanji có nhiều stroke. |
| Không cần GPU | Các mô hình scikit-learn có thể huấn luyện trên CPU, thuận tiện cho môi trường Colab cơ bản. |
| Hiệu năng cao | SVM RBF đạt Accuracy 0,9801 trên test set. |
| Pipeline rõ ràng | Code dùng `Pipeline` để kết hợp scaler, PCA và model, giúp hạn chế sai sót khi train/test. |
| Dataset cân bằng | Mỗi lớp có khoảng 200-201 ảnh, giúp mô hình không bị thiên lệch mạnh về lớp đông mẫu. |
| Có nhiều mô hình so sánh | Notebook thử nghiệm 7 mô hình, giúp chọn lựa dựa trên kết quả thực nghiệm. |
| Có đánh giá đa metric | Dùng Accuracy, Macro Precision, Macro Recall, Macro F1, confusion matrix và cross-validation. |
| Có lưu model | Mô hình sau train được lưu bằng `joblib`, thuận tiện triển khai lại. |
| PCA giảm chiều hiệu quả | Từ 1.512 chiều HOG, PCA giữ 95% variance với 197 thành phần, giảm đáng kể độ phức tạp. |

### 7.2. Limitations

Các hạn chế chính của code và phương pháp hiện tại:

| Hạn chế | Phân tích |
|---|---|
| Nguồn dataset chưa rõ | Code chỉ cho biết đường dẫn Google Drive, không mô tả dataset đến từ đâu. |
| Không có validation set riêng | Việc tinh chỉnh chủ yếu dựa vào cross-validation và test set; chưa có tập validation độc lập. |
| Không augmentation | Mô hình có thể kém bền vững với ảnh bị xoay, lệch, nhiễu hoặc thay đổi độ dày nét. |
| Không resize trong pipeline | Nếu ảnh đầu vào không đồng nhất kích thước, HOG có thể sinh vector khác chiều. |
| `IMG_SIZE` chưa được sử dụng | Biến được khai báo và lưu metadata nhưng không tham gia tiền xử lý. |
| Không có kiểm thử trên dữ liệu thực tế | Chưa đánh giá với ảnh viết tay mới hoặc ảnh từ môi trường triển khai. |
| Không có deep learning baseline | Chưa biết CNN hoặc pretrained model có vượt HOG+SVM trên cùng dataset hay không. |
| Xử lý ảnh còn đơn giản | Chưa có thresholding, căn giữa ký tự, loại nhiễu, chuẩn hóa độ dày nét. |
| Không lưu chart thành file | Các biểu đồ chỉ hiển thị trong notebook, không xuất ra thư mục kết quả. |
| Tốn chi phí inference hơn với SVM probability | `probability=True` trong SVM RBF có thể làm huấn luyện/chạy dự đoán nặng hơn nếu không cần xác suất. |

## 8. Conclusion

Notebook đã xây dựng thành công một pipeline nhận dạng Kanji dựa trên HOG và các mô hình học máy cổ điển. Dữ liệu gồm 16.079 ảnh thuộc 80 lớp Kanji, được chia train/test theo tỷ lệ 80/20 có giữ phân phối lớp. Đặc trưng HOG 1.512 chiều được chuẩn hóa bằng `StandardScaler` và giảm chiều bằng PCA giữ 95% phương sai.

Kết quả thực nghiệm cho thấy `SVM (RBF)` là mô hình tốt nhất với Accuracy 0,9801 và Macro F1 0,9801 trên test set. Cross-validation 5-fold của SVM RBF đạt 0,9740 ± 0,0027, thể hiện hiệu năng ổn định. Điều này chứng minh rằng HOG là đặc trưng phù hợp cho bài toán nhận dạng ký tự Kanji trong điều kiện dataset đã được chuẩn bị tương đối sạch và cân bằng.

Về ý nghĩa thực tiễn, mô hình có thể được dùng làm thành phần nhận dạng Kanji trong ứng dụng học tiếng Nhật, kiểm tra ký tự, hoặc hỗ trợ luyện tập Kanji N5. Pipeline có ưu điểm là dễ triển khai, không cần GPU và có thể lưu lại bằng `joblib`.

Tuy nhiên, hệ thống vẫn còn một số hạn chế: chưa rõ nguồn gốc dataset, chưa có validation set riêng, chưa có data augmentation, chưa kiểm thử trên dữ liệu ngoài thực tế và chưa so sánh với mô hình deep learning. Ngoài ra, biến `IMG_SIZE` được khai báo nhưng chưa được dùng để chuẩn hóa kích thước ảnh, có thể gây rủi ro nếu dữ liệu đầu vào thay đổi.

Các hướng cải thiện trong tương lai gồm:

- Thêm bước resize/căn giữa ảnh để đảm bảo kích thước đầu vào thống nhất.
- Áp dụng augmentation như xoay nhẹ, dịch chuyển, dilation/erosion và thay đổi độ dày nét.
- Lưu biểu đồ đánh giá thành file ảnh để phục vụ báo cáo và tái lập thí nghiệm.
- Thử nghiệm CNN, ResNet, EfficientNet hoặc mô hình lightweight cho nhận dạng ảnh.
- Đánh giá trên dữ liệu thực tế do người dùng viết tay hoặc chụp từ camera.
- Tối ưu inference nếu triển khai lên web hoặc thiết bị cấu hình thấp.
