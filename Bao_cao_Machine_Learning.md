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

