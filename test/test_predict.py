from PIL import Image
from models.predict import predict, class_names

def test_predict_output_shape_and_sum(tmp_path):
    # 用一张纯白图片（或者项目里随便放一张小图）测试
    img = Image.new("RGB", (224, 224), color="white")
    out = predict(img)  # 返回的是 dict[class_name→prob]
    # 应该有 7 个类别
    assert set(out.keys()) == set(class_names)
    # 概率和应为 1（允许一点点浮点误差）
    total = sum(out.values())
    assert abs(total - 1.0) < 1e-3
