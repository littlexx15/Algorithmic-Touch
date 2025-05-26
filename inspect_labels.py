# 专门用来探查模型的标签映射
# inspect_labels.py

from transformers import pipeline

def main():
    # 用和你项目中同样的模型 ID
    clf = pipeline(
        "image-classification",
        model="abdlh/ResNet34_finetuned_for_skin_diseases_by-abdlh"
    )
    # 打印出模型里每个输出 ID 对应的标签
    print("模型支持的标签 (id2label):")
    for idx, label in clf.model.config.id2label.items():
        print(f"{idx}: {label}")

if __name__ == "__main__":
    main()
