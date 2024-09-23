import json

import json
import os
import shutil

Count = 20


def sample_dataset(source_file_path, dist_file_path, target_folder):
    # 创建目标文件夹
    os.makedirs(target_folder, exist_ok=True)

    # 用于计数的变量
    border_count = 0
    other_count = 0

    # 读取原始标注文件
    with open(source_file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # 过滤并处理数据
    filtered_lines = []
    for line in lines:
        data = json.loads(line.strip())

        # 获取原始 filename 和新 filename
        original_filename = data["filename"]
        new_filename = original_filename.replace("images/", "")

        # 读取原始图片并保存到新文件夹

        # 修改 filename
        data["file_name"] = new_filename
        del data["filename"]
        # 将 gt 的值复制到 html 并删除旧的 html 字段
        data["html"] = data.pop("gt")
        # 检查前缀并决定是否保留
        if new_filename.startswith("border_"):
            if border_count < Count:
                filtered_lines.append(json.dumps(data, ensure_ascii=False))
                src_img_path = os.path.join(
                    os.path.dirname(source_file_path), original_filename
                )
                dst_img_path = os.path.join(target_folder, new_filename)
                shutil.copy(src_img_path, dst_img_path)
                border_count += 1
        else:
            if other_count < Count:
                filtered_lines.append(json.dumps(data, ensure_ascii=False))
                src_img_path = os.path.join(
                    os.path.dirname(source_file_path), original_filename
                )
                dst_img_path = os.path.join(target_folder, new_filename)
                shutil.copy(src_img_path, dst_img_path)
                other_count += 1

        # 如果已经收集了足够的样本，则提前结束循环
        if border_count >= Count and other_count >= Count:
            break

    # 将过滤后的数据写入新文件
    with open(dist_file_path, "a", encoding="utf-8") as file:
        for line in filtered_lines:
            file.write(line + "\n")

    print(f"处理完成，共保留 {border_count + other_count} 条记录。")


def process_jsonl_file(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as infile, open(
        output_file, "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            data = json.loads(line)

            # 删除 html 字段
            if "html" in data:
                del data["html"]

            # 将 gt 字段移动到 html 字段的位置
            if "gt" in data:
                data["html"] = data.pop("gt")

            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # input_file_1 = '/Users/macbookc23551/PycharmProjects/TableStructureRec/outputs/train.txt'
    # output_file_1 = 'processed_file1.jsonl'
    # process_jsonl_file(input_file_1, output_file_1)
    sample_dataset(
        f"outputs/table_rec_dataset/val.jsonl",
        f"outputs/benchmark/metadata.jsonl",
        f"outputs/benchmark",
    )
