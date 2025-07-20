import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def load_templates(template_dir):
    """
    加载所有模板图像为灰度图
    """
    templates = {}
    for filename in os.listdir(template_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            path = os.path.join(template_dir, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                templates[filename] = img
            else:
                print(f"模板 {filename} 加载失败")
    return templates


def nms(boxes, scores, iou_threshold=0.3):
    """
    非极大值抑制
    """
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = x1 + boxes[:, 2]
    y2 = y1 + boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / (union + 1e-6)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def match_templates(scene_img, templates, threshold=0.8, iou_thresh=0.01):
    """
    所有模板统一 NMS，避免重复框
    """
    result_img = scene_img.copy()
    gray_scene = cv2.cvtColor(scene_img, cv2.COLOR_BGR2GRAY)

    all_boxes = []
    all_scores = []
    all_labels = []

    for name, tmpl in templates.items():
        res = cv2.matchTemplate(gray_scene, tmpl, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        h, w = tmpl.shape

        for pt in zip(*loc[::-1]):
            all_boxes.append([pt[0], pt[1], w, h])
            all_scores.append(res[pt[1], pt[0]])
            all_labels.append(name)

    # NMS on all boxes
    keep_idxs = nms(all_boxes, all_scores, iou_threshold=iou_thresh)

    total_count = 0
    for i in keep_idxs:
        x, y, w, h = all_boxes[i]
        score = all_scores[i]
        label = all_labels[i]
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(result_img, f"{label} ({score:.2f})", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        total_count += 1

    print(f"\n共检测到 {total_count} 个匹配区域（全局NMS后）")
    return result_img



def detect_with_template_matching(scene_path, template_dir, threshold=0.8):
    """
    主流程：加载图像、模板，执行匹配并显示结果
    """
    if not os.path.exists(scene_path):
        print(f"图像文件不存在: {scene_path}")
        return

    scene_img = cv2.imread(scene_path)
    if scene_img is None:
        print("场景图像加载失败")
        return

    templates = load_templates(template_dir)
    if not templates:
        print("未能加载任何模板")
        return

    result_img = match_templates(scene_img, templates, threshold)

    # 显示检测结果
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title("模板匹配 + NMS 检测结果")
    plt.axis('off')
    plt.show()

    # 保存结果图
    save_path = "matched_result_nms.png"
    cv2.imwrite(save_path, result_img)
    print(f"检测结果已保存为 {save_path}")


if __name__ == "__main__":
    scene_img_path = "./imgs/scene.png"
    templates_folder = "./templates"
    detect_with_template_matching(scene_img_path, templates_folder, threshold=0.7)

