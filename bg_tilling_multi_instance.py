import os
import json
import cv2
import numpy as np
from glob import glob
from multiprocessing import Pool, cpu_count


class PolygonTilerFromSaved:
    def __init__(self, image_folder, polygon_folder, output_json_folder, output_vis_folder, num_boxes=5):
        self.image_paths = sorted(glob(os.path.join(image_folder, "*.png")))
        self.polygon_folder = polygon_folder
        self.output_json_folder = output_json_folder
        self.output_vis_folder = output_vis_folder
        self.num_boxes = num_boxes

        os.makedirs(self.output_json_folder, exist_ok=True)
        os.makedirs(self.output_vis_folder, exist_ok=True)

    def process_image(self, image_path):
        image_name = os.path.basename(image_path)
        base_name = os.path.splitext(image_name)[0]
        polygon_path = os.path.join(self.polygon_folder, f"{base_name}_polygon.json")

        if not os.path.exists(polygon_path):
            return f"⚠️ No polygon found for {image_name}"

        with open(polygon_path, "r") as f:
            polygon_points = json.load(f)

        if len(polygon_points) < 4:
            return f"❌ Polygon for {image_name} has fewer than 4 points"

        img = cv2.imread(image_path)
        bboxes = self.tile_boxes(polygon_points, self.num_boxes)
        if bboxes:
            self.save_json(bboxes, base_name)
            self.save_visualization(img, polygon_points, bboxes, base_name)
            return f"✅ Processed {image_name}"
        return f"⚠️ Skipped {image_name}: could not place boxes"

    def process_all_images(self):
        with Pool(cpu_count()) as pool:
            results = pool.map(self.process_image, self.image_paths)
        for res in results:
            print(res)

    def tile_boxes(self, polygon_points, num_boxes):
        polygon_points = np.array(polygon_points, dtype=np.float32)
        x, y, w, h = cv2.boundingRect(polygon_points)
        width, height = w, h

        src_rect = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
        dst_rect = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(dst_rect, src_rect)
        poly_path = cv2.convexHull(polygon_points)

        min_size_limit = int(0.05 * min(width, height))
        big_size = int(0.5 * min(width, height))
        small_size = int(0.3 * min(width, height))

        num_big = int((2 / 5) * num_boxes)
        num_small = num_boxes - num_big

        def adaptive_spacing(size):
            return int(size * 0.5)

        def try_place_fixed_size(size, count, label, avoid=[], used_anchors=set()):
            spacing = adaptive_spacing(size)
            attempts = 0
            max_attempts = 10000
            boxes = []

            while len(boxes) < count and attempts < max_attempts:
                # --- Bottom-biased Y sampling ---
                y_min_bias = int(height * 0.4)
                y_max_bias = int(height * 0.9)
                y = np.random.randint(y_min_bias, min(height - size, y_max_bias))

                x = np.random.randint(0, width - size)
                unique_anchor = (x // 10, y // 10)
                if unique_anchor in used_anchors:
                    attempts += 1
                    continue

                rect = np.array([
                    [x, y],
                    [x + size, y],
                    [x + size, y + size],
                    [x, y + size]
                ], dtype=np.float32)

                rect_h = np.concatenate([rect, np.ones((4, 1))], axis=1)
                transformed = (M @ rect_h.T).T
                transformed /= transformed[:, 2][:, None]
                transformed = transformed[:, :2]

                if not all(cv2.pointPolygonTest(poly_path, (pt[0], pt[1]), False) >= 0 for pt in transformed):
                    attempts += 1
                    continue

                xs, ys = transformed[:, 0], transformed[:, 1]
                xmin, xmax = xs.min(), xs.max()
                ymin, ymax = ys.min(), ys.max()

                overlap = False
                for prev in avoid + boxes:
                    if (xmin < prev["xmax"] + spacing and xmax > prev["xmin"] - spacing and
                        ymin < prev["ymax"] + spacing and ymax > prev["ymin"] - spacing):
                        overlap = True
                        break
                if overlap:
                    attempts += 1
                    continue

                cx, cy = xs.mean(), ys.mean()
                side = max(xmax - xmin, ymax - ymin) / 2

                boxes.append({
                    "xmin": cx - side,
                    "ymin": cy - side,
                    "xmax": cx + side,
                    "ymax": cy + side,
                    "size": label
                })

                used_anchors.add(unique_anchor)

            return boxes

        used_anchors = set()
        big_boxes, small_boxes = [], []

        while big_size >= min_size_limit:
            big_boxes = try_place_fixed_size(big_size, num_big, "big", used_anchors=used_anchors)
            if len(big_boxes) == num_big:
                break
            big_size -= 5

        while small_size >= min_size_limit:
            small_boxes = try_place_fixed_size(small_size, num_small, "small", avoid=big_boxes, used_anchors=used_anchors)
            if len(small_boxes) == num_small:
                break
            small_size -= 5

        if len(big_boxes) != num_big or len(small_boxes) != num_small:
            return []

        return big_boxes + small_boxes


    def save_json(self, bboxes, base_name):
        output_path = os.path.join(self.output_json_folder, f"{base_name}_bboxes.json")
        with open(output_path, "w") as f:
            json.dump(bboxes, f, indent=4)
        print(f"✅ Saved {len(bboxes)} boxes to {output_path}")

    def save_visualization(self, img, polygon_points, bboxes, base_name):
        vis_img = img.copy()
        polygon_points = np.array(polygon_points, dtype=np.int32)
        cv2.polylines(vis_img, [polygon_points], isClosed=True, color=(0, 255, 255), thickness=2)

        for bbox in bboxes:
            color = (0, 0, 255) if bbox["size"] == "big" else (0, 255, 0)
            cv2.rectangle(
                vis_img,
                (int(bbox["xmin"]), int(bbox["ymin"])),
                (int(bbox["xmax"]), int(bbox["ymax"])),
                color,
                2
            )

        output_img_path = os.path.join(self.output_vis_folder, f"{base_name}_vis.png")
        cv2.imwrite(output_img_path, vis_img)
        print(f"🖼️  Saved visual check image to {output_img_path}")


# === Run Tiler for All Prompts ===
base_dir = "../data/images/bg/final"
num_boxes = 5

for folder_name in sorted(os.listdir(base_dir)):
    if not folder_name.startswith("prompt"):
        continue

    prompt_path = os.path.join(base_dir, folder_name)
    polygon_path = os.path.join(prompt_path, "polygons")

    if not os.path.isdir(polygon_path):
        print(f"❌ Skipping {folder_name}: no polygon folder")
        continue

    print(f"🚀 Processing {folder_name}")
    image_folder = prompt_path
    output_json_folder = os.path.join(prompt_path, "tiling_multi_json")
    output_vis_folder = os.path.join(prompt_path, "tiling_multi_preview")

    tiler = PolygonTilerFromSaved(
        image_folder=image_folder,
        polygon_folder=polygon_path,
        output_json_folder=output_json_folder,
        output_vis_folder=output_vis_folder,
        num_boxes=num_boxes
    )
    tiler.process_all_images()