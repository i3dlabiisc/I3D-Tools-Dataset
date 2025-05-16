import os
import json
import random
import yaml
import cv2
import numpy as np
from glob import glob
from multiprocessing import Pool, cpu_count

class PolygonTilerFromSaved:
    def __init__(self, image_folder, polygon_folder, output_json_folder, output_vis_folder, n=3, m=4, step=0.05):
        self.image_paths = sorted(glob(os.path.join(image_folder, "*.png")))
        self.polygon_folder = polygon_folder
        self.output_json_folder = output_json_folder
        self.output_vis_folder = output_vis_folder
        self.n = n  # upper limit (smallest box size = 1/n)
        self.m = m  # lower limit (largest box size = 1/m)
        self.step = step  # step size between m and n

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
        bboxes = self.tile_one_box(polygon_points)
        if bboxes:
            self.save_json(bboxes, base_name)
            # self.save_visualization(img, polygon_points, bboxes, base_name)
            return f"✅ Processed {image_name}"
        return f"⚠️ Skipped {image_name}: could not place box"

    def process_all_images(self):
        with Pool(cpu_count()) as pool:
            results = pool.map(self.process_image, self.image_paths)
        for res in results:
            print(res)

    def tile_one_box(self, polygon_points):
        polygon_points = np.array(polygon_points, dtype=np.float32)
        x, y, w, h = cv2.boundingRect(polygon_points)
        width, height = w, h

        src_rect = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
        dst_rect = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(dst_rect, src_rect)
        poly_path = cv2.convexHull(polygon_points)

        # === Create random size pool ===
        lower = self.m
        upper = self.n
        step_size = self.step

        possible_sizes = []
        val = lower
        while val <= upper + 1e-6:
            possible_sizes.append(val)
            val += step_size

        if not possible_sizes:
            raise ValueError("No valid box sizes generated. Check n, m, and step.")

        chosen_factor = random.choice(possible_sizes)
        original_box_size = min(width, height) * chosen_factor
        min_box_size = int(0.05 * min(width, height))

        box_size = original_box_size
        shrink_factor = 0.9
        pixel_step = 2

        # === Offsets ===
        offset_y_fraction = 0.05
        offset_x_fraction = 0.05
        offset_y = height * offset_y_fraction
        offset_x = width * offset_x_fraction

        max_random_attempts = 50

        while box_size > min_box_size:
            placed = False
            for y_scan in np.arange(height - box_size - offset_y, 0, -pixel_step):
                for _ in range(max_random_attempts):
                    x_min_allowed = offset_x
                    x_max_allowed = width - box_size - offset_x
                    if x_min_allowed >= x_max_allowed:
                        break

                    x_scan = np.random.uniform(x_min_allowed, x_max_allowed)

                    rect = np.array([
                        [x_scan, y_scan],
                        [x_scan + box_size, y_scan],
                        [x_scan + box_size, y_scan + box_size],
                        [x_scan, y_scan + box_size]
                    ], dtype=np.float32)

                    rect_h = np.concatenate([rect, np.ones((4, 1))], axis=1)
                    transformed = (M @ rect_h.T).T
                    transformed /= transformed[:, 2][:, None]
                    transformed = transformed[:, :2]

                    if all(cv2.pointPolygonTest(poly_path, (pt[0], pt[1]), False) >= 0 for pt in transformed):
                        xs, ys = transformed[:, 0], transformed[:, 1]
                        xmin, xmax = xs.min(), xs.max()
                        ymin, ymax = ys.min(), ys.max()

                        box = [{
                            "xmin": xmin,
                            "ymin": ymin,
                            "xmax": xmax,
                            "ymax": ymax,
                            "size": f"{chosen_factor:.2f} of bbox (square, randomized)"
                        }]
                        return box

            box_size *= shrink_factor

        print(f"❌ WARNING: Could not place even after shrinking. Polygon may be invalid.")
        return []

    def save_json(self, bboxes, base_name):
        output_path = os.path.join(self.output_json_folder, f"{base_name}_bboxes.json")
        with open(output_path, "w") as f:
            json.dump(bboxes, f, indent=4)
        print(f"✅ Saved 1 box to {output_path}")

    def save_visualization(self, img, polygon_points, bboxes, base_name):
        vis_img = img.copy()
        polygon_points = np.array(polygon_points, dtype=np.int32)
        cv2.polylines(vis_img, [polygon_points], isClosed=True, color=(0, 255, 255), thickness=2)

        for bbox in bboxes:
            cv2.rectangle(
                vis_img,
                (int(bbox["xmin"]), int(bbox["ymin"])),
                (int(bbox["xmax"]), int(bbox["ymax"])),
                (255, 0, 0),
                2
            )

        output_img_path = os.path.join(self.output_vis_folder, f"{base_name}_vis.png")
        cv2.imwrite(output_img_path, vis_img)
        print(f"🖼️  Saved visual check image to {output_img_path}")

# === Load tilling parameters from YAML ===
tilling_param_path = "../data/images/fg/tilling_params.yaml"
with open(tilling_param_path, "r") as f:
    tilling_params = yaml.safe_load(f)

# === Run Tiler for All Prompts ===
base_dir = "../data/images/bg/selected"
step = 0.05  # step size in randomization

for folder_name in sorted(os.listdir(base_dir)):
    if not folder_name.startswith("prompt"):
        continue

    prompt_path = os.path.join(base_dir, folder_name)
    polygon_path = os.path.join(prompt_path, "polygons")

    if not os.path.isdir(polygon_path):
        print(f"❌ Skipping {folder_name}: no polygon folder")
        continue

    print(f"\n🚀 Processing {folder_name}")

    for tool_name, params in tilling_params.items():
        m = params["scale_lower_limit"]
        n = params["scale_upper_limit"]

        print(f"🔧 tilling for tool '{tool_name}' with m={m}, n={n}")

        output_json_folder = os.path.join(prompt_path, f"tilling_json_{tool_name}")
        output_vis_folder = os.path.join(prompt_path, f"tilling_preview_{tool_name}")

        tiler = PolygonTilerFromSaved(
            image_folder=prompt_path,
            polygon_folder=polygon_path,
            output_json_folder=output_json_folder,
            output_vis_folder=output_vis_folder,
            n=n,
            m=m,
            step=step
        )
        tiler.process_all_images()
