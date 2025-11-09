import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, lab2rgb, deltaE_ciede2000
import matplotlib.pyplot as plt
import webcolors
from collections import deque

class ImageAnalyzer:
    def __init__(self, i_path:str, o_path:str='', n_colors:int=45, delta_e_threshold:int=15, 
                 max_color_count:int=80, min_cluster_size:int=7):
        self.input_path = i_path
        self.output_path = o_path if o_path != '' else "output-" + i_path
        # TODO validate paths
        self.N_COLORS = n_colors
        self.DELTA_E_THRESHOLD = delta_e_threshold
        self.MAX_COLOR_COUNT = max_color_count
        self.MIN_CLUSTER_SIZE = min_cluster_size

        self.pixels_lab = None #TODO these probably shouldnt default to None. doesn't matter two much
        self.kmeans = None
        self.palette_lab = None
        self.merged_palette_lab = None
        self.segmented_lab = None
        self.coordinates = []

        # Extract colors, merge, and segment colors
        img_bgr = cv2.imread(self.input_path)
        self.img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.extract_color()
        self.merge_color()
        self.segmentation()

        self.color_counts = dict()
        midpoints = self.find_clusters_midpoints(self.segmented_rgb)
        # for c in midpoints:
        #     print(c)

        self.visualize_results(midpoints)    

    def get_coordinates(self):
        """Returns a list of coordinates approximating hold location"""
        print(self.coordinates)
        return self.coordinates

    def extract_color(self):
        """Runs KMeans"""
        self.pixels_lab = rgb2lab(self.img_rgb.reshape(-1, 1, 3) / 255.0)[:, 0, :]
        print(f"Running KMeans on {len(self.pixels_lab):,} pixels in Lab space...")

        self.kmeans = KMeans(n_clusters=self.N_COLORS, random_state=42, n_init="auto")
        self.kmeans.fit(self.pixels_lab)

        self.palette_lab = self.kmeans.cluster_centers_
        labels = self.kmeans.labels_

        # h, w, _ = self.img_rgb.shape
        # palette_rgb = (lab2rgb(palette_lab[np.newaxis, :, :])[0] * 255).astype(np.uint8)
        # quantized_lab = palette_lab[labels].reshape(h, w, 3)
        # quantized_rgb = (lab2rgb(quantized_lab) * 255).astype(np.uint8)

    def merge_color(self):
        """Delta E in lab space"""
        self.merged_palette_lab = []
        used = np.zeros(len(self.palette_lab), dtype=bool)

        for i, c1 in enumerate(self.palette_lab):
            if used[i]:
                continue
            group = [i]
            for j, c2 in enumerate(self.palette_lab):
                if i != j and not used[j]:
                    if deltaE_ciede2000(c1, c2) < self.DELTA_E_THRESHOLD:
                        group.append(j)
                        used[j] = True
            used[i] = True
            avg_lab = np.mean(self.palette_lab[group], axis=0)
            self.merged_palette_lab.append(avg_lab)

        self.merged_palette_lab = np.array(self.merged_palette_lab)
        merged_palette_rgb = (lab2rgb(self.merged_palette_lab[np.newaxis, :, :])[0] * 255).astype(np.uint8)

        print(f"Merged {len(self.palette_lab)} → {len(self.merged_palette_lab)} colors.")

        return self.merged_palette_lab

    def segmentation(self):
        """Cluster-based masking in Lab space"""
        dists = np.linalg.norm(self.pixels_lab[:, None, :] - self.merged_palette_lab[None, :, :], axis=2)
        closest_idx = np.argmin(dists, axis=1)
        h, w, _ = self.img_rgb.shape
        self.segmented_lab = self.merged_palette_lab[closest_idx].reshape(h, w, 3)
        self.segmented_rgb = (lab2rgb(self.segmented_lab) * 255).astype(np.uint8)

    def closest_colour(self, requested_colour):
        distances = {}
        for name in webcolors.names():
            r_c, g_c, b_c = webcolors.name_to_rgb(name)
            rd = (r_c - requested_colour[0]) ** 2
            gd = (g_c - requested_colour[1]) ** 2
            bd = (b_c - requested_colour[2]) ** 2
            distances[name] = rd + gd + bd
        return min(distances, key=distances.get)

    def get_colour_name(self, requested_colour):
        try:
            closest_name = webcolors.rgb_to_name(requested_colour)
        except ValueError:
            closest_name = self.closest_colour(requested_colour)
        return closest_name

    def find_clusters_midpoints(self, arr, connectivity=4):
        n, m = arr.shape[:2]
        visited = np.zeros((n, m), dtype=bool)
        midpoints = []

        # directions for 4- or 8-connectivity
        if connectivity == 4:
            directions = [(1,0), (-1,0), (0,1), (0,-1)]
        else:
            directions = [(dr, dc) for dr in (-1,0,1) for dc in (-1,0,1) if not (dr==0 and dc==0)]

        for r in range(n):
            for c in range(m):
                if visited[r, c]:
                    continue
                color = tuple(arr[r, c])
                # BFS to find all pixels of the same cluster
                q = deque([(r, c)])
                cluster_pixels = []
                visited[r, c] = True
                while q:
                    x, y = q.popleft()
                    cluster_pixels.append((x, y))
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < n and 0 <= ny < m and not visited[nx, ny]:
                            if tuple(arr[nx, ny]) == color:
                                visited[nx, ny] = True
                                q.append((nx, ny))
                # compute midpoint
                xs, ys = zip(*cluster_pixels)
                mid_r, mid_c = np.mean(xs), np.mean(ys)
                if len(cluster_pixels) > self.MIN_CLUSTER_SIZE:
                    color_name = self.get_colour_name(color)
                    midpoints.append({
                        "color": [int(c) for c in color],
                        "color_name": self.get_colour_name(color),
                        "midpoint": (float(mid_r), float(mid_c)),
                        "size": len(cluster_pixels)
                    })
                    if color_name in self.color_counts:
                        self.color_counts[color_name] += 1
                    else:
                        self.color_counts[color_name] = 1

        for mp in midpoints:
            y, x = mp["midpoint"]
            color_name = mp["color_name"]
            if self.color_counts[color_name] < self.MAX_COLOR_COUNT:
                self.coordinates.append((x,y))
        print(self.coordinates)

        return midpoints

    def save_image(self, output_path):
        """"Saves output image to given output path"""
        Image.fromarray(self.segmented_rgb).save(output_path)
        print(f"Saved segmented image to {output_path}")

    def visualize_results(self, midpoints, max_color_count=100):
        # === VISUALIZE RESULTS ===
        fig, ax = plt.subplots(1, 2, figsize=(18, 6))

        ax[0].imshow(self.img_rgb)
        ax[0].set_title("Original (Lighting-Normalized) Image")
        ax[0].axis("off")

        ax[1].imshow(self.segmented_rgb)
        ax[1].set_title("Merged + Labeled Segmentation")
        ax[1].axis("off")

        for mp in midpoints:
            y, x = mp["midpoint"]     # note: imshow swaps axes
            color_name = mp["color_name"]
            if self.color_counts[color_name] < self.MAX_COLOR_COUNT:
                ax[0].plot(x, y, 'wo', color=tuple([c/255.0 for c in mp["color"]]), markersize=6, markeredgecolor='black', markeredgewidth=1.5)
                # ax[0].plot(x, y, 'wo', color=tuple([c/255.0 for c in mp["color"]]), markersize=6)
            # Optional: label with color or cluster id

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    file_name = "holds-small.jpg"
    file_path = rf"./images/{file_name}"
    image_analyzer = ImageAnalyzer(i_path=file_path, n_colors=45, delta_e_threshold=15, 
                max_color_count=40, min_cluster_size=7)
    #TODO changing this ^^^ and running is a lot better for making quick changes and seeing what happens.
    #Don't forget to update Nathan's main file with the optimal values
    print(image_analyzer.color_counts)
    print(image_analyzer.coordinates)