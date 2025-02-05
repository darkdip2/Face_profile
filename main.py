import os
import cv2
import numpy as np
import shutil
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
from ultralytics import YOLO
import random
import insightface
from insightface.app import FaceAnalysis
import warnings
warnings.simplefilter('ignore')


os.makedirs("detections", exist_ok=True)


def select_random_images(source_folder='train', destination_folder='gallery', min_images=1, max_images=5):
    os.makedirs(destination_folder, exist_ok=True)

    for person in os.listdir(source_folder):
        person_folder = os.path.join(source_folder, person)

        if os.path.isdir(person_folder): 
            images = [img for img in os.listdir(person_folder) if img.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            if images:  
                num_images = random.randint(min_images, max_images)  
                selected_images = random.sample(images, min(num_images, len(images)))  
                
                for index, img in enumerate(selected_images, start=1):
                    src_path = os.path.join(person_folder, img)
                    new_filename = f"{person}_{index}.jpg"  
                    dest_path = os.path.join(destination_folder, new_filename)

                    shutil.copy(src_path, dest_path) 

    print(f"Images successfully copied to {destination_folder}!")


select_random_images()

THRESHOLD=.5

app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']) 
app.prepare(ctx_id=0)

def extract_faces(image_path):

    image = cv2.imread(image_path)
    faces = app.get(image)  

    extracted_faces = []
    for face in faces:
        if face.det_score > THRESHOLD:  
            x1, y1, x2, y2 = map(int, face.bbox)
            face_crop = image[y1:y2, x1:x2]  
            extracted_faces.append((face_crop, face.normed_embedding, image_path))
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    output_path = os.path.join("detections", os.path.basename(image_path))
    cv2.imwrite(output_path, image)
    
    return extracted_faces



def process_gallery(folder_path):
    profiles = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            faces = extract_faces(os.path.join(folder_path, filename))
            for face, encoding, path in faces:
                profiles.append((filename, face, encoding, path))
    
    return profiles



def find_optimal_eps(embeddings, min_samples=5):
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors.fit(embeddings)
    distances, _ = neighbors.kneighbors(embeddings)

    distances = np.sort(distances[:, -1])

    plt.plot(distances)
    plt.xlabel("Points sorted by distance")
    plt.ylabel(f"{min_samples}-th Nearest Neighbor Distance")
    plt.title("DBSCAN Epsilon Selection (Elbow Method)")
    plt.grid()
    plt.show()

def tune_dbscan(embeddings, eps_range, min_samples_range):
    best_score = -1
    best_params = None
    
    for eps in eps_range:
        for min_samples in min_samples_range:
            clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit(embeddings)
            labels = clustering.labels_
            if len(set(labels)) > 1 and -1 in labels: 
                score = silhouette_score(embeddings, labels)
                if score > best_score:
                    best_score = score
                    best_params = (eps, min_samples)
    
    return best_params


def tune_kmeans(embeddings, k_range):
    inertia_scores = []
    silhouette_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(embeddings)

        inertia_scores.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(embeddings, labels))

    plt.figure(figsize=(8, 4))
    plt.plot(k_range, inertia_scores, marker='o')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia (Sum of Squared Distances)")
    plt.title("Elbow Method for Optimal K")
    plt.grid()
    plt.show()

    best_k = k_range[np.argmax(silhouette_scores)]
    print(f"Best K using Silhouette Score: {best_k}")

    return best_k


def cluster_faces(profiles, eps=0.5, min_samples=5, use_kmeans=False, n_clusters=None):
    encodings = np.array([profile[2] for profile in profiles])

    if use_kmeans:
        clustering = KMeans(n_clusters=n_clusters, random_state=42).fit(encodings)
        labels = clustering.labels_
    else:
        distance_matrix = cosine_distances(encodings)
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit(distance_matrix)
        labels = clustering.labels_

    clusters = {}
    for i, label in enumerate(labels):
        if label == -1:
            continue  # Ignore noise
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(profiles[i])
    
    return clusters


def find_representative_image(cluster):
    encodings = np.array([profile[2] for profile in cluster])
    centroid = np.mean(encodings, axis=0)

    distances = np.linalg.norm(encodings - centroid, axis=1)
    best_index = np.argmin(distances)

    return cluster[best_index] 



def save_profiles(clusters, output_folder="profiles"):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder) 
    os.makedirs(output_folder, exist_ok=True)

    for label, cluster in clusters.items():
        profile_folder = os.path.join(output_folder, f"profile_{label}")
        os.makedirs(profile_folder, exist_ok=True)

        for filename, face, _, path in cluster:
            save_path = os.path.join(profile_folder, filename)
            cv2.imwrite(save_path, face)

        rep_filename, rep_face, _, _ = find_representative_image(cluster)
        rep_image_path = os.path.join(profile_folder, "representative.jpg")
        cv2.imwrite(rep_image_path, rep_face)

def display_profiles(clusters):
    for label, cluster in clusters.items():
        _, rep_face, _, _ = find_representative_image(cluster)
        cv2.imshow(f"Profile {label}", rep_face)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()


folder_path = "gallery"
profiles = process_gallery(folder_path)

embeddings = np.array([profile[2] for profile in profiles])

max_k=len(embeddings)

print(f'{max_k} Datapoints')


find_optimal_eps(embeddings, min_samples=5)
eps_range = np.arange(0.3, 1.0, 0.1)
min_samples_range = range(3, 10)
#best_eps, best_min_samples = tune_dbscan(embeddings, eps_range, min_samples_range)
#print(f"Using DBSCAN with eps={best_eps}, min_samples={best_min_samples}")
#clusters = cluster_faces(profiles, eps=best_eps, min_samples=best_min_samples, use_kmeans=False)
#save_profiles(clusters, 'DBSCAN')


best_k = tune_kmeans(embeddings, k_range=range(2, max_k))
print(f"Using KMeans with n_clusters = {best_k}")
clusters = cluster_faces(profiles, use_kmeans=True, n_clusters=best_k)
save_profiles(clusters)

profiles_folder = "profiles"

for folder in os.listdir(profiles_folder):
    folder_path = os.path.join(profiles_folder, folder)
    l=len(os.listdir(folder_path))
    if l < 3:
        print(f"Deleting folder: {folder_path} (contains {l} images)")
        shutil.rmtree(folder_path) 


print("Profiles saved successfully in 'profiles/' folder.")

#display_profiles(clusters)



