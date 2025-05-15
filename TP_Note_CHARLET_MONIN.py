import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import matplotlib as mpl
mpl.use('TkAgg')

# Fonction permettant de lire une image, identifier ces caractéristiques et donner ces canaux.

def lire_image(image_path):
    image = mpimg.imread(image_path)
    if len(image.shape) == 3:
        height, width, channels = image.shape
        print(f"Dimension: {width}x{height} | Canaux {channels}")
    elif len(image.shape) == 2:
        height, width = image.shape
        print(f"Dimension: {width}x{height} | Image en noir et blanc")
    else:
        raise ValueError("Format d'image non pris en charge.")
    return image

# Fonction permettant de restaurer une image en noir et blanc fournie et d'un masque de la ligne à éliminer

def restaurer_image_noir_et_blanc(gray_image, masque, k):
    if masque.shape != gray_image.shape:
        raise ValueError("Les dimensions du masque et de l'image ne correspondent pas.")

    gray_image_masked = np.where(masque != 0, 0, gray_image)
    U, s, Vt = np.linalg.svd(gray_image_masked, full_matrices=False)
    S = np.diag(s[:k])

    print("Matrice U:\n", U)
    print("Matrice S:\n", S)
    print("Matrice Vt:\n", Vt)

    approximated_image = U[:, :k].dot(S).dot(Vt[:k, :])
    restored_image = np.where(masque != 0, approximated_image, gray_image)

    return restored_image

# Fonction permettant de restaurer une image en couleur fournie et d'un masque de la ligne à éliminer

def restaurer_image_couleurs(image, masque, k):
    restored_image = np.copy(image)

    for channel in range(image.shape[2]):
        image_channel = image[:, :, channel]
        image_channel_masked = np.where(masque != 0, 0, image_channel)

        U, S, V = np.linalg.svd(image_channel_masked)

        print(f"Matrice U pour le cannal {channel}:\n", U)
        print(f"Matrice S pour le cannal {channel}:\n", S)
        print(f"Matrice V pour le cannal {channel}:\n", V)

        approx_image = np.dot(U[:, :k] * S[:k], V[:k, :])

        restored_image[:, :, channel] = np.where(masque != 0, approx_image, image_channel)

    return restored_image

img = cv2.imread('h:\\Desktop\\analysenum\\TP\\cheval-abime2-nb.png')

#Convertie l'image en gris

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Calcule le SVD

u, s, v = np.linalg.svd(gray_image, full_matrices=False)

#Permet de donner la taille des matrices.

print(f'u.shape:{u.shape},s.shape:{s.shape},v.shape:{v.shape}')

#Afiiche l'image avec différentes valeurs de composantes

comps = [50, 1, 5, 10, 15, 25]
plt.figure(figsize=(12, 6))

for i in range(len(comps)):
    low_rank = u[:, :comps[i]] @ np.diag(s[:comps[i]]) @ v[:comps[i], :]

    if (i == 0):
        plt.subplot(2, 3, i + 1),
        plt.imshow(low_rank, cmap='gray'),
        plt.title(f'Actual Image with n_components = {comps[i]}')

    else:
        plt.subplot(2, 3, i + 1),
        plt.imshow(low_rank, cmap='gray'),
        plt.title(f'n_components = {comps[i]}')
plt.show()

# Exemple d'utilisation

image_path = "cheval-abime2.png"
image_nb_path = "cheval-abime2-nb.png"
masque_path = "cheval-abime2-mask.png"

image = lire_image(image_path)
image_nb = lire_image(image_nb_path)
masque = mpimg.imread(masque_path)

k = 20

restored_nb_image = restaurer_image_noir_et_blanc(image_nb, masque, k)
restored_color_image = restaurer_image_couleurs(image, masque, k)

plt.figure()
plt.imshow(restored_nb_image, cmap="gray")
plt.title("Image restaurée en noir et blanc")
plt.show()

plt.figure()
plt.imshow(restored_color_image, cmap="gray")
plt.title("Image restaurée en couleur")
plt.show()