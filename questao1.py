import numpy as np 
import cv2
import matplotlib.pyplot as plt


def filtroHistograma(img):
    # Filtro passa-baixas (Gaussiano)
    low_pass = cv2.GaussianBlur(img, (3,3), 0)

    # Filtro de mediana 
    median_filtered = cv2.medianBlur(low_pass, 5)

    # Mostra resultados
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1), plt.imshow(img, cmap='gray'), plt.title("Original")
    plt.subplot(1,3,2), plt.imshow(low_pass, cmap='gray'), plt.title("Passa-Baixas (Gaussiano)")
    plt.subplot(1,3,3), plt.imshow(median_filtered, cmap='gray'), plt.title("Após Mediana")
    plt.show()

    cv2.imwrite("brain_filtro.jpg", median_filtered)

    # Histograma
    hist = cv2.calcHist([median_filtered], [0], None, [256], [0,256])
    plt.bar(range(256), hist.ravel(), width=1.0)
    plt.title("Histograma (OpenCV)")
    plt.xlabel("Intensidade")
    plt.ylabel("Número de pixels")
    plt.show()

    return median_filtered

def identificarTumor(img):
    _, binarizada = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    plt.imshow(binarizada, cmap="gray")

    cv2.imwrite("brain_binario.jpg", binarizada)

    # Estrutura (elemento estruturante)
    kernel = np.ones((5,5), np.uint8)

    # Abertura
    im_a = cv2.morphologyEx(binarizada, cv2.MORPH_OPEN, kernel)
    plt.imshow(im_a, cmap="gray")

    cv2.imwrite("brain_abertura.jpg", im_a)

    # Fechamento
    img_filtrada = cv2.morphologyEx(im_a, cv2.MORPH_CLOSE, kernel)
    plt.imshow(img_filtrada , cmap="gray")

    # Componentes conectados
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_filtrada , connectivity=8)

    # Seleciona segundo maior componente (ignora fundo)
    areas = stats[1:, cv2.CC_STAT_AREA]
    sorted_indices = np.argsort(areas)[::-1]  # ordem decrescente
    largest_label = 1 + sorted_indices[1] 

    # Cria máscara do maior componente
    mask = np.zeros_like(img_filtrada )
    mask[labels == largest_label] = 255

    # Obtem bounding box e centroide
    x, y, w, h, area = stats[largest_label]
    cx, cy = centroids[largest_label]

    # Converte para BGR para desenhar em cores
    output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Desenha bounding box
    cv2.rectangle(output, (x, y), (x+w, y+h), (0,255,0), 2)

    # Desenha centroide
    cv2.circle(output, (int(cx), int(cy)), 5, (0,0,255), -1)

    # Mostra resultados
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1), plt.imshow(img, cmap='gray'), plt.title("Original")
    plt.subplot(1,3,2), plt.imshow(mask, cmap='gray'), plt.title("Maior Componente (Tumor)")
    plt.subplot(1,3,3), plt.imshow(output[...,::-1]), plt.title("Bounding Box + Centroide")
    plt.show()

    cv2.imwrite("brain_tumor.jpg", mask)
    cv2.imwrite("brain_tumor_identificado.jpg", output[...,::-1])


def identificarTumorLimiar2(img):
    binarizada = cv2.inRange(img, 100, 200)
    plt.imshow(binarizada, cmap="gray")

    cv2.imwrite("brain_binario_limiar2.jpg", binarizada)

    # Estrutura (elemento estruturante)
    kernel = np.ones((5,5), np.uint8)

    # Abertura
    im_a = cv2.morphologyEx(binarizada, cv2.MORPH_OPEN, kernel)
    plt.imshow(im_a, cmap="gray")

    cv2.imwrite("brain_abertura_limiar2.jpg", im_a)

    img_filtrada = cv2.morphologyEx(im_a, cv2.MORPH_CLOSE, kernel)
    plt.imshow(img_filtrada, cmap="gray")

    cv2.imwrite("brain_fechamento_limiar2.jpg", img_filtrada)

    # Componentes conectados
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_filtrada , connectivity=8)

    # Seleciona maior componente (ignora fundo)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + np.argmax(areas)

    # Cria máscara do maior componente
    mask = np.zeros_like(img_filtrada )
    mask[labels == largest_label] = 255

    # Obtem bounding box e centroide
    x, y, w, h, area = stats[largest_label]
    cx, cy = centroids[largest_label]

    # Converte para BGR 
    output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Desenha bounding box (retângulo verde)
    cv2.rectangle(output, (x, y), (x+w, y+h), (0,255,0), 2)

    # Desenha centroide 
    cv2.circle(output, (int(cx), int(cy)), 5, (0,0,255), -1)

    # Mostra resultados
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1), plt.imshow(img, cmap='gray'), plt.title("Original")
    plt.subplot(1,3,2), plt.imshow(mask, cmap='gray'), plt.title("Maior Componente (Tumor)")
    plt.subplot(1,3,3), plt.imshow(output[...,::-1]), plt.title("Bounding Box + Centroide")
    plt.show()

    cv2.imwrite("brain_tumor_limiar2.jpg", mask)
    cv2.imwrite("brain_tumor_identificado_limiar2.jpg", output[...,::-1])

img = cv2.imread("brain.jpg", cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap="gray")

img_sem_ruido = filtroHistograma(img)

identificarTumor(img_sem_ruido)

identificarTumorLimiar2(img_sem_ruido)

