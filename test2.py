import cv2
import numpy as np

def suggest_and_adjust_roi_interactive(image_path, output_size=224, padding_ratio=0.1, thresh_percent=0.6):
    """
    Sugiere un ROI basado en threshold adaptativo y permite ajuste manual interactivo.
    Muestra la predicción inicial como recuadro y devuelve coordenadas finales.
    """
    # --- Leer imagen ---
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("No se pudo leer la imagen")
    H, W = img.shape[:2]
    original = img.copy()

    # --- Convertir a grayscale y suavizar ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    # --- Threshold adaptativo ---
    max_val = np.max(gray)
    thresh_val = int(max_val * thresh_percent)
    _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

    # --- Morphology ---
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # --- Contornos ---
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) > 50]
    if len(valid) == 0:
        raise ValueError("No se detectó mano para sugerir ROI")

    # --- Combinar contornos ---
    all_points = np.vstack(valid)
    x, y, w, h = cv2.boundingRect(all_points)

    # --- Padding ---
    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)
    x1 = max(x - pad_x, 0)
    y1 = max(y - pad_y, 0)
    x2 = min(x + w + pad_x, W)
    y2 = min(y + h + pad_y, H)

    # --- Dibujar recuadro sugerido ---
    img_suggest = img.copy()
    cv2.rectangle(img_suggest, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(img_suggest, "ROI sugerido", (x1, max(y1-5,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # --- Selección interactiva ---
    r = cv2.selectROI("Ajusta ROI si quieres", img_suggest, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    # --- Si no se ajusta manualmente, usar sugerido ---
    if r == (0,0,0,0):
        x1_final, y1_final, w_final, h_final = x1, y1, x2-x1, y2-y1
    else:
        x1_final, y1_final, w_final, h_final = r
    x2_final = x1_final + w_final
    y2_final = y1_final + h_final

    # --- Recortar ROI ---
    roi = original[y1_final:y2_final, x1_final:x2_final]

    # --- Convertir a cuadrado ---
    h_roi, w_roi = roi.shape[:2]
    size = max(h_roi, w_roi)
    square = np.zeros((size, size, 3), dtype=np.uint8)
    y_offset = (size - h_roi)//2
    x_offset = (size - w_roi)//2
    square[y_offset:y_offset+h_roi, x_offset:x_offset+w_roi] = roi

    # --- Resize final ---
    roi_resized = cv2.resize(square, (output_size, output_size), interpolation=cv2.INTER_LINEAR)

    # --- Imprimir coordenadas finales ---
    print(f"Coordenadas del ROI final (en pixeles de la imagen original):")
    print(f"  Superior (y1): {y1_final}")
    print(f"  Inferior (y2): {y2_final}")
    print(f"  Izquierda (x1): {x1_final}")
    print(f"  Derecha (x2): {x2_final}")

    return roi_resized

# ===============================
# Uso
# ===============================
roi_final = suggest_and_adjust_roi_interactive("processed_data/fabrizio_madera/raw_jpg/fab_mad_test1_snap0001_0001s.jpg", output_size=224)

# Mostrar ROI final
cv2.imshow("ROI 224x224", roi_final)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
Determina el ROI en la imagen RGB (automático o manual).
Obtén las coordenadas exactas del ROI: (x1, y1, x2, y2).
Haz crop sobre la matriz térmica usando esas mismas coordenadas:
Ahora la matriz térmica 224×224 coincide espacialmente con la imagen RGB recortada.
Guarda o usa la matriz térmica recortada para análisis o entrenamiento.
Esto mantiene todos los valores de temperatura solo del ROI seleccionado.
"""