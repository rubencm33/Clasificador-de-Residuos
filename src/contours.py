import cv2


def get_main_object(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None

    # Extraer el contorno más grande (el objeto principal)
    cnt = max(contours, key=cv2.contourArea)

    # Obtener el recuadro delimitador (Bounding Box)
    x, y, w, h = cv2.boundingRect(cnt)

    return cnt, (x, y, w, h)