import numpy as np
import cv2


class Rect(object):
    def __init__(self, x1, y1, x2, y2):
        super().__init__()
        self.x1 = int(x1)
        self.x2 = int(x2)
        self.y1 = int(y1)
        self.y2 = int(y2)
    
    def __repr__(self):
        return f"x1 - {self.x1},\
                 y1 - {self.y1},\
                 x2 - {self.x2},\
                 y2 - {self.y2}"


def hex_to_rgb(h: str):
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb):
    return str.upper('%02x%02x%02x' % tuple(rgb))


hex_color_map = {
    "FFFDD0": "Кремовый",
    "000000": "Черный",
    "7B3F00": "Шоколадный",
    "BEBEBE": "Серый",
    "00BFFF": "Голубой",
    "91302B": "Рыжий",
    "FFFFFF": "Белый"
}


colors = np.array([hex_to_rgb(elem) for elem in hex_color_map.keys()])


def get_mask(img, rect: Rect):
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    r = (rect.x1, rect.y1, rect.x2 - rect.x1, rect.y2 - rect.y1) # format : x,y,w,h and only int numbers
    cv2.grabCut(img,mask,r,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    return mask2


def closest_colour(requested_colour):
    min_colours = {}
    for color in colors:
        r_c, g_c, b_c = color
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = color
    return hex_color_map[rgb_to_hex(min_colours[min(min_colours.keys())])]


def count_colors(img, mask):
    reduced_img = img*mask[:,:,np.newaxis]
    count = {}
    for p in reduced_img[mask!=0]:
        res = closest_colour(p)
        if not count.get(res):
            count[res] = 0
        count[res] += 1
    return count


def choose_color(colors_count: dict):
    res_colors = np.array(sorted(list(colors_count.items()), key=lambda item: item[-1]))
    top = res_colors[:, -1][-3:].astype(np.float32)
    norm = np.linalg.norm(top)
    if ((top / norm <= 0.64) & (top / norm >= 0.5)).all():
        return ["Многоцвет"]
    else:
        return res_colors[-2:, 0].tolist()


def pred_color(img: np.ndarray, boxes: np.ndarray):
    res = []
    for box in boxes:
        rect = Rect(*box[:4])
        mask = get_mask(img, rect)
        res.append(choose_color(count_colors(img, mask)))
    return res


if __name__ == "__main__":
    pass
