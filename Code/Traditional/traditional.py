import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

def license_plate(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_blur = cv2.GaussianBlur(hsv, [5, 5], 0)
    #cv2.imshow('hsv image', hsv)
    #cv2.imshow("imae hsv blur", hsv_blur)

    img_mask = cv2.inRange(hsv_blur, np.array([100, 115, 115]), np.array([124, 255, 255]))
    #cv2.imshow('image mask', img_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img_lcs = cv2.morphologyEx(img_mask, cv2.MORPH_OPEN, kernel, iterations = 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    img_lcs = cv2.morphologyEx(img_lcs, cv2.MORPH_CLOSE, kernel, iterations = 2)
    # cv2.imshow('lcs', img_lcs)

    contours, hierarchy = cv2.findContours(img_lcs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        img_mask = cv2.inRange(hsv_blur, np.array([35, 10, 160]), np.array([70, 100, 200]))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img_lcs = cv2.morphologyEx(img_mask, cv2.MORPH_OPEN, kernel, iterations = 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
        img_lcs = cv2.morphologyEx(img_lcs, cv2.MORPH_CLOSE, kernel, iterations = 1)
        contours, hierarchy = cv2.findContours(img_lcs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i in contours:
        x, y, w, h = cv2.boundingRect(i)
        if w > h * 1.2:
            lcs_oblique = img[y:y + h, x:x + w - 5]
            lcs_oblique_bin = img_lcs[y:y + h, x:x + w - 5]
            cv2.imshow('lcs_oblique', lcs_oblique)
            angle = cv2.minAreaRect(i)[2]
            break

    lcs_area = np.where(lcs_oblique_bin==255)
    x1, y1, x2, y2 = min(lcs_area[1]), min(lcs_area[0]), max(lcs_area[1]), max(lcs_area[0])
    dx = x2 - x1
    dy = lcs_area[0].shape[0] // dx
    
    if angle < 45:
        src_vertex = np.array([[x1, y1], [x2, y2 - dy], [x1, y1 + dy], [x2, y2]], dtype=np.float32)
        dst_vertex = np.array([[x1, y1], [x1 + int(1.5 * dx), y1], [x1, y1 + dy], [x1 + int(1.5 * dx), y1 + dy]], dtype=np.float32)
    elif angle < 70:
        src_vertex = np.array([[x1, y2 - dy + 25], [x1, y2], [x2, y1 + 20], [x2 - 5, y1 + dy - 30]], dtype=np.float32)
        dst_vertex = np.array([[x1, y1], [x1, y1 + dy], [x1 + int(1.5 * dx), y1], [x1 + int(1.5 * dx), y1 + dy]], dtype=np.float32)
    elif angle<=90:
        src_vertex = np.array([[x1, y2 - dy], [x1, y2], [x2, y1], [x2, y1 + dy]], dtype=np.float32)
        dst_vertex = np.array([[x1, y1], [x1, y1 + dy], [x1 + int(dx),y1], [x1 + int(dx),y1 + dy]], dtype=np.float32)
    else:
        src_vertex = np.array([[x1 + 20, y2 - dy + 27], [x1, y2], [x2, y1 + 20], [x2 - 25, y1 + dy - 20]], dtype=np.float32)
        dst_vertex = np.array([[x1, y1], [x1, y1 + dy], [x1 + int(1.5 * dx), y1], [x1 + int(1.5 * dx), y1 + dy]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_vertex, dst_vertex)
    lcs_orth = cv2.warpPerspective(lcs_oblique, M, (int(dx * 1.5), dy))
    #cv2.imshow('lcs', lcs_orth)
    
    return lcs_orth

def backgroud_checking(img):
    h, w = img.shape[0], img.shape[1]
    B, O = 0, 0
    for i in range(h):
        for j in range(w):
            B += img[i, j, 0]
            O += img[i, j, 1]
    
    if B > O:
        return True
    else:
        return False

def preprocess(img):
    img_bgr = cv2.resize(img, (int(200*img.shape[1]/img.shape[0]), 200))
    isBlue = backgroud_checking(img_bgr)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cv2.imshow('img_gray', img_gray)
    img_blur = cv2.GaussianBlur(img_gray, [5, 5], 5)
    cv2.imshow('img_blur', img_blur)

    if isBlue:
        ret, img_thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 9))
    else:
        ret, img_thresh = cv2.threshold(img_blur, 50, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        
    img_open = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 10))
    img_dilated = cv2.dilate(img_open, kernel, iterations=1)
    #cv2.imshow('img_dilated', img_open)
    #cv2.imshow("kernel", img_dilated)
    return img_open, img_dilated

def split_character(lcs_char, lcs_char_shape):
    contours, hierarchy = cv2.findContours(lcs_char_shape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    chars = []
    for i in contours:
        rect = cv2.boundingRect(i)
        chars.append(rect)

    chars = sorted(chars, key=lambda x:x[0], reverse=False)
    char_imgs = []
    for char in chars:
        if char[3] > char[2] * 1.5 and char[3] < char[2] * 2.2:
            splited_char = lcs_char[char[1]:char[1] + char[3], char[0] + 8:char[0] + char[2] - 8]
            char_imgs.append(splited_char)
    
    for i, char_img in enumerate(char_imgs):
        cv2.imshow('char {}'.format(i), char_img)

    return char_imgs

def template_matching(char_imgs):
    path = 'template_data/'
    nums = ['{}'.format(i) for i in range(10)]
    caps = [chr(i) for i in range(65, 91)]
    caps = caps[0:8] + caps[9:14] + caps[15:25]
    prvs = ['藏','川','鄂','甘','赣','贵','桂','黑','沪','吉','冀','津','晋','京','辽','鲁','蒙','闽','宁','青','琼','陕','苏','皖','湘','新','渝','豫','粤','云','浙']
    srns = nums + caps

    prv_temps = [glob.glob(path + prv + '/*') for prv in prvs]
    cap_temps = [glob.glob(path + cap + '/*') for cap in caps]
    srn_temps = [glob.glob(path + srn + '/*') for srn in srns]

    result = ''
    for i, char_img in enumerate(char_imgs):
        if i == 0:
            chars = prvs
            temps = prv_temps
        elif i == 1:
            chars = caps
            temps = cap_temps
        else:
            chars = srns
            temps = srn_temps
        
        scores = []
        for temp in temps:
            score = []
            for sample in temp:
                template = cv2.imdecode(np.fromfile(sample, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                template = cv2.resize(template, (char_img.shape[1], char_img.shape[0]))
                score.append(cv2.matchTemplate(char_img, template, cv2.TM_CCOEFF)[0][0])
            scores.append(max(score))

        index = np.argmax(np.array(scores))
        result = result + chars[index]
        if i == 1:
            continue

    return result

if __name__ == '__main__':
    img_bgr = cv2.imread('dataset/train_images/840.jpg')
    #cv2.imshow('img_bgr', img_bgr)
    lp = license_plate(img_bgr)
    char, char_shape = preprocess(lp)
    char_imgs = split_character(char, char_shape)
    print(template_matching(char_imgs))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

