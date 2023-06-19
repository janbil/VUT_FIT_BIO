import math

import cv2
import fingerprint_enhancer		# Load the library
import imageio
import numpy as np
from PIL import ImageDraw
from cv2.mat_wrapper import Mat
import matplotlib.pyplot as plt
import core_funcs
from optparse import OptionParser

parser = OptionParser(usage="%prog [options] sourceimage [MODE]")

parser.add_option("-i", dest="images", default=False, action="store_true",
        help="Show intermediate images.")

options, args = parser.parse_args()
if len(args) == 0 or len(args) > 2:
    parser.print_help()
    exit(1)

sourceimage = args[0]
if len(args) == 1:
    mode = "BERGDATA"
else:
    mode = args[1]

#parser.add_option("-d", "--dry-run", dest="dryrun", default=False, action="store_true",
#        help="Do not save the result.")
#
#parser.add_option("-b", "--no-binarization", dest="binarize", default=True, action="store_false",
#        help="Use this option to disable the final binarization step")

options, args = parser.parse_args()


def damage_fingerprint(img):
    row = 0
    for rows in img:
        col = 0
        for point in rows:
            if row < img.shape[0]*5/8 and row > img.shape[0]*3/8 and col < img.shape[1]*5/8 and col > img.shape[1]*3/8:
                img[row][col] = 255
            col += 1
        row += 1
    return img

def Pavlidis_algo(img):
    """
    Method implmenting thinning algorithm.
    :param img: Binarized image
    :return: Thinned image
    """
    Q = []
    row = 0
    for rows in img:
        col = 0
        for point in rows:
            if point != 0:
                Q.append((row, col))
            col+=1
        row+=1
    S = []
    while len(Q) != 0:
        contours, hierarchy = cv2.findContours(image=img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)
        Lq = []
        Mq = []
        traversed = []
        Bq = []
        Kq = []
        #contours = [x for x in contours if len(x)>25]

        for shape in contours:
            mod_cont = []
            mod_cont.clear()
            for array in shape:
                mod_cont.append((array[0][1], array[0][0]))
            Bq.append(mod_cont)
        for mod_cont in Bq:
            index = -1

            for row, col in mod_cont:
                index += 1
                if img[row + 1, col] == 0 and img[row, col - 1] == 0 and img[row - 1, col] == 0 and img[
                    row, col + 1] == 0:
                    Mq.append((row, col))
                    continue
                if (row, col) in traversed:
                    Mq.append((row, col))
                    continue
                if index == 0:
                    if ((row + 1, col) in mod_cont and mod_cont[index + 1] != (row + 1, col)) or (
                            (row, col + 1) in mod_cont and mod_cont[index + 1] != (row, col + 1)) or (
                            (row - 1, col) in mod_cont and mod_cont[index + 1] != (row - 1, col)) or (
                            (row, col - 1) in mod_cont and mod_cont[index + 1] != (row, col - 1)):
                        Mq.append((row, col))
                        continue
                elif index == len(mod_cont)-1:
                    if ((row + 1, col) in mod_cont and mod_cont[index - 1] != (row + 1, col)) or (
                            (row, col + 1) in mod_cont and mod_cont[index - 1] != (row, col + 1)) or (
                            (row - 1, col) in mod_cont and mod_cont[index - 1] != (row - 1, col)) or (
                            (row, col - 1) in mod_cont and mod_cont[index - 1] != (row, col - 1)):
                        Mq.append((row, col))
                        continue
                else:
                    if ((row + 1, col) in mod_cont and (
                            mod_cont[index - 1] != (row + 1, col) and mod_cont[index + 1] != (row + 1, col))):
                        Mq.append((row, col))
                        continue
                    if ((row, col + 1) in mod_cont and (
                            mod_cont[index - 1] != (row, col + 1) and mod_cont[index + 1] != (row, col + 1))):
                        Mq.append((row, col))
                        continue
                    if ((row - 1, col) in mod_cont and (
                            mod_cont[index - 1] != (row - 1, col) and mod_cont[index + 1] != (row - 1, col))):
                        Mq.append((row, col))
                        continue
                    if ((row, col - 1) in mod_cont and (
                            mod_cont[index - 1] != (row, col - 1) and mod_cont[index + 1] != (row, col - 1))):
                        Mq.append((row, col))
                        continue
                traversed.append((row, col))
                Lq.append((row, col))
                #index += 1
        for mod_cont in Bq:
            for row, col in mod_cont:
                if (row-1, col) in S:
                    Kq.append((row, col))
                elif (row, col-1) in S:
                    Kq.append((row, col))
                elif (row+1, col) in S:
                    Kq.append((row, col))
                elif (row, col+1) in S:
                    Kq.append((row, col))
        S = list(set(S) | set(Mq) | set(Kq))
        for mod_cont in Bq:
            for row, col in mod_cont:
                if (row, col) in Q:
                    Q.remove((row, col))
        row = 0
        sum = 0
        for rows in img:
            col = 0
            for point in rows:
                if(point != 0):

                    if (row, col) not in Q:
                        sum += 1
                        img[row][col] = 0
                col += 1
            row += 1
        #for row, col in Q:
        #    img[row][col] = 0

    for(row, col) in S:
        img[row][col] = 255
    #return img
    newarr = np.zeros(img.shape)
    newedges = np.zeros(img.shape)
    row = 0
    for rows in img:
        col = 0
        for point in rows:
            if point != 0 and row != 0 and col != 0 and col != img.shape[1]-1 and row != img.shape[0]-1:
                p1 = img[row + 1][col - 1]
                p2 = img[row][col - 1]
                p3 = img[row - 1][col - 1]
                p4 = img[row - 1][col]
                p5 = img[row - 1][col + 1]
                p6 = img[row][col + 1]
                p7 = img[row + 1][col + 1]
                p0 = img[row + 1][col]
                #if p2 == 0 or p0 == 0 or p4 == 0 or p6 == 0:
                #    suma = int(p2 == 0) + int(p0 == 0) + int(p4 == 0) + int(p6 == 0)
                #    if suma == 1:
                #        newedges[row][col] = 1
                if p2 != 0 or p0 != 0 or p4 != 0 or p6 != 0:
                    suma = int(p2 != 0) + int(p0 != 0) + int(p4 != 0) + int(p6 != 0)
                    if suma == 1:
                        if not(p1 != 0 and p0!=0 and p7!=0) and not(p3 != 0 and p4!=0 and p5 != 0) and not(p6 != 0 and p7!=0):
                            newedges[row][col] = 1
#
                if p1 != 0 or p2 != 0 or p3 != 0:
                    if p5 != 0 or p6 != 0 or p7 != 0:
                        if p0 == 0 and p4 == 0:
                            newedges[row][col] = 1
                if p1 != 0 or p0 != 0 or p7 != 0:
                    if p3 != 0 or p4 != 0 or p5 != 0:
                        if p2 == 0 and p6 == 0:
                            newedges[row][col] = 1
                if newedges[row + 1][col - 1]:
                    if p2 == 0 and p0 == 0:
                        if p3 != 0 or p4 != 0 or p5 != 0 or p6 != 0 or p7 != 0:
                            newedges[row][col] = 1
                if newedges[row - 1][col - 1] == 1:
                    if p2 == 0 and p4 == 0:
                        if p0 != 0 or p1 != 0 or p5 != 0 or p6 != 0 or p7 != 0:
                            newedges[row][col] = 1
                if newedges[row - 1][col + 1] == 1:
                    if p4 == 0 and p6 == 0:
                        if p0 != 0 or p1 != 0 or p2 != 0 or p3 != 0 or p7 != 0:
                            newedges[row][col] = 1
                if newedges[row + 1][col + 1]:
                    if p0 == 0 and p6 == 0:
                        if p1 != 0 or p2 != 0 or p3 != 0 or p4 != 0 or p5 != 0:
                            newedges[row][col] = 1
                if newedges[row][col] == 0:
                    newarr[row][col] = 255
                    img[row][col] = 0
#
            col += 1
        row += 1
#
    row = 0
    for rows in img:
        col = 0
        for point in rows:
            if point != 0 and row != 0 and col != 0 and col != img.shape[1]-1 and row != img.shape[0]-1:
                p1 = img[row + 1][col - 1]
                p2 = img[row][col - 1]
                p3 = img[row - 1][col - 1]
                p4 = img[row - 1][col]
                p5 = img[row - 1][col + 1]
                p6 = img[row][col + 1]
                p7 = img[row + 1][col + 1]
                p0 = img[row + 1][col]
                if p0 == 0 and p1 == 0 and p2 == 0 and p3 != 0 and p4 != 0 and p5 == 0 and p6 != 0 and p7 == 0:
                    img[row][col] = 0
                if p0 != 0 and p1 != 0 and p2 == 0 and p3 == 0 and p4 == 0 and p5 == 0 and p6 != 0 and p7 == 0:
                    img[row][col] = 0
                if p0 == 0 and p1 != 0 and p2 != 0 and p3 == 0 and p4 != 0 and p5 == 0 and p6 == 0 and p7 == 0:
                    img[row][col] = 0
                if p0 == 0 and p1 == 0 and p2 == 0 and p3 == 0 and p4 != 0 and p5 == 0 and p6 != 0 and p7 != 0:
                    img[row][col] = 0
            col+=1
        row+=1
    contours, hierarchy = cv2.findContours(image=img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)
    contours = [x for x in contours if len(x)<25]
    for shape in contours:
        for array in shape:
            img[array[0][1]][array[0][0]] = 0
    return img

def make_move(img, row, col, came_from, limit):
    if limit == 0:
        return True
    if came_from!=1:
        if img[row + 1][col] != 0:
            return make_move(img, row + 1, col, 1, limit - 1)
    if came_from != 2:
        if img[row + 1][col + 1] != 0:
            return make_move(img, row + 1, col + 1, 2, limit - 1)
    if came_from != 3:
        if img[row][col + 1] != 0:
            return make_move(img, row, col + 1, 3, limit - 1)
    if came_from != 4:
        if img[row - 1][col + 1] != 0:
            return make_move(img, row - 1, col + 1, 4, limit - 1)
    if came_from != 5:
        if img[row - 1][col] != 0:
            return make_move(img, row - 1, col, 5, limit - 1)
    if came_from != 6:
        if img[row - 1][col - 1] != 0:
            return make_move(img, row - 1, col - 1, 6, limit - 1)
    if came_from != 7:
        if img[row + 1][col - 1] != 0:
            return make_move(img, row + 1, col - 1, 7, limit - 1)
    if came_from != 8:
        if img[row][col - 1] != 0:
            return make_move(img, row, col - 1, 8, limit - 1)

    else:
        return False

def take_path (img, row, col):
    toret = True
    if img[row+1][col] != 0:
        toret = toret and make_move(img, row+1, col, 1, 9)
    if img[row+1][col+1] != 0:
        toret = toret and make_move(img, row+1, col+1, 2, 9)
    if img[row][col+1] != 0:
        toret = toret and make_move(img, row, col+1, 3, 9)
    if img[row-1][col+1] != 0:
        toret = toret and make_move(img, row-1, col+1, 4, 9)
    if img[row-1][col] != 0:
        toret = toret and make_move(img, row-1, col, 5, 9)
    if img[row-1][col-1] != 0:
        toret = toret and make_move(img, row-1, col-1, 6, 9)
    if img[row+1][col-1] != 0:
        toret = toret and make_move(img, row+1, col-1, 7, 9)
    if img[row][col-1] != 0:
        toret = toret and make_move(img, row, col-1, 8, 9)
    return toret

"""Method for detecting markants from fingerprint skeleton"""
def showMarkants(img, edges):
    num_ends = 0
    num_doubles = 0
    row = 0
    markants = np.zeros(img.shape)
    ends = np.zeros(img.shape)
    doubles = np.zeros(img.shape)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if img[row][col] != 0 and row > 25 and col >25 and col < img.shape[1]-25 and row < img.shape[0]-25:
                p7 = img[row + 1][col - 1]
                p8 = img[row][col - 1]
                p9 = img[row - 1][col - 1]
                p2 = img[row - 1][col]
                p3 = img[row - 1][col + 1]
                p4 = img[row][col + 1]
                p5 = img[row + 1][col + 1]
                p6 = img[row + 1][col]
                N = int(p2 != 0) + int(p3 != 0) + int(p4 != 0) + int(p5 != 0) + int(p6 != 0) + int(p7 != 0) + int(
                    p8 != 0) + int(p9 != 0)
                if(N == 0):
                    img[row][col] = 0
                if N == 1:
                    k = 0
                    false = False
                    for r in edges:
                        l = 0
                        for c in r:
                            if c == 255:
                                if abs(row - k) < 15 and abs(col - l) < 15:
                                    false = True
                                    break
                            l += 1
                        k += 1
                    if false:
                        continue
                    num_ends += 1
                    if p6 != 0:
                        ends[row][col] = 8
                    elif p7 != 0:
                        ends[row][col] = 1
                    elif p8 != 0:
                        ends[row][col] = 2
                    elif p9 != 0:
                        ends[row][col] = 3
                    elif p2 != 0:
                        ends[row][col] = 4
                    elif p3 != 0:
                        ends[row][col] = 5
                    elif p4 != 0:
                        ends[row][col] = 6
                    elif p5 != 0:
                        ends[row][col] = 7
                    for x in range(2):
                        for y in range(2):
                            if row + x < img.shape[0] and row-x >= 0 and col + y < img.shape[1] and col - y >=0:
                                markants[row + x][col + y] = 155
                                markants[row + x][col - y] = 155
                                markants[row - x][col + y] = 155
                                markants[row - x][col - y] = 155
                if N > 2:
                    if not take_path(img, row, col):
                        continue
                    num_doubles += 1
                    doubles[row][col] = 1

                    for x in range(2):
                        for y in range(2):
                            if row + x < img.shape[0] and row-x >= 0 and col + y < img.shape[1] and col - y >=0:
                                markants[row + x][col + y] = 50
                                markants[row + x][col - y] = 50
                                markants[row - x][col + y] = 50
                                markants[row - x][col - y] = 50
    return (num_ends, num_doubles)

"""Method detecting unrealistic singularities"""
def sing_norm(singularities, shape):
    num = 0
    same = []
    if len(singularities) > 1:
        for sing in singularities:
            row, col = sing[0]
            for sing2 in singularities:
                row2, col2 = sing2[0]
                if sing2 not in same:
                    if row != row2 or col != col2:
                        if abs(row - row2) < 70 and abs(col - col2) < 35 and sing[1] == "loop" and sing2[1] == "loop":
                            same.append(sing)
    for s in same:
        singularities.remove(s)
    if len(singularities) > 3 or len(singularities) == 0:
        print("No singularities or more then 3 singularities.")
        num+=1
    found = False
    for sing in singularities:
        if sing[1] == "loop":
            found = True
    if not found:
        print("No loop detected")
        num+=1
    for sing in singularities:
        row, col = sing[0]
        for sing2 in singularities:
            row2,col2 = sing2[0]
            if sing != sing2:
                if sing[1] == "loop" and sing2[1] == "delta":
                    if row >= row2:
                        print("Delta higher then loop.")
                        num+=1
                if sing[1] == "loop" and sing2[1] == "loop":
                    print("More then one loop")
                    num += 1
        if row < 0.3*shape[0]:
            print("Singularity very high, in:")
            num += 1
            print(str(100 - (row / (shape[0] / 100))) + "% of image")
        if row > 0.7*shape[0] and sing == "loop":
            print("Loop very low, in:")
            print(str(100 - (row / (shape[0] / 100))) + "% of image")
            num += 1
    return num


signum = lambda x: -1 if x < 0 else 1
cells = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

def get_angle(left, right):
    angle = left - right
    if abs(angle) > 180:
        angle = -1 * signum(angle) * (360 - abs(angle))
    return angle

def poincare_index_at(i, j, angles, tolerance):
    deg_angles = [math.degrees(angles[i - k][j - l]) % 180 for k, l in cells]
    index = 0
    for k in range(0, 8):
        if abs(get_angle(deg_angles[k], deg_angles[k + 1])) > 90:
            deg_angles[k + 1] += 180
        index += get_angle(deg_angles[k], deg_angles[k + 1])

    if 180 - tolerance <= index and index <= 180 + tolerance:
        return "delta"
    if -180 - tolerance <= index and index <= -180 + tolerance:
        return "loop"
    if 360 - tolerance <= index and index <= 360 + tolerance:
        return "whorl"
    return "none"

"""Method for detecting singularities"""
def calculate_singularities(im, angles, tolerance, W, edges):
    #result = im.convert("RGB")

    #draw = ImageDraw.Draw(im)

    colors = {"loop" : (150, 0, 0), "delta" : (0, 150, 0), "whorl": (0, 0, 150)}
    singularities = []
    for i in range(1, len(angles) - 1):
        for j in range(1, len(angles[i]) - 1):
            singularity = poincare_index_at(i, j, angles, tolerance)

            if singularity != "none":

                row = i
                col = j
                k = 0
                false = False
                for r in edges:
                    l = 0
                    for c in r:
                        if c == 255:
                            if (abs(i-k)<30 and abs(j-l)<30) or col>len(angles[i])-25 or col < 25 or row < 25 or row > len(angles)-25:
                                false = True
                                break
                        l += 1
                    k += 1
                if false:
                    #print("passed")
                    continue

                found = False
                for sing in singularities:
                    srow, scol = sing[0]
                    if (abs(srow - i) < 20 and abs(scol - j) < 20):
                        found = True
                if found:
                    #print("Already found prob")
                    #print(singularity)
                    break
                singularities.append(((row, col),singularity))
                print(str(i) + " , " + str(j) + " , " + singularity)
                for x in range(8):
                    for y in range(8):
                        if row + x < im.shape[0] and row - x >= 0 and col + y < im.shape[1] and col - y >= 0:

                            if singularity == "delta":
                                im[row + x][col + y] = 0
                                im[row + x][col - y] = 0
                                im[row - x][col + y] = 0
                                im[row - x][col - y] = 0
                            elif singularity == "loop" or "whorl":
                                im[row + x][col + y] = 255
                                im[row + x][col - y] = 255
                                im[row - x][col + y] = 255
                                im[row - x][col - y] = 255


                #draw.ellipse([(i * W, j * W), ((i + 1) * W, (j + 1) * W)], outline = colors[singularity])
    if options.images:
        cv2.imshow('enhanced_image', im);  # display the result
        cv2.waitKey(0)

    #del draw

    return singularities
#img = cv2.imread('./original/original/BERGDATA/1101_1_2_L2.bmp', 0)


if __name__ == '__main__':
    name = sourceimage
    #mode = 'BERGDATA'
    #mode = 'SAGEMMSO'
    #mode = 'SYNTHETIC'
    # image = imageio.v2.imread(name).astype("float64")
    img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
    image = np.asarray(img).astype("float64")
    if options.images:
        cv2.imshow('enhanced_image', img);  # display the result
        cv2.waitKey(0)

    #img = damage_fingerprint(img)
    #image = damage_fingerprint(image)
    kernel = np.ones((5, 5), np.uint8)
    if mode == 'BERGDATA':
        kernel = np.ones((5, 5), np.uint8)
    if mode == 'SAGEMMSO':
        kernel = np.ones((5, 5), np.uint8)
        r = 0
        for row in img:
            c = 0
            for col in row:
                img[r][c] = (255 - col)
                c += 1
            r += 1
    if mode == 'SYNTHETIC':
        kernel = np.ones((7, 7), np.uint8)
        r = 0
        for row in img:
            c = 0
            for col in row:
                img[r][c] = (255 - col)
                c += 1
            r += 1

    img_dilation = cv2.dilate(img, kernel, iterations=5)
    if options.images:
        cv2.imshow('enhanced_image', img_dilation);  # display the result
        cv2.waitKey(0)
    if mode == 'BERGDATA':
        img_dilation = np.where(img_dilation > 200, img_dilation, 0)
    if mode == 'SAGEMMSO':
        img_dilation = np.where(img_dilation > 150, img_dilation, 0)
    if mode == 'SYNTHETIC':
        img_dilation = np.where(img_dilation > 200, img_dilation, 0)
    r = 0
    for row in img_dilation:
        c = 0
        for col in row:
            if col > 0:
                img_dilation[r][c] = 255
            c += 1
        r += 1
    if options.images:
        cv2.imshow('enhanced_image', img_dilation);  # display the result
        cv2.waitKey(0)
    edges = cv2.Canny(img_dilation, 10, 254)
    if options.images:
        cv2.imshow('enhanced_image', edges);  # display the result
        cv2.waitKey(0)
    image = core_funcs.normalize(image)  # read input image
    mask = core_funcs.findMask(image)
    image = np.where(mask == 1.0, core_funcs.localNormalize(image), image)
    orientations = np.where(mask == 1.0, core_funcs.estimateOrientations(image), -1.0)
    core_funcs.showOrientations(image, orientations, "orientations", 8)
    if options.images:
        plt.show()
    singularities = calculate_singularities(image, orientations, 6, 16, edges)
    num = sing_norm(singularities, image.shape)
    print("Starting thinning")
    out = fingerprint_enhancer.enhance_Fingerprint(img)
    out = Pavlidis_algo(out)
    if options.images:
        cv2.imshow('enhanced_image', out);  # display the result
        cv2.waitKey(0)
    endings, doubles = showMarkants(out, edges)
    if endings + doubles < 50:
        print("Less then 50 markants.")
        num +=2
    if endings +doubles > 150:
        print("More then 150 markants.")
        num += 2
    est = 100 - num * 15
    if est < 0:
        est = 0
    print("Total estimation of realisticity of fingerprint: " + str(est) + "%")