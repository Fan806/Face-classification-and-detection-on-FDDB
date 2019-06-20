import os
import math
from PIL import Image

fold_Number = ['01','02','03','04','05','06','07','08','09','10']
prefixFDDB = 'FDDB-folds/FDDB-fold-'
prefixPics = 'originalPics/'
suffix = '-ellipseList.txt'

def CropPaste(left, upper, right, lower, img):
    width, height = img.size
    out = img.crop((left,upper,right,lower))

    new_w, new_h = out.size
    wleft = hupper = wright = hlower = 0

    # if the face is on the boader of the image
    if left<0:
        wleft = -left
    if upper<0:
        hupper = -upper
    if right>width:
        wright = right-width
    if lower>height:
        hlower = lower-height

    for i in range(new_h):
        pix_left = out.getpixel((wleft,i))
        pix_right = out.getpixel((new_w-wright-1,i))
        if wleft!=0:
            for j in range(wleft):
                out.putpixel((j,i),pix_left)
        if wright!=0:
            for j in range(wright):
                out.putpixel((new_w-wright+j,i),pix_right)

    for i in range(new_w):
        pix_upper = out.getpixel((i,hupper))
        pix_lower = out.getpixel((i,new_h-hlower-1))
        if hupper!=0:
            for j in range(hupper):
                out.putpixel((i,j),pix_upper)
        if hlower!=0:
            for j in range(hlower):
                out.putpixel((i,new_h-hlower+j),pix_lower)

    return out

def generateNeg8(num,bbox_w,bbox_h,left,upper,right,lower,img,path,j):
    if num=='01' or num=='02' or num=='03' or num=='04' or num=='10':
        stride_w = (1/3)*bbox_w
        stride_h = (1/3)*bbox_h

        # left 1/3
        left1 = int(left-stride_w)
        right1 = int(right-stride_w)
        out_neg2 = CropPaste(left1, upper, right1, lower, img)
        out_neg2 = out_neg2.resize((96,96))
        out_neg2.save(path+'%d2.jpg'%j)

        # right 1/3
        left2 = int(left+stride_w)
        right2 = int(right+stride_w)
        out_neg3 = CropPaste(left2, upper, right2, lower, img)
        out_neg3 = out_neg3.resize((96,96))
        out_neg3.save(path+'%d3.jpg'%j)

        # up 1/3
        upper3 = int(upper-stride_h)
        lower3 = int(lower-stride_h)
        out_neg4 = CropPaste(left, upper3, right, lower3, img)
        out_neg4 = out_neg4.resize((96,96))
        out_neg4.save(path+'%d4.jpg'%j)

        # low 1/3
        upper4 = int(upper+stride_h)
        lower4 = int(lower+stride_h)
        out_neg5 = CropPaste(left, upper4, right, lower4, img)
        out_neg5 = out_neg5.resize((96,96))
        out_neg5.save(path+'%d5.jpg'%j)

        # left up
        out_neg6 = CropPaste(left1, upper3, right1, lower3, img)
        out_neg6 = out_neg6.resize((96,96))
        out_neg6.save(path+'%d6.jpg'%j)

        # right up
        out_neg7 = CropPaste(left2, upper3, right2, lower3, img)
        out_neg7 = out_neg7.resize((96,96))
        out_neg7.save(path+'%d7.jpg'%j)

        # left low
        out_neg8 = CropPaste(left1, upper4, right1, lower4, img)
        out_neg8 = out_neg8.resize((96,96))
        out_neg8.save(path+'%d8.jpg'%j)

        # right low
        out_neg9 = CropPaste(left2, upper4, right2, lower4, img)
        out_neg9 = out_neg9.resize((96,96))
        out_neg9.save(path+'%d9.jpg'%j)


for num in fold_Number:
    f = open(prefixFDDB+num+suffix, 'r')
    annotations = f.readlines()
    f.close()

    # the ith line represents the graph path
    i=0
    while(i<len(annotations)):
        path = annotations[i].rstrip('\n')
        face_Num = int(annotations[i+1])
        face_elipses = annotations[i+2:i+face_Num+2]
        i += (face_Num+2)
        print("generate samples from picture path: ",path)

        # generate samples
        class_path = 'classify/'
        detection_path = 'detection/'

        sample_path = '_'.join(path.split('/'))+'/'
        positive_path = 'samples/positive/'
        negative_path = 'samples/negative/'
        pos_path = positive_path+sample_path
        neg_path = negative_path+sample_path

        # generate sample
        with Image.open(prefixPics+path+'.jpg') as img:
            detection_folder = detection_path+sample_path
            class_folder = class_path+'_'.join(path.split('/'))

            # generate test sample
            if num=='09' or num=='10':
                if not os.path.exists(detection_folder):
                    os.makedirs(detection_folder)
                with open(detection_folder+"ground_truth.txt", 'w') as f:
                    f.write(str(face_Num))
                width, height = img.size
                bound = min(width, height)
                if bound<32:
                    tmpimg = img.resize((96, 96))
                    size = 96
                    size_folder = os.path.join(detection_folder, str(size))
                    if not os.path.exists(size_folder):
                        os.mkdir(size_folder)
                    tmpimg.save(size_folder)
                else:
                    # 从32开始取bounding box的大小，bounding box始终为正方形
                    stride = int(32/3)
                    for size in range(32, bound+1, stride):
                        if width<size or height<size:
                            tmpimg = img.resize((96,96))
                        else:
                            size_folder = os.path.join(detection_folder, str(size))+'/'
                            if not os.path.exists(size_folder):
                                os.mkdir(size_folder)
                            stride = int(size/3)
                            for left in range(0, width-size+1, stride):
                                for upper in range(0, height-size+1, stride):
                                    bbox = (left, upper, left+size-1, upper+size-1)
                                    tmpimg = img.crop(bbox).resize((96,96))
                                    tmpimg.save(size_folder+str(bbox)+'.jpg')

            # negative samples
            # 1. resize original images into 96x96 pixels
            if num=='01' or num=='02' or num=='03' or num=='04':
                if not os.path.exists(negative_path):
                    os.makedirs(negative_path)
                if not os.path.exists(neg_path):
                    os.makedirs(neg_path)
                out_neg1 = img.resize((96,96))
                out_neg1.save(neg_path+'1.jpg')

            if num=='10':
                neg_test1 = img.resize((96,96))
                neg_test1.save(class_folder+"1.jpg")

            for j in range(len(face_elipses)):
                line_split = face_elipses[j].split()

                # obtain info from current elipse
                major_axis_radius = float(line_split[0])
                minor_axis_radius = float(line_split[1])
                angle = float(line_split[2])
                center_x = float(line_split[3])
                center_y = float(line_split[4])

                # calculate bounding box of rotated elipse
                calc_x = math.sqrt(major_axis_radius**2 * math.cos(angle)**2 + minor_axis_radius**2 * math.sin(angle)**2)
                calc_y = math.sqrt(major_axis_radius**2 * math.sin(angle)**2 + minor_axis_radius**2 * math.cos(angle)**2)

                # positive sample
                label = 1

                # bounding box
                bbox_w = (calc_x*2)*(1+1/3)
                bbox_h = (calc_y*2)*(1+1/3)

                left = int(center_x-bbox_w/2)
                upper = int(center_y-bbox_h/2)
                right = int(left+bbox_w)
                lower = int(upper+bbox_h)

                left1 = int(center_x - calc_x)
                upper1 = int(center_y - calc_y)
                right1 = int(left + calc_x*2)
                lower1 = int(upper + calc_y*2)

                if num=='10':
                    generateNeg8(num,calc_x,calc_y,left1,upper1,right1,lower1,img,class_folder,j)
                    continue

                # positive samples
                if num!='10' and num!='09':
                    if not os.path.exists(positive_path):
                        os.makedirs(positive_path)
                    if not os.path.exists(pos_path):
                        os.makedirs(positive_path+sample_path)
                    out_pos = CropPaste(left, upper, right, lower, img)
                    out_pos = out_pos.resize((96,96))
                    out_pos.save(pos_path+'%d.jpg'%j)

                # # negative sample
                # # 2. generate eight negative images based on each face by sliding the bounding box by 1/3
                generateNeg8(num,calc_x,calc_y,left1,upper1,right1,lower1,img,neg_path,j)


