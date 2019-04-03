'''
Added by Minming Qian, for converting kiktech json data to pascal voc format
prepare the data from img and the json file
maybe no need to generate the xml file
but with xml, then it can take usage of the pascal voc
'''
import os
from glob import glob
from PIL import Image
import shutil
import json
import xml.etree.cElementTree as ET
import xml.dom.minidom
import pdb
import PIL.Image
import PIL.ImageDraw
import base64
import io
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def remove_ignore_region(image, ignores):
    # Get size
    width, height = image.size

    # Create new Image and a Pixel Map
    new = create_image(width, height)
    pixels = new.load()

    # Transform to black
    for i in range(width):
        for j in range(height):
            if pixel_in_ignore(i, j, ignores):
                pixels[i, j] = (0, 0, 0)
            else:
                pixels[i, j] = get_pixel(image, i, j)

    # Return new image
    return new

def pixel_in_ignore(x, y, ignores):
    for ignore in ignores:
        xmin = ignore[0]
        ymin = ignore[1]
        xmax = ignore[2]
        ymax = ignore[3]

        if x > xmin and x < xmax and y > ymin and y < ymax:
            return True

    return False


# Open an Image
def open_image(path):
  newImage = Image.open(path)
  return newImage

# Save Image
def save_image(image, path):
  image.save(path, 'png')

# Create a new image with the given size
def create_image(i, j):
  image = Image.new("RGB", (i, j), "white")
  return image

# Get the pixel from the given image
def get_pixel(image, i, j):
    # Inside image bounds?
    width, height = image.size
    if i > width or j > height:
      return None

    # Get Pixel
    pixel = image.getpixel((i, j))
    return pixel


def read_ignore_region(annotation_path):
    '''
    Get ignore regions
    :param annotation_path:
    :return:
    '''
    ignores = []
    with open(annotation_path) as json_data:
        d = json.load(json_data)
        for shape in d['shapes']:
            if shape['label'] == "ignore":

                points = shape["points"]

                if len(points) != 4:
                    continue

                [xmin, ymin] = points[0]
                [xmax, ymax] = points[0]
                for point in points:
                    [xmin, ymin] = [min(point[0], xmin), min(point[1], ymin)]
                    [xmax, ymax] = [max(point[0], xmax), max(point[1], ymax)]

                ignores.append([xmin, ymin, xmax, ymax])
    return ignores

def remove_ignores(image_file_path, json_file_path, output_file_path):
    image = open_image(image_file_path)
    ignores = read_ignore_region(json_file_path)
    output_image = remove_ignore_region(image, ignores)
    save_image(output_image, output_file_path)

def img_b64_to_arr(img_b64):
    f = io.BytesIO()
    f.write(base64.b64decode(img_b64))
    img_arr = np.array(Image.open(f))
    return img_arr


def img_arr_to_b64(img_arr):
    img_pil = Image.fromarray(img_arr)
    f = io.BytesIO()
    img_pil.save(f, format='PNG')
    img_bin = f.getvalue()
    img_b64 = base64.encodestring(img_bin)
    return img_b64

def polygons_to_mask(img_shape, polygons):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    xy = list(map(tuple, polygons))
    PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask

def shapes_to_label(img_shape, shapes, label_name_to_value, type='class'):
    assert type in ['class', 'instance']

    cls = np.zeros(img_shape[:2], dtype=np.int32)
    if type == 'instance':
        ins = np.zeros(img_shape[:2], dtype=np.int32)
        instance_names = ['_background_']
    for shape in shapes:
        polygons = shape['points']

        #Add by Minming: if no more than 3 points, just ignore this
        if len(polygons) < 3:
            continue
        #End add by Minming

        label = shape['label']
        if type == 'class':
            cls_name = label
        elif type == 'instance':
            cls_name = label.split('-')[0]
            if label not in instance_names:
                instance_names.append(label)
            ins_id = len(instance_names) - 1
        cls_id = label_name_to_value[cls_name]
        mask = polygons_to_mask(img_shape[:2], polygons)
        cls[mask] = cls_id
        if type == 'instance':
            ins[mask] = ins_id

    if type == 'instance':
        return cls, ins
    return cls

# label_list = ('drive_lane', 'forklift', 'goods_pallet',
#               'pallet', 'guardrail','pallet_group', 'person',
#               'pillar', "blue_pallet", "yellow_pallet", 'ignore')
# # the blue and yellow pallet are all pallet now,
# # this list for dataset without ignore

def clear_files(source_folder):
    '''remove (1), in some files there are files name with (1)'''
    img_result = [y for x in os.walk(source_folder) for y in glob(os.path.join(x[0], "*.png"))]
    print(img_result)
    for img in img_result:
        shutil.move(img, img.replace(" (1)",""))

    json_result = [y for x in os.walk(source_folder) for y in glob(os.path.join(x[0], "*.json"))]
    for jsonfile in json_result:
        shutil.move(jsonfile, jsonfile.replace(" (1)",""))

def copy_all_files(source_image_files, target_folder, ignore_black):

    # Added by Jie
    # label_list = ('pallet', 'pillar', 'person', 'goods_pallet', 'yellow_pallet', 'blue_pallet', \
    # 	'guardrail', 'pallet_group', 'forklift', 'ignore')
    label_name_to_value = {
        u'_background_': 0, 
        u'drive_line':1,
        u'drive_lane':1,
        u'drive_line_big':1,
        u'pallet':0, 
        u'pillar':0,
        u'person':0, 
        u'goods_pallet':0, 
        u'yellow_pallet':0, 
        u'blue_pallet':0,
        u'guardrail':0, 
        u'pallet_group':0, 
        u'forklift':0, 
        u'ignore':0
        }  # TODO
    # label_name_to_value = {
    #     u'_background_': 0, 
    #     u'drive_line':1, 
    #     u'pallet':2, 
    #     u'pillar':3,
    #     u'person':4, 
    #     u'goods_pallet':5, 
    #     u'yellow_pallet':6, 
    #     u'blue_pallet':7,
    #     u'guardrail':8, 
    #     u'pallet_group':9, 
    #     u'forklift':10, 
    #     u'ignore':255
    #     }

    #todo, instead using number, use the original filename
    for i, img in enumerate(source_image_files, 1):
        #Added by Minming Qian, if the json file not exist, then not copy the file
        source_json_file = img.replace(" (1)","").replace("jpg", "json")
        source_json_file = img.replace(" (1)","").replace("png", "json")

        if not os.path.exists(source_json_file):
            continue

        im = Image.open(img)
        #convert to jpg file
        img_rgb = im.convert('RGB')
        jpg_file_path = os.path.join(target_folder,'JPEGImages',
                                     os.path.basename(img).replace('png', 'jpg'))
        img_rgb.save(jpg_file_path)


        target_json_file = os.path.join(target_folder,
                                        'JSONAnnotations',
                                        os.path.basename(source_json_file))
        shutil.copyfile(source_json_file, target_json_file)

        # ------------ Added by Jie: Save segmantation GT
        try:
            json_data = json.load(open(source_json_file))
        except:
            print("load data failed, continue to next image")
            continue

        # print('source_json_file', source_json_file, type(json_data))
        js_im = img_b64_to_arr(json_data['imageData'])  # img_b64_to_arr  img_b64_to_array
        lbl = shapes_to_label(js_im.shape, json_data['shapes'], label_name_to_value, type='class')
        target_seg_mask = os.path.join(target_folder,
                                       'PNGSegmentationMask',
                                       os.path.basename(img).replace('jpg', 'png'))
        Image.fromarray(lbl).save(target_seg_mask)
        # -- End add

        # Added by minming
        if ignore_black:
            remove_ignores(jpg_file_path, target_json_file, jpg_file_path)


def convert_annotation(json_files):
    '''read json,
    convert to xml'''
    for file in json_files:
        with open(file) as json_data:
            try:
                from_json_to_xml(json_data, file)
            except:
                print("faild to convert")

def from_json_to_xml(json_data, file_name):
    annotation = ET.Element("annotation")
    folder = ET.SubElement(annotation, "folder").text = "kiktech2018"
    filename  = ET.SubElement(annotation, "filename").text = os.path.basename(file_name).replace("json", "jpg")


    d = json.load(json_data)
    for shape in d["shapes"]:
        #information from the shape
        label = shape["label"]

        # # Add by Jie
        # if label == "drive_line":
        # 	pass

        # End add
        if label == "blue_pallet" or label == "yellow_pallet":
            label = "pallet"
        if label not in label_list:
            continue
        points = shape["points"]

        #Add by Minming, here to remove those on in rectangle shepe, points not equal to 4
        if len(points) != 4:
            continue

        [xmin, ymin] = points[0]
        [xmax, ymax] = points[0]
        for point in points:
            [xmin, ymin] = [min(point[0], xmin), min(point[1], ymin)]
            [xmax, ymax] = [max(point[0], xmax), max(point[1], ymax)]

        #
        object = ET.SubElement(annotation, "object")
        name = ET.SubElement(object, "name").text = label
        difficult = ET.SubElement(object, "difficult").text = "0"
        bndbox = ET.SubElement(object, "bndbox")

        #Modify by Minming, the bounding box should always start with 1, and the maximum must small than width
        xminxml = ET.SubElement(bndbox, "xmin").text = str(xmin + 1)
        yminxml = ET.SubElement(bndbox, "ymin").text = str(ymin + 1)
        xmaxxml = ET.SubElement(bndbox, "xmax").text = str(xmax)
        ymaxxml = ET.SubElement(bndbox, "ymax").text = str(ymax)

        print(label)
        print(points)
        print([xmin, ymin], [xmax, ymax])
        print(file_name)
        assert xmax > xmin
        assert ymax > ymin

    #Add by Minming for one box
    object = ET.SubElement(annotation, "object")
    name = ET.SubElement(object, "name").text = "person"
    difficult = ET.SubElement(object, "difficult").text = "0"
    bndbox = ET.SubElement(object, "bndbox")

    # Modify by Minming, the bounding box should always start with 1, and the maximum must small than width
    xminxml = ET.SubElement(bndbox, "xmin").text = str(10)
    yminxml = ET.SubElement(bndbox, "ymin").text = str(10)
    xmaxxml = ET.SubElement(bndbox, "xmax").text = str(20)
    ymaxxml = ET.SubElement(bndbox, "ymax").text = str(20)

    tree = ET.ElementTree(annotation)
    xml_filepath = os.path.join(file_name.replace("JSONAnnotations","Annotations").replace("json","xml"))
    tree.write(xml_filepath)


    xml_content = xml.dom.minidom.parse(xml_filepath)  # or xml.dom.minidom.parseString(xml_string)
    pretty_xml_as_string = xml_content.toprettyxml()
    print(pretty_xml_as_string)
    with open(xml_filepath, "w") as f:
        f.write(pretty_xml_as_string)

#todo here to generate dataset for comparison purpose
def prepare_image_sets(target_folder, jpg_img_files):

    test_list = []
    train_list = []
    val_list = []
    for i, img_name in enumerate(jpg_img_files, 1):
        index = os.path.basename(img_name).replace(".jpg", "")
        # here to set the image range
        # if int(index) > 65:
        #     continue
        if not (i % 4):
            # shutil.copyfile(img_name, os.path.join("image", os.path.basename(img_name)))
            test_list.append(index)
        elif not (i + 1) % 4:
            val_list.append(index)
        else:
            train_list.append(index)

    train_file = os.path.join(target_folder, "ImageSets", "Main", "train.txt")
    val_file = os.path.join(target_folder, "ImageSets", "Main", "val.txt")
    test_file = os.path.join(target_folder, "ImageSets", "Main", "test.txt")
    trainval_file = os.path.join(target_folder, "ImageSets", "Main", "trainval.txt")

    trainval_list = train_list + val_list

    #todo write in each class
    with open(train_file, "w") as f:
        [f.write("{}\n".format(x)) for x in train_list]

    with open(val_file, 'w') as f:
        [f.write("{}\n".format(x)) for x in val_list]

    with open(test_file, 'w') as f:
        [f.write("{}\n".format(x)) for x in test_list]

    with open(trainval_file, 'w') as f:
        [f.write("{}\n".format(x)) for x in trainval_list]

#use glob get all files
def get_files(target_folder):
    img_files = [y for x in os.walk(os.path.join(target_folder, "JPEGImages")) for y in glob(os.path.join(x[0], "*.jpg"))]
    json_files = [x.replace("jpg", "json").replace("JPEGImages", "JSONAnnotations") for x in img_files]

    return img_files, json_files

def get_original_files(source_folder):
    img_files = [y for x in os.walk(source_folder) for y in glob(os.path.join(x[0], "*.jpg"))]
    json_files = [x.replace("jpg", "json") for x in img_files]

    png_files = [y for x in os.walk(source_folder) for y in glob(os.path.join(x[0], "*.png"))]
    img_files += png_files
    png_json_files = [x.replace("png", "json") for x in img_files]
    json_files += png_json_files

    return img_files, json_files



def making_directories(target_folder, version):

    kiktech_folder = os.path.join(target_folder, version)
    print('target_folder', target_folder)
    print('version', version)
    print('kiktech_folder', kiktech_folder)
    image_sets_folder = os.path.join(kiktech_folder, "ImageSets")
    json_file_folder = os.path.join(kiktech_folder, "JSONAnnotations")
    png_mask_folder = os.path.join(kiktech_folder, "PNGSegmentationMask")
    xml_file_folder = os.path.join(kiktech_folder, "Annotations")
    jpeg_file_folder = os.path.join(kiktech_folder, "JPEGImages")
    main_set_folder = os.path.join(image_sets_folder, "Main")

    folders = [target_folder, kiktech_folder, image_sets_folder, json_file_folder, png_mask_folder, 
               xml_file_folder, jpeg_file_folder, main_set_folder]

    print(folders)

    [os.mkdir(x) for x in folders if not os.path.exists(x)]


    return kiktech_folder

def generate(dataset_name,  source_folder, target_folder='./', ignore_black=False):
    clear_files(source_folder)
    kiktech_folder = making_directories(target_folder, dataset_name)
    img_files,_ = get_original_files(source_folder)

    copy_all_files(img_files, kiktech_folder, ignore_black)

    pdb.set_trace()
    jpg_img_files, json_files = get_files(kiktech_folder)
    prepare_image_sets(kiktech_folder, jpg_img_files)
    convert_annotation(json_files)

if __name__ == '__main__':
    # Modified by Minming, remove drive_lane for detection dataset
    import sys
    label_list = ('pallet', 'pillar', 'person', 'goods_pallet', 'yellow_pallet', 'blue_pallet', 'guardrail', 'pallet_group', 'forklift', 'ignore')

    raw_data_folder = sys.argv[1]
    dataset_name = sys.argv[2]
    if len(sys.argv) > 3:
        target_folder = sys.argv[3]
    else:
        target_folder = "./"

    #todo add paramaters for setting up different output foldername, with the date
    generate(dataset_name, raw_data_folder, target_folder, False)


