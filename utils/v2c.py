
import xml.etree.ElementTree as ET
import os
import json
import cv2
 
coco = dict()
coco['images'] = []
coco['type'] = 'instances'
coco['annotations'] = []
coco['categories'] = []
 
category_set = dict()
image_set = set()
 
category_item_id = -1
image_id = 0000000
annotation_id = 0
 
 
def addCatItem(name):
    global category_item_id
    category_item = dict()
    # category_item['supercategory'] = 'none'
    category_item_id += 1
    category_item['id'] = category_item_id
    category_item['name'] = name
    coco['categories'].append(category_item)
    category_set[name] = category_item_id
    return category_item_id
 
 
def addImgItem(file_name, size):
    global image_id
    if file_name is None:
        raise Exception('Could not find filename tag in xml file.')
    if size['width'] is None:
        raise Exception('Could not find width tag in xml file.')
    if size['height'] is None:
        raise Exception('Could not find height tag in xml file.')
    image_id += 1
    image_item = dict()
    image_item['id'] = image_id
    image_item['file_name'] = file_name
    image_item['width'] = size['width']
    image_item['height'] = size['height']
    coco['images'].append(image_item)
    image_set.add(file_name)
    return image_id
 
 
def addAnnoItem(object_name, image_id, category_id, bbox):
    global annotation_id
    annotation_item = dict()
 
    annotation_item['area'] = bbox[2] * bbox[3]
    annotation_item['iscrowd'] = 0
    annotation_item['ignore'] = 0
    annotation_item['image_id'] = image_id
    annotation_item['bbox'] = bbox
    annotation_item['category_id'] = category_id
    annotation_id += 1
    annotation_item['id'] = annotation_id
    coco['annotations'].append(annotation_item)
 
 
def parseXmlFiles(xml_path):
    for i in datasets:
        bndbox = dict()
        size = dict()
        current_image_id = None
        current_category_id = None
 
        xml_file = os.path.join(xml_path, i + '.xml')
        file_name = i + '.jpg'
        h, w, c = cv2.imread(os.path.join('/mnt/data-1/data/lcx/train/image', file_name)).shape
        size['width'] = w
        size['height'] = h
        size['depth'] = 3
        
 
        tree = ET.parse(xml_file)
        root = tree.getroot()
        if root.tag != 'annotation':
            raise Exception('pascal voc xml root element should be annotation, rather than {}'.format(root.tag))

        # elem is <folder>, <filename>, <size>, <object>
        for elem in root:
            current_parent = elem.tag
            current_sub = None
            object_name = None

 
            if file_name in category_set:
                raise Exception('file_name duplicated')
            elif current_image_id is None and file_name is not None: 
                if file_name not in image_set:
                    current_image_id = addImgItem(file_name, size)

                else:
                    raise Exception('duplicated image: {}'.format(file_name))

            # subelem is <width>, <height>, <depth>, <name>, <bndbox>
            for subelem in elem:
                bndbox['xmin'] = None
                bndbox['xmax'] = None
                bndbox['ymin'] = None
                bndbox['ymax'] = None

                current_sub = subelem.tag
                if current_parent == 'object' and subelem.tag == 'name':
                    object_name = subelem.text
                    if object_name not in category_set:
                        current_category_id = addCatItem(object_name)
                    else:
                        current_category_id = category_set[object_name]

                elif current_parent == 'size':
                    if size[subelem.tag] is not None:
                        raise Exception('xml structure broken at size tag.')
                    size[subelem.tag] = int(subelem.text)

                # option is <xmin>, <ymin>, <xmax>, <ymax>, when subelem is <bndbox>
                for option in subelem:
                    if current_sub == 'bndbox':
                        if bndbox[option.tag] is not None:
                            raise Exception('xml structure corrupted at bndbox tag.')
                        bndbox[option.tag] = int(option.text)

                # only after parse the <object> tag
                if bndbox['xmin'] is not None:
                    if object_name is None:
                        raise Exception('xml1 structure broken at bndbox tag')
                    if current_image_id is None:
                        raise Exception('xml2 structure broken at bndbox tag')
                    if current_category_id is None:
                        raise Exception('xml3 structure broken at bndbox tag')
                    bbox = []
                    bbox.append(bndbox['xmin'])
                    bbox.append(bndbox['ymin'])
                    bbox.append(bndbox['xmax'] - bndbox['xmin'])
                    bbox.append(bndbox['ymax'] - bndbox['ymin'])
                    addAnnoItem(object_name, current_image_id, current_category_id, bbox)
                    print(11111)

 
 
if __name__ == '__main__':

    train_data = '/mnt/data-1/data/lcx/train/train.txt'
    with open(train_data) as f:
        datasets = [str(line.strip()) for line in f.readlines()]  
    # print(datasets)        
    xml_path = '/mnt/data-1/data/lcx/train/box'                 
    json_file ='/mnt/data-1/data/lcx/CenterNet-master/data/Water/annotations/train.json'                                                               
    parseXmlFiles(xml_path)                                      
    json.dump(coco, open(json_file, 'w'))