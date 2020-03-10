import xml.dom
import xml.dom.minidom
import os 
import cv2 
import json 

"""
   生成XML文件
"""
_ANNOTATION_SAVE_PATH = '/mnt/data-1/data/chenxin.lu/chongqing_tianchi/PyTorch-YOLOv3/visualizer/ann1_24'
_INDENT = ' ' * 4
_NEW_LINE = '\n'
_FOLDER_NODE = 'Ann'
_ROOT_NODE = 'annotation'
_DATABASE_NAME = 'chongqing_tianchi'
_ANNOTATION = 'chongqing_tianchi'
_AUTHOR = 'lcxzzz'
_SEGMENTED = '0'
_DIFFICULT = '0'
_TRUNCATED = '0'
_POSE = 'Unspecified'

def createElementNode(doc, tag, attr): 
    element_node = doc.createElement(tag)
    text_node = doc.createTextNode(attr)
    element_node.appendChild(text_node)
    return element_node


def createChildNode(doc, tag, attr, parent_node):
    child_node = createElementNode(doc, tag, attr)
    parent_node.appendChild(child_node)


def createObjectNode(doc, ann):
    object_node = doc.createElement('object')
    createChildNode(doc, 'name', 'c' + str(ann[0]), object_node)
    createChildNode(doc, 'pose', _POSE, object_node)
    createChildNode(doc, 'truncated',  _TRUNCATED, object_node)
    createChildNode(doc, 'difficult', _DIFFICULT, object_node)
    bndbox_node = doc.createElement('bndbox')
    createChildNode(doc, 'xmin', str(ann[1][0]), bndbox_node)
    createChildNode(doc, 'ymin', str(ann[1][1]), bndbox_node)
    createChildNode(doc, 'xmax', str(ann[1][0] + ann[1][2]), bndbox_node)
    createChildNode(doc, 'ymax', str(ann[1][1] + ann[1][3]), bndbox_node)
    object_node.appendChild(bndbox_node)
    return object_node


# 将documentElement写入XML文件
def writeXMLFile(doc, filename):
    tmpfile = open('tmp.xml', 'w')
    doc.writexml(tmpfile, addindent=' ' * 4, newl='\n', encoding='utf-8')
    tmpfile.close()
    fin = open('tmp.xml')
    fout = open(filename, 'w')
    lines = fin.readlines()
    for line in lines[1:]:
        if line.split():
            fout.writelines(line)
    fin.close()
    fout.close()


if __name__ == "__main__":
    # 图片的路径
    img_path = "/mnt/data-1/data/chenxin.lu/chongqing_tianchi/PyTorch-YOLOv3/data/custom/images/trainval"
    ann_json_path = '/mnt/data-1/data/chenxin.lu/chongqing_tianchi/data/train_data/annotations_pro.json'
    fileList = os.listdir(img_path)
    if fileList == 0: 
        os._exit(-1)
    if not os.path.exists(_ANNOTATION_SAVE_PATH):
        os.mkdir(_ANNOTATION_SAVE_PATH)
    with open(ann_json_path, "r") as f:
        ann_json = json.load(f)
    for i, file_name in enumerate(fileList) :
        image_anns = []
        image_dict = [anns for anns in ann_json['images'] if (anns['file_name'] == file_name)][0]
        width, height, channel = image_dict['width'], image_dict['height'], 3
        image_id = image_dict['id']
        ann_dicts = [annotation for annotation in ann_json['annotations'] if (annotation['image_id'] == image_id)]
        for ann_dict in ann_dicts:
            # image_ann.append(ann_dict['id'])
            # image_ann.append(ann_dict['file_name'])
            image_ann = []
            image_ann.append(ann_dict['category_id'])
            image_ann.append(ann_dict['bbox'])
            image_anns.append(image_ann)
        saveName = file_name.strip(".jpg")
        xml_file_name = os.path.join(_ANNOTATION_SAVE_PATH, (saveName + '.xml'))
        my_dom = xml.dom.getDOMImplementation()
        doc = my_dom.createDocument(None, _ROOT_NODE, None)
        root_node = doc.documentElement
        createChildNode(doc, 'folder', _FOLDER_NODE, root_node)
        createChildNode(doc, 'filename', saveName + '.jpg', root_node)
        source_node = doc.createElement('source')
        createChildNode(doc, 'database', _DATABASE_NAME, source_node)
        createChildNode(doc, 'annotation', _ANNOTATION, source_node)
        createChildNode(doc, 'image', 'flickr', source_node)
        createChildNode(doc, 'flickrid', 'NULL', source_node)
        root_node.appendChild(source_node)
        owner_node = doc.createElement('owner')
        createChildNode(doc, 'flickrid', 'NULL', owner_node)
        createChildNode(doc, 'name', _AUTHOR, owner_node)
        root_node.appendChild(owner_node)
        size_node = doc.createElement('size')
        createChildNode(doc, 'width', str(width), size_node)
        createChildNode(doc, 'height', str(height), size_node)
        createChildNode(doc, 'depth', str(channel), size_node)
        root_node.appendChild(size_node)
        createChildNode(doc, 'segmented', _SEGMENTED, root_node)
        # print(image_anns)
        for i in range(len(image_anns)):
            object_node = createObjectNode(doc, image_anns[i])
            root_node.appendChild(object_node)
        writeXMLFile(doc, xml_file_name)