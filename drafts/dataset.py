from lxml import etree
from google.protobuf import text_format
from models.research.object_detection.protos import string_int_label_map_pb2
from models.research.object_detection.utils import dataset_util, label_map_util
from PIL import Image
import glob, hashlib, io, os
import tensorflow as tf


def load_pascal_voc(origin_dir):
    data = {}
    cwd = os.getcwd()
    try:
        os.chdir(origin_dir)
    except OSError:
        print('Could not change directory.')

    for file in glob.glob("*.xml"):
        with tf.io.gfile.GFile(file, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data[file] = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
    try:
        os.chdir(cwd)
    except OSError:
        print('Could not return to previous directory.')
    return data


def get_classes(label_map_path):
    label_map_dict = label_map_util.get_label_map_dict(label_map_path)
    return list(label_map_dict.keys())


def get_label_map_from_pascal_voc(label_map_path, data):
    if not os.path.exists(label_map_path) or os.stat(label_map_path).st_size == 0:
        labels = []
        for image in data:
            if 'object' in data[image]:
                for obj in data[image]['object']:
                    if obj['name'] not in labels:
                        labels.append(obj['name'])
        label_map = string_int_label_map_pb2.StringIntLabelMap()
        labels.sort()
        for i in range(len(labels)):
            label_map.item.add(name=labels[i], id=i+1)
        label_map_util._validate_label_map(label_map)
        label_map_string = text_format.MessageToString(label_map)
        with tf.io.gfile.GFile(label_map_path, 'wb') as fid:
            fid.write(label_map_string)
        print('Label map saved as: ', label_map_path)
    label_map_dict = label_map_util.get_label_map_dict(label_map_path)
    return label_map_dict


def dict_to_tf_features(data, img_path, label_map_dict, ignore_difficult_instances=False):
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    examples = []

    for image in data:
        if 'object' in data[image]:
            for obj in data[image]['object']:
                difficult = bool(int(obj['difficult']))
                if ignore_difficult_instances and difficult:
                    continue
                difficult_obj.append(int(difficult))

                filename = data[image]['filename']
                img_full_path = os.path.join(img_path, filename)
                with tf.io.gfile.GFile(img_full_path, 'rb') as fid:
                    encoded_jpg = fid.read()
                encoded_jpg_io = io.BytesIO(encoded_jpg)
                img = Image.open(encoded_jpg_io)
                if img.format != 'JPEG':
                    raise ValueError('Image format not JPEG')
                key = hashlib.sha256(encoded_jpg).hexdigest()

                width = int(data[image]['size']['width'])
                height = int(data[image]['size']['height'])
                xmin.append(float(obj['bndbox']['xmin']) / width)
                ymin.append(float(obj['bndbox']['ymin']) / height)
                xmax.append(float(obj['bndbox']['xmax']) / width)
                ymax.append(float(obj['bndbox']['ymax']) / height)
                classes_text.append(obj['name'].encode('utf8'))
                classes.append(label_map_dict[obj['name']])
                truncated.append(int(obj['truncated']))
                poses.append(obj['pose'].encode('utf8'))
            examples.append(tf.train.Example(features=tf.train.Features(feature={
                'image/height': dataset_util.int64_feature(height),
                'image/width': dataset_util.int64_feature(width),
                'image/filename': dataset_util.bytes_feature(
                  filename.encode('utf8')),
                'image/source_id': dataset_util.bytes_feature(
                  filename.encode('utf8')),
                'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
                'image/encoded': dataset_util.bytes_feature(encoded_jpg),
                'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
                'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
                'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
                'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
                'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
                'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
                'image/object/class/label': dataset_util.int64_list_feature(classes),
                'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
                'image/object/truncated': dataset_util.int64_list_feature(truncated),
                'image/object/view': dataset_util.bytes_list_feature(poses),
            })))

    print('Pascal VOC data loaded.')
    return examples


def pascal_voc_to_tfrecord(origin_dir, output_path, label_map_path):
    data = load_pascal_voc(origin_dir)
    label_map_dict = get_label_map_from_pascal_voc(label_map_path, data)
    messages = dict_to_tf_features(data, origin_dir, label_map_dict)
    writer = tf.io.TFRecordWriter(output_path)
    for message in messages:
        writer.write(message.SerializeToString())
    writer.close()
    print('TFRecord saved as: ', output_path)

