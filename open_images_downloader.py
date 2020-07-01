import time
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import botocore
import logging
from multiprocessing import Pool, Manager
import pandas as pd
import os
import argparse
import sys
import functools
from urllib import request
from random import sample 

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))


def parse_args():
    parser = argparse.ArgumentParser(description='Dowload open image dataset by class.')

    parser.add_argument("--root", type=str, default="data", help="The root directory that you want to store the image data.")
    parser.add_argument("include_depiction", action="store_true", help="Do you want to include drawings or depictions?")
    parser.add_argument("--class_names", type=str, help="the classes you want to download.")
    parser.add_argument("--num_workers", type=int, default=10, help="the number of worker threads used to download images.")
    parser.add_argument("--retry", type=int, default=10, help="retry times when downloading.")
    parser.add_argument("--filter_file", type=str, default="", help="This file specifies the image ids you want to exclude.")
    parser.add_argument('--remove_overlapped', action='store_true', help="Remove single boxes covered by group boxes.")
    parser.add_argument('--max-boxes-train', type=int, default=-1, help='limit the total number of training bounding boxes to the specified number (default is to use all)')
    parser.add_argument('--max-boxes-test', type=int, default=-1, help='limit the total number of test bounding boxes to the specified number (default is to use all)')
    parser.add_argument('--max-boxes-val', type=int, default=-1, help='limit the total number of validation bounding boxes to the specified number (default is to use all)')
    parser.add_argument('--stats-only', action='store_true', help='only list the number of images from the selected classes, and quit')
    
    return parser.parse_args()


def download(bucket, root, retry, counter, lock, path):
    i = 0
    src = path
    dest = f"{root}/{path}"
    while i < retry:
        try:
            if not os.path.exists(dest):
                s3.download_file(bucket, src, dest)
            else:
                logging.info(f"{dest} already exists.")
            with lock:
                counter.value += 1
                if counter.value % 100 == 0:
                    logging.warning(f"Downloaded {counter.value} images.")
            return
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                logging.warning(f"The file s3://{bucket}/{src} does not exist.")
                return
            i += 1
            logging.warning(f"Sleep {i} and try again.")
            time.sleep(i)
    logging.warning(f"Failed to download the file s3://{bucket}/{src}. Exception: {e}")


def batch_download(bucket, file_paths, root, num_workers=10, retry=10):
    with Pool(num_workers) as p:
        m = Manager()
        counter = m.Value('i', 0)
        lock = m.Lock()
        download_ = functools.partial(download, bucket, root, retry, counter, lock)
        p.map(download_, file_paths)


def http_download(url, path):
    with request.urlopen(url) as f:
        with open(path, "wb") as fout:
            buf = f.read(1024)
            while buf:
                fout.write(buf)
                buf = f.read(1024)


def log_counts(values):
    for k, count in values.value_counts().iteritems():
        print("    {:s}:  {:d}/{:d} = {:.2f}".format(k, count, len(values), count/len(values)))


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.WARNING,
                        format='%(asctime)s - %(name)s - %(message)s')

    args = parse_args()
    bucket = "open-images-dataset"
    
    # split the --class_names argument into an array
    names = [e.strip() for e in args.class_names.split(",")]
    class_names = []
    group_filters = []
    percentages = []
    for name in names:
        t = name.split(":")
        class_names.append(t[0].strip())
        if len(t) >= 2 and t[1].strip():
            group_filters.append(t[1].strip())
        else:
            group_filters.append("")
        if len(t) >= 3 and t[2].strip():
            percentages.append(float(t[2].strip()))
        else:
            percentages.append(1.0)

    num_classes = len(class_names)
    
    # make sure the output dir exists
    if not os.path.exists(args.root):
        os.makedirs(args.root)

    # apply optional filters to exclude unwanted images
    excluded_images = set()
    if args.filter_file:
        for line in open(args.filter_file):
            img_id = line.strip()
            if not img_id:
                continue
            excluded_images.add(img_id)

    # download the class description list
    class_description_file = os.path.join(args.root, "class-descriptions-boxable.csv")
    if not os.path.isfile(class_description_file):
        url = "https://storage.googleapis.com/openimages/2018_04/class-descriptions-boxable.csv"
        logging.warning(f"Download {url}.")
        http_download(url, class_description_file)

    # load the class descriptions and filter by the requested classes
    class_descriptions = pd.read_csv(class_description_file, names=["id", "ClassName"])
    class_descriptions = class_descriptions[class_descriptions['ClassName'].isin(class_names)]

    # verify that all the requested classes were found
    print("requested {:d} classes, found {:d} classes".format(num_classes, len(class_descriptions)))
    
    if num_classes != len(class_descriptions):
        missing_classes = []
        
        for class_name in class_names:
            if len(class_descriptions[class_descriptions['ClassName']==class_name]) == 0:
                print("couldn't find class '{:s}'".format(class_name))
                missing_classes.append(class_name)
                
        raise Exception("couldn't find classes '{:s}' in the Open Images dataset.  Please confirm that the --class_names argument contains valid classes.".format(','.join(missing_classes)))
        
    # determine per-class image file limits
    image_files = []
    annotations_max = {}
    annotations_count = {}
    
    annotations_max["train"] = int(args.max_boxes_train / num_classes)
    annotations_max["validation"] = int(args.max_boxes_val / num_classes)
    annotations_max["test"] = int(args.max_boxes_test / num_classes)

    # build the image list to download
    for dataset_type in ["train", "validation", "test"]:
        annotations_count[dataset_type] = 0
        print("\n-------------------------------------\n'{:s}' set classes\n-------------------------------------".format(dataset_type))
    
        # create subdirectory for this set
        image_dir = os.path.join(args.root, dataset_type)
        os.makedirs(image_dir, exist_ok=True)

        # download the annotations for train/val/test
        annotation_file = f"{args.root}/{dataset_type}-annotations-bbox.csv"
        if not os.path.exists(annotation_file):
            url = f"https://storage.googleapis.com/openimages/2018_04/{dataset_type}/{dataset_type}-annotations-bbox.csv"
            logging.warning(f"Download {url}.")
            http_download(url, annotation_file)
        logging.warning(f"Read annotation file {annotation_file}")
        annotations = pd.read_csv(annotation_file)
        annotations = pd.merge(annotations, class_descriptions,
                               left_on="LabelName", right_on="id",
                               how="inner")
        if not args.include_depiction:
            annotations = annotations.loc[annotations['IsDepiction'] != 1, :]

        print(' ')

        # determine any images to filter from each class
        filtered = []
        
        for class_name, group_filter, percentage in zip(class_names, group_filters, percentages):
            sub = annotations.loc[annotations['ClassName'] == class_name, :]
            excluded_images |= set(sub['ImageID'].sample(frac=1 - percentage))

            if group_filter == '~group':
                excluded_images |= set(sub.loc[sub['IsGroupOf'] == 1, 'ImageID'])
            elif group_filter == 'group':
                excluded_images |= set(sub.loc[sub['IsGroupOf'] == 0, 'ImageID'])

            num_annotations = len(sub)

            print('dataset_type ' + str(dataset_type))
            print('class_name   ' + str(class_name))
            print('group_filter ' + str(group_filter))
            print('percentage   ' + str(percentage))
            print('max boxes    ' + str(annotations_max[dataset_type]))
            print('total boxes  ' + str(len(sub)))

            if annotations_max[dataset_type] > 0 and len(sub) > annotations_max[dataset_type]:
                exclude_count = len(sub) - annotations_max[dataset_type]
                excluded_images |= set(list(sub['ImageID'])[:exclude_count]) #set(sample(set(sub['ImageID']), exclude_count)) #.sample(n=exclude_count))
                num_annotations = num_annotations - exclude_count
                print('num excluded ' + str(exclude_count))

            print('used boxes   ' + str(num_annotations))
            print(' ')

            annotations_count[dataset_type] += num_annotations
            filtered.append(sub)


        # get the list of bounding boxes for our classes
        annotations = pd.concat(filtered)
        annotations = annotations.loc[~annotations['ImageID'].isin(excluded_images), :]

        # remove single boxes covered by group boxes
        if args.remove_overlapped:
            images_with_group = annotations.loc[annotations['IsGroupOf'] == 1, 'ImageID']
            annotations = annotations.loc[~(annotations['ImageID'].isin(set(images_with_group)) & (annotations['IsGroupOf'] == 0)), :]
        
        # shuffle the ordering of the dataset
        annotations = annotations.sample(frac=1.0)

        print("-------------------------------------\n '{:s}' set statistics\n-------------------------------------".format(dataset_type))
        print("  Bounding boxes count: {:d}".format(annotations.shape[0]))
        print("  Bounding box distribution: ")
        log_counts(annotations['ClassName'])
        print("  Approximate image stats: ")
        log_counts(annotations.drop_duplicates(["ImageID", "ClassName"])["ClassName"])
        print(" ")
        
        #logging.warning(f"Shuffle dataset.")

        # save our selected annotations to file
        sub_annotation_file = f"{args.root}/sub-{dataset_type}-annotations-bbox.csv"
        logging.warning(f"Saving '{dataset_type}' data to {sub_annotation_file}.")
        annotations.to_csv(sub_annotation_file, index=False)
        image_files.extend(f"{dataset_type}/{id}.jpg" for id in set(annotations['ImageID']))
        
    # display the bounding box counts for each dataset
    print('\n-------------------------------------\n overall bounding box counts\n-------------------------------------')

    for dataset_type in ["train", "validation", "test"]:
        print("  {:<15s} {:d}".format("{:s} set:".format(dataset_type), annotations_count[dataset_type]))

    print(' ')
    print('total bounding boxes: {:d}'.format(sum(annotations_count.values())))
    print('total images:         {:d}'.format(len(image_files)))
    print(' ')
    
    # download the images
    if not args.stats_only:    
        logging.warning(f"Starting to download {len(image_files)} images.")
        batch_download(bucket, image_files, args.root, args.num_workers, args.retry)
        logging.warning("Task Done.")
