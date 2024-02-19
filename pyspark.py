# -*- coding: utf-8 -*-
"""Pyspark.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15asQAXKX0UXQul9NEueE4mg5ph3FmFtr
"""

from pyspark.sql import SparkSession
import cv2
import numpy as np
from pyspark.sql.functions import udf
from pyspark.sql.types import BinaryType, StringType, StructType, StructField
import os
import pandas as pd

# SparkSession
spark = SparkSession.builder \
    .appName("ResizeImages") \
    .getOrCreate()

# Input Path
input_directory = '/dbfs/mnt/projectteam5/New_Dataset/Original'

# Output path
output_directory = '/dbfs/mnt/projectteam5/Pyspark_output/'

# resize function
def resize_image(image_path, width=256, height=256):
    # Read the image using OpenCV
    img = cv2.imread(image_path)
    # Resize the image
    resized_img = cv2.resize(img, (width, height))
    # Encode the resized image back into binary format
    retval, buffer = cv2.imencode('.jpg', resized_img)
    resized_image_data = buffer.tobytes()
    return resized_image_data

# User Defiened function
resize_image_udf = udf(resize_image, BinaryType())

image_paths = []
for label in os.listdir(input_directory):
    label_dir = os.path.join(input_directory, label)
    if os.path.isdir(label_dir):
        for file in os.listdir(label_dir):
            image_paths.append(os.path.join(label_dir, file))

# data frame schema
schema = StructType([StructField("image_path", StringType(), nullable=True)])

# data frame with image path
image_paths_df = spark.createDataFrame([(path,) for path in image_paths], schema)

# resize function to dataframe
resized_images_df = image_paths_df.withColumn("resized_image", resize_image_udf(image_paths_df["image_path"]))

# conversion to pandas dataframe
pandas_df = resized_images_df.toPandas()

# save image
for index, row in pandas_df.iterrows():
    image_data = np.frombuffer(row['resized_image'], dtype=np.uint8)
    img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    label = os.path.basename(os.path.dirname(row['image_path']))
    cv2.imwrite(os.path.join(output_directory, f"{label}_resized_image_{index}.jpg"), img)

spark.stop()