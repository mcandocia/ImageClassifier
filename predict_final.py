import full_neural_network as fnn
import os
import numpy as np
import csv
import prepare_image_data
from transform_images import original_dims, target_dims
from PIL import Image
import re

NETWORK_FILENAME = 'final_downloads/StateFarmDistractbatch_model_inetwork_2.pickle'#'batch_model_inetwork_3.pickle'
BATCH_SIZE = 50
TRANSFORMED_DIRECTORY = '/home/max/workspace/StateFarmDistract/test_transformed/'
SOURCE_DIRECTORY = '/home/max/workspace/StateFarmDistract/test/'
OUTPUT_FILENAME = 'batnet_final.csv'
ALREADY_TRANSFORMED = True

def scale_image(filename):
    new_filename = TRANSFORMED_DIRECTORY + filename
    dst_im = Image.new("RGB",target_dims,(0,0,0))
    try:
        base_image = Image.open(SOURCE_DIRECTORY + filename)
    except:
        print filename
        raise
    bi = base_image.resize(target_dims)
    dst_im.paste(bi,((0,0)))
    dst_im.save(new_filename)
    
def fix_name(text):
    return re.sub('.*/','',text)

def main():
    filenames = os.listdir(SOURCE_DIRECTORY)
    if not ALREADY_TRANSFORMED:
        for i, file in enumerate(filenames):
            scale_image(file)
            if i % 250 == 0:
                print i
        print 'scaled images'
    network = fnn.load_network_isolate(NETWORK_FILENAME, modified_batch_size = BATCH_SIZE)
    print 'now beginning learning process'
    with open(OUTPUT_FILENAME,'wb') as f:
        writer = csv.writer(f)
        writer.writerow(['img'] +['c' + str(x) for x in range(10)])
        current_batch = []
        for i, fn in enumerate(filenames):
            current_batch.append(TRANSFORMED_DIRECTORY + fn)
            if (i + 1) % BATCH_SIZE == 0:
                images = prepare_image_data.process_images(current_batch)
                results = network.predict(images,['probs'])['probs']
                for name, probs in zip(current_batch,results):
                    writer.writerow([fix_name(name)] + list(probs))
                current_batch = []
            if (i % 250 == 0):
                print i
        print 'last loop'
        if current_batch <> []:
            #recycle filenames as many times as necessary
            current_batch_length = len(current_batch)
            current_batch += [current_batch[0]] * (BATCH_SIZE - current_batch_length)
            images = prepare_image_data.process_images(current_batch)
            results = network.predict(images,['probs'])['probs']
            #shortening results
            images = images[0:current_batch_length]
            results = results[0:current_batch_length]
            for name, probs in zip(current_batch,results):
                writer.writerow([fix_name(name)] + list(probs))
    print 'done predicting!!!'

if __name__=='__main__':
    main()
