from PIL import Image
import numpy as np
import os
import dbinfo
import psycopg2
import sys
import re

new_directory = '/home/max/workspace/StateFarmDistract/train_transformed/'
regexp = re.compile('train')

#original dims and scales currently not used
original_dims = (640,480)
target_dims = (320,240)

scales = (target_dims[0]/float(original_dims[0]), target_dims[1]/float(original_dims[1]))

def main():
    conn = psycopg2.connect("dbname=%s user=%s password=%s host=%s port=%s" %
(dbinfo.dbname,dbinfo.user,dbinfo.password,dbinfo.host,dbinfo.port))
    c = conn.cursor()
    c.execute("SELECT filename FROM sf_photos;")
    if not os.path.exists(new_directory):
        print 'making new directories'
        os.makedirs(new_directory)
        for cls in ['/c' + str(x) for x in range(10)]:
            os.makedirs(new_directory + cls)
    else:
        print "Directories already exist...continuing"
    while True:
        res = c.fetchmany(1000)
        if len(res) == 0:
            break
        for fn in res:
            scale_image(fn[0])
        sys.stdout.write('.')
        sys.stdout.flush()
    print ''
    print 'finished transforming images'
    c.execute("DROP TABLE IF EXISTS sf_transformed;")
    c.execute("CREATE TABLE sf_transformed AS (SELECT * FROM sf_photos);")
    c.execute("UPDATE sf_transformed SET filename=regexp_replace(filename,'train','train_transformed');")
    conn.commit()
    print 'created sf_transformed table'
    conn.close()
    return 0

def scale_image(filename):
    new_filename = regexp.sub('train_transformed', filename)
    dst_im = Image.new("RGB",target_dims,(0,0,0))
    try:
        base_image = Image.open(filename)
    except:
        print filename
        raise
    bi = base_image.resize(target_dims)
    dst_im.paste(bi,((0,0)))
    dst_im.save(new_filename)

if __name__=='__main__':
    main()
