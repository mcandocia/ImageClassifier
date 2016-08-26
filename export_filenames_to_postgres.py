import csv
import psycopg2
from numpy import random
import dbinfo

train_proportion = 0.6
test_proportion = 0.2
validation_proportion = 0.2

test_threshold = train_proportion + test_proportion

MAX_BATCH_SIZE = 5000

def main():
    conn = psycopg2.connect("dbname=%s user=%s password=%s host=%s port=%s" %
(dbinfo.dbname,dbinfo.user,dbinfo.password,dbinfo.host,dbinfo.port))
    print 'connected to database'
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS sf_photos;")
    cur.execute("CREATE TABLE sf_photos(subject text, class text, filename text,role integer, random_index integer);")
    with open('driver_imgs_list.csv','rb') as f:
        reader = csv.reader(f)
        reader.next()#skips header
        data_rows = []
        i = 0
        for row in reader:
            data_rows.append(row)
            i += 1
            if i == MAX_BATCH_SIZE:
                store_data(data_rows,cur)
                data_rows = []
                i = 0
        if i <> 0:
            store_data(data_rows,cur)
        print "stored database"
    conn.commit()
    conn.close()

def rand_cat():
    x = random.random()
    if x < train_proportion:
        return [1]
    elif x < test_threshold:
        return [2]
    else:
        return [3]

maxindex = pow(2,22)
def rand_index():
    return [round(random.random() * maxindex)]

def make_path(name,category):
    return '/home/max/workspace/StateFarmDistract/train/%s/%s' % (category,name)

def store_data(data,cursor):
    ldata = [[row[0],row[1],make_path(row[2],row[1])] for row in data]
    reformatted_data = [list(row) + rand_cat() + rand_index() for row in ldata]
    argstring = ','.join(cursor.mogrify("(%s,%s,%s,%s,%s)",x) for x in reformatted_data)
    cursor.execute("INSERT INTO sf_photos VALUES " + argstring)
    print "inserted entries into db"
    
if __name__=='__main__':
    main()
