import os

phasing_dir=os.path.dirname(os.path.realpath(__file__))
feeds_data=os.path.join(phasing_dir,'feeds.dat')

def getNode(feed):
    with open(feeds_data,'r') as feeds_file:
        for line in feeds_file:
            if line.split()[0]==feed:
                return int(line.split()[1])
    return None

def getVolt(feed):
    with open(feeds_data,'r') as feeds_file:
        for line in feeds_file:
            if line.split()[0]==feed:
                return int(line.split()[2])
    return None
