import time

#make time eazier to read
class Ptime():
    def __init__(self):
        self.saved_time = ""
        self.ptime = ""
    
    def set_time_now(self):
        self.saved_time = str(time.ctime())
        
    def get_origin_time(self):
        return self.saved_time
        
    def get_time(self):
        time_list = self.saved_time.split(' ')
        if '' in time_list:
            time_list.remove('')
        if(int(time_list[2]) < 10):
            time_list[2] = "0" + time_list[2]
        mask = [4, 1, 2, 0, 3]
        self.ptime = ""
        for i in mask:
            self.ptime += time_list[i] + "_"
        return self.ptime
