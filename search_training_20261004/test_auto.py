import threading
from auto_search_training import complete_branch

class Controller:
    def __init__(self):self.cycles=0;self.calls=0
    def worker(self,mode,settings):
        self.calls+=1
        self.cycles=1 if self.cycles==0 else 5
c=Controller();checks=[]
complete_branch(c,(8,2,50,64,5,1),checks.append,threading.Event())
assert c.cycles==5 and c.calls==2 and checks==[0,1,1,5]
c=Controller();c.cycles=3
complete_branch(c,(8,2,50,64,5,1),lambda cycle:None,threading.Event())
assert c.calls==1 and c.cycles==5
c=Controller()
def bad_monitor(cycle):
    if cycle==1:raise RuntimeError('monitor stop')
try:complete_branch(c,(8,2,50,64,5,1),bad_monitor,threading.Event())
except RuntimeError:pass
else:raise AssertionError('Did not stop after monitor failure')
assert c.calls==1
print('PASS: automatic first-cycle continuation, resume, monitor failure stop')
