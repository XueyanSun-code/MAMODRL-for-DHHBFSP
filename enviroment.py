import simpy

class enviroment(simpy.Environment):
    def reset_now(self,t):
        self._now=t